"""Tsugi TorchInductor backend (楔の本体・ADR-003).

``torch.compile(model, backend="tsugi")`` で Tsugi 経由のカーネル生成を行う。
開発者は torch を叩くだけで NVIDIA/AMD 両対応になる。

状態: 骨格（skeleton）。実 lowering は Phase 4 で実装。
       実 GPU 検証は NVIDIA・AMD 実機が必要。未検証の経路は「未検証」と明記する。
"""
from __future__ import annotations

from typing import Any, Callable, List


def _tsugi_compile(gm: Any, example_inputs: List[Any]) -> Callable:
    """TorchDynamo から FX GraphModule を受け取り、Tsugi カーネルへ変換する。

    Phase 4 の実装計画:
      1. TorchInductor の lowering を再利用して融合機会を取得
      2. hot op (matmul/attention/norm/elementwise) を tsugi.tile IR へ変換
      3. SPEC.md §3 パイプラインでコンパイル (tsugi.tile→gpu→NVVM/ROCDL)
      4. 標準 GEMM 等は cuBLAS/rocBLAS へ escape-hatch (性能優先・R5)
      5. 残りは torch eager にフォールバック (正しさ優先)

    現状: codegen は未実装だが、FX グラフに静的検証（propagation）を走らせて警告を出す
    —— 「検証だけ先に届ける」楔の早期価値。実行は eager に素通し（嘘をつかない）。
    """
    # 検証だけ先に届ける: FX グラフを静的監査し増幅 op / モデル発散を警告（codegen 不要）。
    try:
        import warnings

        from .fxbridge import audit_fx
        # 代表 logit があればタスク影響（判断フリップ率）へ翻訳。example 出力を best-effort で利用。
        ref_logits = None
        try:
            out = gm.forward(*example_inputs)
            t = out[0] if isinstance(out, (tuple, list)) else out
            ref_logits = t.detach().cpu().numpy()
        except Exception:  # noqa: BLE001 — 取れなければ発散のみ報告
            ref_logits = None
        # 代表 *入力* も best-effort で取り出し、scale=1 / cond=1 の暗黙仮定を実測で
        # 置き換える（FEATURE-AUDIT.md A-3）。実 LLM の活性は massive activations で
        # 中央値の ~1000 倍に達しうるため、scale=1 のままでは認証 atol を桁で誤る。
        sample = None
        try:
            first = example_inputs[0] if example_inputs else None
            if first is not None:
                sample = first.detach().cpu().numpy()
        except Exception:  # noqa: BLE001 — 取れなければ従来通り静的仮定で報告
            sample = None
        rep = audit_fx(gm, ref_logits=ref_logits, sample=sample)
        # nondeterministic_ops/requires_noise_floor は audit_fx が既に計算済みだが、
        # 従来この警告メッセージに一切反映されていなかった（audit_fx の戻り値が facade
        # ＝ユーザー向け警告に届いていない・他ラウンドで見つけた facade 未接続と同型）。
        # scatter_add 等の atomicAdd 由来 op はグラフに数値 op（matmul/softmax 等）が
        # 無くても存在しうるため、n_ops==0 でも requires_noise_floor だけで警告を出す。
        if rep["n_ops"] or rep["requires_noise_floor"]:
            task = (f", task_flip_bound≤{rep['task_flip_bound'] * 100:.1f}%"
                    if rep["task_flip_bound"] is not None else "")
            dyn = " [has_dynamic_shapes: per-shape 再検証が必要]" if rep["has_dynamic_shapes"] else ""
            nondet = (f" [non-deterministic: {rep['nondeterministic_ops']} → "
                     "noise floor 実測が必須（静的許容では不十分）]"
                     if rep["requires_noise_floor"] else "")
            # 正規化層（LayerNorm/RMSNorm）はほぼ scale-invariant で、上流のスケール型
            # クロスベンダー乖離を実質的にリセットする効果を持つが、propagate() は
            # これを考慮しない（FEATURE-AUDIT.md A-5）。安全な方向（過大評価）だが、
            # ユーザーが WARN を額面通り受け取り過剰反応しないよう明示する。
            norm = (" [has_normalization: model_divergence は正規化層のscaleリセット効果を"
                   "未考慮の保守的な上界（実際の発散はこれより小さい可能性）]"
                   if rep.get("has_normalization") else "")
            # A-3: 代表入力から scale/cond を実測できたなら、その旨と外れチャネルを報告。
            # 実測できていなければ「cond=1 は下界」の但し書きを従来通り残す（暗黙化しない）。
            if rep.get("cond_measured"):
                basis = (f" [sample 実測: scale={rep['sample_scale']:.3g}・"
                         "増幅 op の cond をデータから測定済み（静的下界を解消）]")
            else:
                basis = " (cond=1 lower bound)"
            # massive activations（一部チャネルが中央値の ~1000 倍）は単一 scale 仮定を壊す。
            spread = rep.get("channel_spread")
            outlier = (f" [outlier channels: scale 広がり ×{spread:.0f} → 単一 scale 仮定が"
                       "崩れる・per-channel 検証を検討]"
                       if spread is not None and spread >= 10.0 else "")
            warnings.warn(
                f"[tsugi] verification-only (no codegen yet): {rep['n_ops']} numeric ops, "
                f"amplifiers={rep['amplifiers']}, model_divergence≈{rep['model_divergence']:.2e}"
                f"{task}{dyn}{nondet}{norm}{basis}{outlier}. "
                "cross-vendor 等価性は実機で audit_cross_vendor を。",
                stacklevel=2)
    except Exception:  # noqa: BLE001 — 検証は best-effort・実行を壊さない
        pass

    def _forward(*args: Any, **kwargs: Any) -> Any:
        # 未実装のため eager に委譲。性能利得なし（明示）。
        return gm.forward(*args, **kwargs)

    return _forward


_BACKEND_REGISTERED: bool = False  # 冪等ガード: 二重 import による重複登録を防ぐ


def register() -> None:
    """backend="tsugi" を torch に登録する（冪等）。

    二重 import / reload でも安全: 一度登録済みなら即 return。
    torch._dynamo.register_backend は既登録名で再呼出しするとエラーになるベンダーがあるため
    module-level フラグで guard する（torch.list_backends() より安定）。
    """
    global _BACKEND_REGISTERED
    if _BACKEND_REGISTERED:
        return
    try:
        from torch._dynamo import register_backend
    except ImportError as exc:  # torch 未導入環境
        raise RuntimeError(
            "Tsugi torch backend requires PyTorch with TorchDynamo"
        ) from exc

    register_backend(name="tsugi", compiler_fn=_tsugi_compile)
    _BACKEND_REGISTERED = True


# import 時に自動登録（torch があれば）
try:
    register()
except Exception:  # noqa: BLE001 — torch 無し環境では沈黙（ライブラリとして壊さない）
    pass
