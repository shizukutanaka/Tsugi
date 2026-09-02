"""Tsugi TorchInductor backend (楔の本体・ADR-003).

``torch.compile(model, backend="tsugi")`` で Tsugi 経由のカーネル生成を行う。
開発者は torch を叩くだけで NVIDIA/AMD 両対応になる。

状態: 骨格（skeleton）。実 lowering は Phase 4 で実装。
       実 GPU 検証は NVIDIA・AMD 実機が必要。未検証の経路は「未検証」と明記する。
"""
from __future__ import annotations

from typing import Any, Callable, List


def _activation_input(example_inputs: List[Any]):
    """dynamo の `example_inputs` から **活性**（本物の入力）を選ぶ。

    第 62 回の発見: `example_inputs[0]` を代表入力として使っていたが、dynamo は
    **重みを引数へ持ち上げる**ので先頭は `nn.Parameter` であることが多い。実際
    `Sequential(Linear, LayerNorm, Softmax)` では

        [Parameter(64,64), Parameter(64), Tensor(256,64), Parameter(64), Parameter(64)]

    となり、A-3 で導入した「sample 実測: scale=…」は**重み行列の統計を活性の統計として
    報告していた**（scale=0.0718 は重みのスケール）。実測と称して別のものを測るのは、
    静的仮定を残すより悪い——利用者はそれを活性の実測だと読む。

    活性は素の `Tensor`、持ち上げられた重みは `nn.Parameter` なので型で分けられる
    （名前の綴り `L_args_0_` に頼らない——dynamo のバージョンで変わる）。
    見つからなければ **None**（重みで代用しない）。
    """
    try:
        from torch.nn import Parameter
    except Exception:                       # noqa: BLE001 — torch 無し
        return None
    for t in example_inputs or ():
        if isinstance(t, Parameter):
            continue
        detach = getattr(t, "detach", None)
        if detach is None:
            continue
        return detach().cpu().numpy()
    return None


def _sim_inputs(example_inputs: List[Any], max_rows: int = 256) -> list:
    """模倣へ渡す引数列（グラフの引数と同順・同数）を作る。

    行数を切るのは活性だけ——重みを切ると行列積の形が壊れる。持ち上げられた重みは
    `nn.Parameter` なので型で見分ける（`_activation_input` と同じ規律）。
    """
    try:
        from torch.nn import Parameter
    except Exception:                       # noqa: BLE001 — torch 無し
        return []
    out = []
    for t in example_inputs or ():
        detach = getattr(t, "detach", None)
        if detach is None:
            return []                       # 1 本でも配列化できなければ位置対応が壊れる
        a = detach().cpu().numpy()
        if not isinstance(t, Parameter) and getattr(a, "ndim", 0) >= 1:
            a = a[:max_rows]
        out.append(a)
    return out


def _tsugi_compile(gm: Any, example_inputs: List[Any]) -> Callable:
    """TorchDynamo から FX GraphModule を受け取り、Tsugi カーネルへ変換する。

    実装状況（正直な線引き・第 60 回時点）:
      - ✅ FX グラフの静的監査（propagation・非決定 op・dynamic shape・タスク影響）
      - ✅ **FX → Tsugi IR 降下 → 実 PTX/AMDGCN 生成 → ベンダーのアセンブラで検証**
        （`fxlower` + `codegen`。GPU 不要・L2 まで）
      - ❌ 生成した機械語の**実行**（要実機・L3）。よって実行は eager に素通しする
      - ❌ 融合・escape-hatch（cuBLAS/rocBLAS 委譲）・autotuning

    実行を eager に委ねるのは「嘘をつかない」ため——生成物は L2 までしか検証されて
    おらず、走らせて正しい保証が無い。**検証は今届き、実行は実機が来てから**。
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
            sample = _activation_input(example_inputs)
        except Exception:  # noqa: BLE001 — 取れなければ従来通り静的仮定で報告
            sample = None
        rep = audit_fx(gm, ref_logits=ref_logits, sample=sample)
        # nondeterministic_ops/requires_noise_floor は audit_fx が既に計算済みだが、
        # 従来この警告メッセージに一切反映されていなかった（audit_fx の戻り値が facade
        # ＝ユーザー向け警告に届いていない・他ラウンドで見つけた facade 未接続と同型）。
        # scatter_add 等の atomicAdd 由来 op はグラフに数値 op（matmul/softmax 等）が
        # 無くても存在しうるため、n_ops==0 でも requires_noise_floor だけで警告を出す。
        if rep["n_ops"] or rep["requires_noise_floor"]:
            # 天井（静的伝播）でなく **実測** を第一に出す（第 62 回）。天井は実測の
            # 100〜1000 倍になりうるので、これを "task_flip_bound" として単独で見せると
            # 毎回「≤ 40〜80%」という無情報な警告になり、読む人が全体を無視する。
            task = (f", task_flip_bound≤{rep['task_flip_bound'] * 100:.1f}%（天井・予測ではない）"
                    if rep["task_flip_bound"] is not None else "")
            try:
                from .simulate import refusal_reason, simulate_cross_vendor
                # dynamo は重みを引数へ持ち上げるので、**全引数を順に**渡す。1 本だけ
                # 渡すと束縛が一意に決まらず（第 62 回）、諦めるか誤った実測になる。
                _ins = _sim_inputs(example_inputs)
                _sim = simulate_cross_vendor(gm, _ins) if _ins else None
                _w = _sim.worst if _sim is not None else None
                if _w is not None and _w.n:
                    task = (f", 実測フリップ {_w.flip_rate * 100:.3f}%"
                            f"（上界 {_w.flip_rate_ub * 100:.3f}%・最悪クラス {_w.name}・"
                            f"n={_w.n}・CPU 2 ベンダー模倣＝実機発散の下界）" + task)
                elif _ins:
                    # 走らなかったことを黙らない（天井だけが残るとは利用者に見えない）
                    task += f"（実測は未取得: {refusal_reason(gm, _ins)}）"
            except Exception:  # noqa: BLE001 — 模倣は best-effort・警告は出し続ける
                pass
            dyn = " [has_dynamic_shapes: per-shape 再検証が必要]" if rep["has_dynamic_shapes"] else ""
            nondet = (f" [non-deterministic: {rep['nondeterministic_ops']} → "
                     "noise floor 実測が必須（静的許容では不十分）]"
                     if rep["requires_noise_floor"] else "")
            # 正規化層の扱い（A-5 の数値実験で当初想定が反転）: 旧警告は「実際の発散は
            # これより小さい可能性」と無条件に主張していたが、LayerNorm は平均優勢入力
            # （μ/RMS→1）で相対発散を amp≈RMS/σ に *増幅* する——旧文言自体が偽OK を
            # 誘導する未検証主張だったため撤回。RMSNorm のみ無条件安定（amp=1）。
            norm = (" [has_normalization: RMSNorm は scale 中立（amp=1・実測検証済み）。"
                   "LayerNorm は平均優勢入力で相対発散を amp≈RMS/σ に増幅しうる"
                   "（sample 指定時は実測 cond に反映済み）]"
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
            # codegen: 楔ユーザーにも「単一ソース → 両ベンダーの実機械語」を届ける。
        # 失敗しても実行は壊さない（best-effort・警告は出続ける）。
        codegen_note = ""
        try:
            from tsugi import codegen as _cg

            from .fxlower import fx_to_ir
            _lm = fx_to_ir(gm)
            _ok = []
            for _t in _cg.TARGETS:
                _asm = _cg.verify_codegen(_lm.module, target=_t)[1]
                if _asm.available and _asm.ok:
                    _ok.append(_t)
            codegen_note = (
                f" codegen: 呼び出し {_lm.report.n_calls} 件中 "
                f"{len(_lm.report.covered)} を IR へ降下し "
                f"{len(_ok)}/{len(_cg.TARGETS)} ターゲットでアセンブル検証"
                + ("（**partial**: 表せない op "
                   f"{sorted(set(_lm.report.unsupported))} があるため生成物はモデル"
                   "全体ではない）" if _lm.report.partial else "")
                + "。実行は未検証（要実機）")
        except Exception:  # noqa: BLE001
            codegen_note = " codegen: 降下できず（静的監査のみ）"

        warnings.warn(
                f"[tsugi] verification-only (codegen は L2 まで検証済み・実行は "
                f"eager 素通し): {rep['n_ops']} numeric ops, "
                f"amplifiers={rep['amplifiers']}, model_divergence≈{rep['model_divergence']:.2e}"
                f"{task}{dyn}{nondet}{norm}{basis}{outlier}. "
                f"{codegen_note}. "
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
