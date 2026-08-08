"""FX グラフ → Tsugi 検証層の橋（torch.compile で「検証だけ先に届ける」）。

GPU codegen 完成前でも、torch.compile(model, backend="tsugi") の FX グラフに対し
静的検証（propagation/増幅 op の可視化）を走らせて警告を出せる —— これが楔の早期価値。

torch を import せずに動く（FX ノードは duck-typed で読む）。aten op 名 → 論理 op の
写像は stand-in グラフで検証済み。**実 torch.fx との結線は torch 環境が要る（本環境では
実 FX に対しては未検証）**＝主張と実装の一致。
"""
from __future__ import annotations

from typing import Any

from tsugi.nondeterminism import classify_nondeterminism
from tsugi.propagation import GraphOp, is_amplifier, propagate

_CALL_OPS = ("call_function", "call_method", "call_module")


def _kind_of(target_name: str) -> str | None:
    """aten/torch op 名を Tsugi の論理 op 種別へ写す（None=数値発散に無関係）。"""
    t = target_name.lower()
    if any(k in t for k in ("addmm", "mm", "matmul", "bmm", "linear", "einsum", "conv")):
        return "matmul"
    if "softmax" in t:
        return "softmax"
    if "rsqrt" in t:
        return "rsqrt"
    if "exp" in t:
        return "exp"
    if any(k in t for k in ("mean", "sum", "var", "layer_norm", "rms_norm", "_norm")):
        return "reduce"
    if any(k in t for k in ("add", "sub", "mul", "div", "tanh", "gelu",
                            "relu", "sigmoid", "cast", "to_copy", "scale", "neg")):
        return "scale"
    return None   # placeholder/output/get_attr/view 等は除外


def _node_K(node: Any, default: int = 512) -> int:
    """ノードの出力 shape 末尾次元を K の目安に（shape meta が無ければ既定）。"""
    meta = getattr(node, "meta", None) or {}
    tm = meta.get("tensor_meta") if isinstance(meta, dict) else None
    shape = getattr(tm, "shape", None) if tm is not None else None
    if shape:
        try:
            return int(shape[-1])
        except (TypeError, ValueError, IndexError):
            return default
    return default


def _node_is_symbolic(node: Any) -> bool:
    """shape meta に symbolic 次元（torch.SymInt）が含まれるかを検査する。

    torch.compile(dynamic=True) や export 経路では、次元が具体的な int でなく
    torch.SymInt になる。SymInt は int() 変換で TypeError/ValueError を送出するため、
    その失敗で symbolic を判定する。

    shape guard 効果: symbolic shape があると torch.compile は実行時形状ごとに
    ガードを立て、ガード違反で再コンパイルを行う。形状別特化カーネルはタイル幅・
    縮約順序・アキュムレータ幅が変わり得るため、等価性は 1 形状のみ認証では不十分。
    実際の運用形状をカバーする per-shape 検証が必要。
    """
    meta = getattr(node, "meta", None) or {}
    tm = meta.get("tensor_meta") if isinstance(meta, dict) else None
    shape = getattr(tm, "shape", None) if tm is not None else None
    if not shape:
        return False
    for dim in shape:
        try:
            int(dim)
        except (TypeError, ValueError):
            return True
    return False


def fx_to_graph_ops(gm: Any) -> list[GraphOp]:
    """FX GraphModule を propagation 用の論理 op 列へ写す（duck-typed・torch 不要）。"""
    ops: list[GraphOp] = []
    for node in gm.graph.nodes:
        if getattr(node, "op", None) not in _CALL_OPS:
            continue
        kind = _kind_of(str(getattr(node, "target", "")))
        if kind is None:
            continue
        ops.append(GraphOp(kind, K=_node_K(node)) if kind == "matmul" else GraphOp(kind))
    return ops


def _is_normalization(target_name: str) -> bool:
    """op 名が LayerNorm/RMSNorm 系（正規化）かを判定する。

    正規化層は数学的にほぼ scale-invariant（LN(c·x)≈LN(x)・c は正のスカラー）。
    上流で蓄積した *スケール型* のクロスベンダー乖離を実質的にリセットする効果を持つが、
    propagation.propagate() の現行モデルはこれを考慮せず reduce と同じ増幅則を適用する
    （FEATURE-AUDIT.md A-5）。安全な方向の近似だが恣意的な減衰係数を検証なしに導入する
    リスクを避けるため、まずは正規化層の存在を可視化するに留める
    （model_divergence は正規化層が多いモデルほど保守的な上界＝実際より緩めに
    見積もっている可能性が高いという事実を隠さない）。
    """
    t = target_name.lower()
    return "layer_norm" in t or "rms_norm" in t or "_norm" in t


def fx_call_target_names(gm: Any) -> list[str]:
    """FX グラフの呼び出しノードの raw target 名を列挙する（非決定 op 照合用）。

    _kind_of は scatter_add/index_add 等を論理 op に畳まないため、生 target 名を別途
    取り出して nondeterminism カタログに照合する（atomicAdd 由来の非決定検出）。
    """
    names: list[str] = []
    for node in gm.graph.nodes:
        if getattr(node, "op", None) in _CALL_OPS:
            names.append(str(getattr(node, "target", "")))
    return names


def audit_fx(gm: Any, ref_logits=None, sample=None) -> dict:
    """FX グラフに静的検証（propagation）を走らせ、要点を dict で返す。

    codegen 前でも「このモデルはクロスベンダーでどれだけ発散しうるか・どの増幅 op が
    あるか」を告げる。cond は静的不明ゆえ既定 1（=下界・実機/実データで定量化すべき）。

    ref_logits（代表的な出力 logit 分布）を渡すと、モデル発散を *タスク影響* に翻訳し
    判断フリップ率の上界 `task_flip_bound` を返す（静的グラフ → ユーザーに見える差）。

    sample（代表的な *入力* テンソル）を渡すと「scale=1 / cond=1」の暗黙仮定を実測で
    置き換える（FEATURE-AUDIT.md A-3）。`audit()` は B-1a/B-1b で既にこれを持っていたが
    torch 経路には無く、実 LLM では致命的だった——massive activations は中央値の
    ~1000 倍に達し（docs/SOURCES.md）、scale=1 仮定は認証 atol を桁で誤らせる。
    返り値に `sample_scale`（実 RMS）・`channel_spread`（外れチャネル検出）・
    `cond_measured` を追加し、増幅 op の cond を実測値で置き換えて発散を再計算する。
    """
    ops = fx_to_graph_ops(gm)
    rep = propagate(ops)
    amps = sorted({o.kind for o in ops if is_amplifier(o.kind)})
    # atomicAdd 由来の非決定 op を静的に検出（PyTorch 公式カタログ照合）。
    # これらがあれば静的許容では不十分で、実機 noise floor 実測が必須。
    nondet = classify_nondeterminism(fx_call_target_names(gm))
    # dynamic shape 検出: torch.compile(dynamic=True) や export 経路では shape が
    # torch.SymInt になる。形状ごとにカーネルが特化されるため、
    # 1 形状で認証した等価性は他の形状に転用できない（per-shape 再検証が必要）。
    has_dynamic_shapes = any(
        _node_is_symbolic(node)
        for node in gm.graph.nodes
        if getattr(node, "op", None) in _CALL_OPS
    )
    # 正規化層（LayerNorm/RMSNorm）の有無を検出する。propagate() はこれらを scale-invariant
    # と扱わず通常の reduce と同じ増幅則を適用するため、正規化層があるモデルでは
    # model_divergence が実際より保守的（過大）な上界になっている可能性が高い
    # （FEATURE-AUDIT.md A-5・fail-safe の安全な方向だが、隠さず明示する）。
    has_normalization = any(
        _is_normalization(str(getattr(node, "target", "")))
        for node in gm.graph.nodes
        if getattr(node, "op", None) in _CALL_OPS
    )
    out = {
        "n_ops": len(ops),
        "model_divergence": rep.model_divergence,
        "naive_sum": rep.naive_sum,
        "amplifiers": amps,
        "dominant": rep.dominant.kind if rep.dominant is not None else None,
        "nondeterministic_ops": list(nondet.nondet_ops),
        "requires_noise_floor": nondet.requires_noise_floor,
        "has_dynamic_shapes": has_dynamic_shapes,
        "has_normalization": has_normalization,
        "task_flip_bound": None,
    }
    if sample is not None:
        # 代表入力があれば「scale=1 / cond=1」の暗黙仮定を実測で置き換える
        # （`audit()` が B-1a/B-1b で既に持っていた機能。torch 経路には無かった＝A-3）。
        import numpy as _np

        from tsugi.envelope import channel_scale_spread
        from tsugi.propagation import empirical_cond
        x = _np.asarray(sample, dtype=_np.float64)
        if x.size:
            out["sample_scale"] = float(_np.sqrt(_np.mean(x ** 2)))
            # massive activations（一部チャネルが中央値の ~1000 倍）の検出。
            # 実 LLM 活性はこの型の外れ値を持ち、単一 scale 仮定が破れる
            # （docs/SOURCES.md「outlier feature / massive activations」節）。
            out["channel_spread"] = channel_scale_spread(x)
            # データ依存 cond を実測して伝播をやり直す（静的 cond=1 は *下界*）。
            measured = False
            for o in ops:
                if is_amplifier(o.kind) and o.cond == 1.0:
                    o.cond = empirical_cond(x, o.kind)
                    measured = True
            if measured:
                rep = propagate(ops)
                out["model_divergence"] = rep.model_divergence
                out["naive_sum"] = rep.naive_sum
                out["dominant"] = rep.dominant.kind if rep.dominant is not None else None
            out["cond_measured"] = measured
    if ref_logits is not None:
        import numpy as _np
        from tsugi.decision import flip_bound_from_divergence
        out["task_flip_bound"] = flip_bound_from_divergence(ref_logits, rep.model_divergence)
        # 実 logit の RMS scale を測定し certify_from_sample の代替 scale として公開する。
        # audit_fx を呼んだ後に certify_from_sample(x, K, dtype) へ渡す目安になる。
        _rf = _np.asarray(ref_logits, dtype=_np.float64)
        out["ref_scale"] = float(_np.sqrt(_np.mean(_rf ** 2))) if _rf.size else 1.0
    return out
