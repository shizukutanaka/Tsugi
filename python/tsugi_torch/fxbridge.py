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


def audit_fx(gm: Any, ref_logits=None) -> dict:
    """FX グラフに静的検証（propagation）を走らせ、要点を dict で返す。

    codegen 前でも「このモデルはクロスベンダーでどれだけ発散しうるか・どの増幅 op が
    あるか」を告げる。cond は静的不明ゆえ既定 1（=下界・実機/実データで定量化すべき）。

    ref_logits（代表的な出力 logit 分布）を渡すと、モデル発散を *タスク影響* に翻訳し
    判断フリップ率の上界 `task_flip_bound` を返す（静的グラフ → ユーザーに見える差）。
    """
    ops = fx_to_graph_ops(gm)
    rep = propagate(ops)
    amps = sorted({o.kind for o in ops if is_amplifier(o.kind)})
    # atomicAdd 由来の非決定 op を静的に検出（PyTorch 公式カタログ照合）。
    # これらがあれば静的許容では不十分で、実機 noise floor 実測が必須。
    nondet = classify_nondeterminism(fx_call_target_names(gm))
    out = {
        "n_ops": len(ops),
        "model_divergence": rep.model_divergence,
        "naive_sum": rep.naive_sum,
        "amplifiers": amps,
        "dominant": rep.dominant.kind if rep.dominant is not None else None,
        "nondeterministic_ops": list(nondet.nondet_ops),
        "requires_noise_floor": nondet.requires_noise_floor,
        "task_flip_bound": None,
    }
    if ref_logits is not None:
        from tsugi.decision import flip_bound_from_divergence
        out["task_flip_bound"] = flip_bound_from_divergence(ref_logits, rep.model_divergence)
    return out
