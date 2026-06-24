"""FX→Tsugi 検証橋のテスト（SOCRATIC Q23/Q26）。

torch を使わず duck-typed の stand-in FX グラフで aten op→論理 op の写像と audit_fx を検証。
実 torch.fx との結線は torch 環境が要る（本環境では未検証・主張と実装の一致）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi_torch.fxbridge import audit_fx, fx_to_graph_ops  # noqa: E402


class _TM:
    def __init__(self, shape):
        self.shape = shape


class _Node:
    def __init__(self, op, target, shape=None):
        self.op = op
        self.target = target
        self.meta = {"tensor_meta": _TM(shape)} if shape else {}


class _Graph:
    def __init__(self, nodes):
        self.nodes = nodes


class _GM:
    """torch.fx.GraphModule の最小 duck-type（.graph.nodes）。"""
    def __init__(self, nodes):
        self.graph = _Graph(nodes)


def _transformer_block():
    # 代表的な attention+MLP ブロックの aten 列を模す
    return _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),       # qkv proj → matmul
        _Node("call_function", "aten.bmm.default", (8, 64)),          # attn scores → matmul
        _Node("call_function", "aten._softmax.default"),              # softmax（増幅）
        _Node("call_function", "aten.bmm.default", (8, 64)),          # attn·V → matmul
        _Node("call_function", "aten.mean.dim"),                      # layernorm 統計 → reduce
        _Node("call_function", "aten.mul.Tensor"),                    # scale（elementwise）
        _Node("call_function", "aten.addmm.default", (8, 2048)),      # MLP → matmul
        _Node("call_function", "aten.gelu.default"),                  # activation → scale
        _Node("get_attr", "weight"),                                  # 除外される
        _Node("output", "output"),                                    # 除外される
    ])


def test_fx_maps_aten_to_logical_ops():
    ops = fx_to_graph_ops(_transformer_block())
    kinds = [o.kind for o in ops]
    assert kinds == ["matmul", "matmul", "softmax", "matmul", "reduce", "scale",
                     "matmul", "scale"]
    # placeholder/get_attr/output は数値 op でないので除外
    assert all(k in ("matmul", "softmax", "reduce", "scale") for k in kinds)


def test_fx_matmul_K_from_shape_meta():
    ops = fx_to_graph_ops(_transformer_block())
    matmuls = [o for o in ops if o.kind == "matmul"]
    assert matmuls[0].K == 512        # addmm 出力末尾次元
    assert matmuls[-1].K == 2048      # MLP 出力末尾次元


def test_audit_fx_surfaces_amplifiers():
    rep = audit_fx(_transformer_block())
    assert rep["n_ops"] == 8
    assert "softmax" in rep["amplifiers"] and "reduce" in rep["amplifiers"]
    assert rep["model_divergence"] > 0.0


def test_audit_fx_empty_graph():
    rep = audit_fx(_GM([_Node("placeholder", "x"), _Node("output", "output")]))
    assert rep["n_ops"] == 0
    assert rep["model_divergence"] == 0.0


def test_audit_fx_translates_to_task_flip_bound():
    # ref_logits を渡すと静的グラフ発散 → タスク判断フリップ率上界に翻訳（静的→タスク）
    import numpy as np
    rep_none = audit_fx(_transformer_block())
    assert rep_none["task_flip_bound"] is None       # logit 無しなら None
    logits = np.random.default_rng(0).standard_normal((1000, 256)).astype(np.float32)
    rep = audit_fx(_transformer_block(), ref_logits=logits)
    assert rep["task_flip_bound"] is not None
    assert 0.0 <= rep["task_flip_bound"] <= 1.0      # 確率（上界）


def test_audit_fx_flags_nondeterministic_atomic_ops():
    # scatter_add 等 atomicAdd 由来の非決定 op を検出し noise floor 実測必須を宣言
    # （PyTorch 公式: https://pytorch.org/docs/stable/notes/randomness.html）
    gm = _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),
        _Node("call_function", "aten.scatter_add.default"),    # 非決定（forward atomicAdd）
        _Node("call_function", "aten._softmax.default"),
        _Node("output", "output"),
    ])
    rep = audit_fx(gm)
    assert rep["requires_noise_floor"], "scatter_add を含むのに noise floor 不要扱い"
    assert any("scatter_add" in n for n in rep["nondeterministic_ops"])

    # 決定論的グラフ（matmul/softmax のみ）は noise floor 不要
    det = audit_fx(_transformer_block())
    assert not det["requires_noise_floor"]
    assert det["nondeterministic_ops"] == []


def main() -> int:
    ok = True
    tests = [
        test_fx_maps_aten_to_logical_ops,
        test_fx_matmul_K_from_shape_meta,
        test_audit_fx_surfaces_amplifiers,
        test_audit_fx_empty_graph,
        test_audit_fx_translates_to_task_flip_bound,
        test_audit_fx_flags_nondeterministic_atomic_ops,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
