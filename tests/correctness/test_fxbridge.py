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


class _SymInt:
    """torch.SymInt の最小 duck-type — int() で失敗する symbolic 次元。

    torch.compile(dynamic=True) や torch.export で生成される symbolic 次元を模す。
    int() が TypeError を送出することで _node_is_symbolic が dynamic を判定できる。
    """
    def __repr__(self) -> str:
        return "s0"

    def __int__(self) -> int:
        raise TypeError("symbolic SymInt — cannot convert to int")


def test_audit_fx_ref_scale_from_logits():
    """ref_logits を渡すと ref_scale（RMS）が audit 出力に含まれる（Q14: scale 推定）。

    certify_from_sample(x, K, dtype) に渡す scale ヒントになる。
    scale=1 仮定でモデルを認証すると、実 logit scale（例: 数十）との乖離で
    check_tensor が scale 超過 BLOCK を誤発火する。ref_scale はその乖離を事前に示す。
    """
    import numpy as np

    gm = _transformer_block()

    # ref_logits なしでは ref_scale は含まれない
    rep_no_logits = audit_fx(gm)
    assert "ref_scale" not in rep_no_logits, "logits 無しで ref_scale が含まれている"

    # scale ≈ 10 の logit（LLM の未正規化出力に近い）
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((500, 128)).astype(np.float32) * 10.0
    rep = audit_fx(gm, ref_logits=logits)
    assert "ref_scale" in rep, "logits 有りなのに ref_scale が含まれていない"
    actual_rms = float(np.sqrt(np.mean(logits ** 2)))
    assert abs(rep["ref_scale"] - actual_rms) / actual_rms < 0.01, \
        f"ref_scale={rep['ref_scale']:.2f} が実 RMS={actual_rms:.2f} と乖離"
    assert rep["task_flip_bound"] is not None, "ref_logits 有りなのに task_flip_bound=None"


def test_audit_fx_warns_dynamic_shapes():
    """dynamic=True コンパイルの symbolic shape を検出し has_dynamic_shapes=True を返す。

    torch.compile shape guard の研究知見（2025）:
    - shape guard は形状ごとにカーネルを特化する（タイル幅・縮約順序・アキュムレータ幅）
    - 特化カーネルの数値特性は形状依存 → 1 形状で認証した等価性は他形状に転用不可
    - has_dynamic_shapes=True は「per-shape 再検証が必要」のシグナル
    """
    dynamic_gm = _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (_SymInt(), _SymInt())),
        _Node("call_function", "aten._softmax.default"),
        _Node("output", "output"),
    ])
    rep = audit_fx(dynamic_gm)
    assert rep["has_dynamic_shapes"], "symbolic shape を dynamic と判定できていない"

    # 静的形状グラフ（int 次元）は dynamic でない
    rep_static = audit_fx(_transformer_block())
    assert not rep_static["has_dynamic_shapes"], "静的形状グラフが dynamic と誤判定された"

    # 形状なしノード（meta 無し）は dynamic 扱いしない
    no_meta_gm = _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default"),   # shape=None → meta なし
        _Node("output", "output"),
    ])
    assert not audit_fx(no_meta_gm)["has_dynamic_shapes"], \
        "shape meta なし → dynamic と誤判定（既定の int 扱いが期待値）"


def test_audit_fx_detects_normalization_layers():
    """LayerNorm/RMSNorm 系の op を検出し has_normalization=True を返す（FEATURE-AUDIT.md A-5）。

    正規化層はほぼ scale-invariant（LN(c·x)≈LN(x)）で、上流のスケール型クロスベンダー
    乖離を実質的にリセットする効果を持つが、propagation.propagate() はこれを考慮せず
    通常の reduce と同じ増幅則を適用する。恣意的な減衰係数を未検証のまま導入するのは
    危険（過大な dilution は偽OK の温床になりうる）なので、まずは正規化層の存在を
    可視化するに留める——has_normalization=True は「model_divergence はこの効果を
    未考慮の保守的な上界（実際より緩め）」というシグナル。
    """
    norm_gm = _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),
        _Node("call_function", "aten.native_layer_norm.default"),
        _Node("output", "output"),
    ])
    assert audit_fx(norm_gm)["has_normalization"]

    rms_gm = _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),
        _Node("call_function", "aten._rms_norm.default"),
        _Node("output", "output"),
    ])
    assert audit_fx(rms_gm)["has_normalization"]

    # 正規化層が無いグラフは False のまま（既存の _transformer_block は norm を含まない）
    rep_no_norm = audit_fx(_transformer_block())
    assert not rep_no_norm["has_normalization"]


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
        test_audit_fx_ref_scale_from_logits,
        test_audit_fx_warns_dynamic_shapes,
        test_audit_fx_detects_normalization_layers,
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
