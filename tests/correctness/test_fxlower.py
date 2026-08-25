"""FX → Tsugi IR 降下のテスト——**想定ユーザーの経路が機械語まで届くか**。

2 段構え:
  1. stand-in FX グラフ（torch 不要・常に実行）— 構造と正直さの契約
  2. **実 torch.fx**（torch があれば実行・無ければ正直に skip）— 実 FX との結線と、
     降下した IR を NumPy で評価して **torch eager と意味論を照合**する

(2) が本スイートの核心。分解（softmax/layer_norm/gelu…）が正しいかは、
アセンブラにも LLVM にも問えない——**実行して突き合わせる以外に方法がない**。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi import codegen as cg  # noqa: E402
from tsugi.interp import evaluate  # noqa: E402
from tsugi_torch.fxbridge import audit_fx, fx_to_graph_ops  # noqa: E402
from tsugi_torch.fxlower import fx_to_ir  # noqa: E402

try:
    import torch
    import torch.fx
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


# --- stand-in FX グラフ（torch 非依存・duck-typed） -------------------------

@dataclass(eq=False)
class N:
    op: str
    target: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


@dataclass
class G:
    nodes: list


class GM:
    def __init__(self, ns):
        self.graph = G(ns)


def _standin(*targets: str) -> GM:
    return GM([N("placeholder", "x"),
               *[N("call_function", t) for t in targets],
               N("output", "output")])


def test_lowering_produces_ir_that_both_vendors_assemble():
    """FX → IR → 実機械語。楔ユーザーが tile-DSL 経路と同じものを受け取る。"""
    lm = fx_to_ir(_standin("aten.addmm.default", "aten.native_layer_norm.default",
                           "aten._softmax.default"))
    assert lm.module.op_kinds(), "IR が空"
    assert not lm.report.partial, lm.report.unsupported
    for t in cg.TARGETS:
        em, asm = cg.verify_codegen(lm.module, target=t)
        if cg.toolchain(t) is None:
            assert asm.ok is None
            continue
        assert asm.ok is True, f"{t}: {asm.stderr[:300]}"
        assert cg.verify_encoding(lm.module, target=t).ok is True


def test_unrepresentable_ops_make_the_module_partial_not_silently_dropped():
    """表せない op を黙って落とすと偽OK になる。partial と言えること。"""
    lm = fx_to_ir(_standin("aten.addmm.default", "aten._log_softmax.default",
                           "aten.special_bessel_j0.default"))
    assert lm.report.partial
    assert any("log_softmax" in u for u in lm.report.unsupported)
    assert any("bessel" in u for u in lm.report.unsupported)
    txt = "\n".join(lm.report.to_lines())
    assert "partial" in txt and "過小評価" in txt


def test_exact_gelu_is_refused_rather_than_approximated():
    """erf 版 gelu を tanh 近似で黙って置換しない（実測 max|Δ|≈4.6e-4）。

    等価性を検証する道具が、検証対象を別の関数に差し替えるのは偽OK そのもの。
    tanh 版と明示されているときだけ分解する。
    """
    exact = fx_to_ir(_standin("aten.gelu.default"))
    assert exact.report.partial, "erf 版 gelu が黙って通ってしまった"
    assert any("erf" in u for u in exact.report.unsupported)
    tanh = fx_to_ir(GM([N("placeholder", "x"),
                        N("call_function", "aten.gelu.default",
                          kwargs={"approximate": "tanh"}),
                        N("output", "output")]))
    assert not tanh.report.partial
    assert any("tanh" in d for d in tanh.report.decomposed)


def test_shape_only_ops_are_recorded_as_ignored():
    """view/permute はデータ移動を伴うが本 IR は形状を持たない。黙って通さない。"""
    lm = fx_to_ir(_standin("aten.view.default", "aten.permute.default",
                           "aten.mul.Tensor"))
    assert lm.report.shape_only
    assert "形状のみ" in "\n".join(lm.report.to_lines())


def test_unspecified_eps_is_declared_as_an_assumption():
    """eps を静的に決められないときは仮定したと言う（黙って数値を選ばない）。"""
    lm = fx_to_ir(_standin("aten.native_layer_norm.default"))
    assert lm.report.assumptions
    assert any("eps" in a for a in lm.report.assumptions)
    # 明示されていれば仮定は立たない
    given = fx_to_ir(GM([N("placeholder", "x"),
                         N("call_function", "aten.native_layer_norm.default",
                           kwargs={"eps": 1e-6}),
                         N("output", "output")]))
    assert not given.report.assumptions


# --- 実 torch.fx（あれば） --------------------------------------------------

def test_call_module_targets_resolve_or_the_whole_audit_is_a_false_ok():
    """`call_module` の target は "0"/"1" という経路名で op の種類を表さない。

    解決しないと `_kind_of` が全ノードで None を返し、`audit_fx` は
    「0 numeric ops・発散 0」を報告する——**想定ユーザーのモデルが必ず無害判定になる**。
    実際にそうなっていた（stand-in は aten 名を使うため露見しなかった）。回帰固定。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 実 FX に対する結線は未検証（正直に skip）")
        return
    m = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.Softmax(-1))
    gm = torch.fx.symbolic_trace(m)
    assert all(n.op != "call_function" for n in gm.graph.nodes), \
        "この回帰テストは call_module 経路を突く前提"
    rep = audit_fx(gm)
    assert rep["n_ops"] >= 3, f"実 FX で {rep['n_ops']} ops しか見えていない（偽OK）"
    assert rep["model_divergence"] > 0.0
    assert rep["has_normalization"] is True
    assert "softmax" in rep["amplifiers"] and "layer_norm" in rep["amplifiers"]
    assert len(fx_to_graph_ops(gm)) >= 3


def test_lowered_ir_means_the_same_thing_as_the_model():
    """降下した IR を評価し **torch eager と突き合わせる**（意味論の照合）。

    分解が正しいかはアセンブラにも LLVM にも問えない。実行して比べるしかない。
    この照合で実際に 2 件の欠陥が出た: layer_norm の eps 欠落（max|Δ|≈1.9e-5）と、
    erf 版 gelu の無断置換（max|Δ|≈4.6e-4）。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 意味論照合は未実施（正直に skip）")
        return
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=torch.float64)
    cases = {
        "Softmax": nn.Softmax(-1),
        "LayerNorm": nn.LayerNorm(16, elementwise_affine=False),
        "RMSNorm": nn.RMSNorm(16, eps=1e-6, elementwise_affine=False),
        "GELU(tanh)": nn.GELU(approximate="tanh"),
        "SiLU": nn.SiLU(),
        "Tanh": nn.Tanh(),
        "Sigmoid": nn.Sigmoid(),
        "ReLU": nn.ReLU(),
    }
    for name, mod in cases.items():
        mod = mod.double()
        ref = mod(x).detach().numpy()
        lm = fx_to_ir(torch.fx.symbolic_trace(mod))
        assert not lm.report.partial, f"{name}: {lm.report.unsupported}"
        got = evaluate(lm.module, [x.numpy()])[-1]
        err = float(np.max(np.abs(got - ref)))
        assert err < 1e-9, f"{name}: 意味論が一致しない max|Δ|={err:.3e}"


def test_real_model_reaches_machine_code_through_the_product_facade():
    """`tsugi.verify(gm)` が実 nn.Module から実機械語の検証まで一気に届く。"""
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: facade の実 FX 経路は未検証（正直に skip）")
        return
    import tsugi
    m = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16),
                      nn.GELU(approximate="tanh"), nn.Softmax(-1))
    ad = tsugi.verify(torch.fx.symbolic_trace(m))
    names = [p.name for p in ad.phases]
    assert any(n.startswith("codegen") for n in names), names
    txt = ad.to_text()
    if cg.toolchain("amd_cdna") is not None:
        assert "L2-アセンブル検証済み" in txt
    assert "L3" in txt and "常に空" in txt          # 実行は未検証と言い続ける
    assert ad.exit_code in (0, 1, 2)


def main() -> int:
    ok = True
    for t in (test_lowering_produces_ir_that_both_vendors_assemble,
              test_unrepresentable_ops_make_the_module_partial_not_silently_dropped,
              test_exact_gelu_is_refused_rather_than_approximated,
              test_shape_only_ops_are_recorded_as_ignored,
              test_unspecified_eps_is_declared_as_an_assumption,
              test_call_module_targets_resolve_or_the_whole_audit_is_a_false_ok,
              test_lowered_ir_means_the_same_thing_as_the_model,
              test_real_model_reaches_machine_code_through_the_product_facade):
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    print(f"  torch: {'あり（実 FX と意味論を照合済み）' if HAVE_TORCH else '無し（実 FX 経路は未検証）'}")
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
