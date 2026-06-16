"""error propagation の検証（新視点・第4ラウンド）。

per-kernel 等価 ⇏ per-model 等価 を、合成モデルと実 numpy シミュレーションの
両方で実証する。深さに比例して発散が累積し、ill-conditioned op で増幅されること、
そして単一カーネルの許容では足りないことを示す。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tsugi.equivalence import simulate_vendor_matmul  # noqa: E402
from tsugi.propagation import (  # noqa: E402
    GraphOp,
    empirical_cond,
    is_amplifier,
    model_tolerance,
    propagate,
)
from tsugi.tolerance import expected_gemm_abs_error  # noqa: E402


def test_depth_accumulates_divergence():
    # 同じ matmul を 1 層 vs 12 層: モデル発散は深さに比例して増える
    one = model_tolerance([GraphOp("matmul", K=256)])
    deep = model_tolerance([GraphOp("matmul", K=256)] * 12)
    assert deep > one
    # ほぼ線形（12 層は 1 層の 10〜14 倍程度）
    assert 10 < deep / one < 14, f"depth scaling off: {deep / one}"


def test_naive_per_kernel_underestimates():
    # 増幅 op を挟むと素朴な local 和より合成発散が大きい
    ops = [GraphOp("matmul", K=512),
           GraphOp("softmax", cond=8.0),   # ill-conditioned（大 logit）
           GraphOp("matmul", K=512)]
    rep = propagate(ops)
    assert rep.model_divergence > rep.naive_sum, \
        "composition must exceed naive per-kernel sum"


def test_ill_conditioned_op_is_dominant():
    ops = [GraphOp("matmul", K=128),
           GraphOp("reduce", cond=50.0),   # 強い相殺
           GraphOp("add")]
    rep = propagate(ops)
    assert rep.dominant.kind == "reduce", f"dominant should be reduce, got {rep.dominant.kind}"


def _run_chain(L, n, K, seed=0, residual=False, relative=False):
    """L 層の matmul+rmsnorm を 2 ベンダー（累積順序違い）で流し発散を返す。

    residual=True なら pre-norm 残差ブロック x = x + matmul(rmsnorm(x), w)。
    relative=True なら相対発散（||a-b||/||a||）、既定は最大絶対発散。
    """
    rng = np.random.default_rng(seed)
    x_a = rng.standard_normal((n, K)).astype(np.float16)
    x_b = x_a.copy()
    weights = [rng.standard_normal((K, K)).astype(np.float16) * (1.0 / np.sqrt(K))
               for _ in range(L)]

    def rmsnorm(x):
        return x / (np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=1, keepdims=True)) + 1e-6)

    for w in weights:
        # vendor A: f32 累積 / vendor B: split-k（累積順序差）= 別ベンダー相当
        if residual:  # pre-norm 残差: x = x + matmul(rmsnorm(x), w)
            fa = simulate_vendor_matmul(rmsnorm(x_a).astype(np.float16), w, accum="f32")
            fb = simulate_vendor_matmul(rmsnorm(x_b).astype(np.float16), w, accum="f32", split_k=8)
            x_a = x_a.astype(np.float64) + fa
            x_b = x_b.astype(np.float64) + fb
        else:         # 平坦チェーン: x = rmsnorm(matmul(x, w))
            x_a = rmsnorm(simulate_vendor_matmul(x_a.astype(np.float16), w, accum="f32"))
            x_b = rmsnorm(simulate_vendor_matmul(x_b.astype(np.float16), w, accum="f32", split_k=8))
    a, b = x_a.astype(np.float64), x_b.astype(np.float64)
    if relative:
        return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
    return float(np.abs(a - b).max())


def test_per_kernel_pass_but_model_diverges_numpy():
    # 実 numpy: 同じ許容内の単一カーネル発散が、深さで実際に累積することを実証。
    # per-kernel(1層)では小さい発散が、モデル(12層)では桁違いに育つ。
    n, K = 64, 64
    one_layer = _run_chain(1, n, K)
    deep = _run_chain(12, n, K)

    # (1) 合成で実測発散が深さとともに育つ（per-kernel 検証では見えない）
    assert deep > one_layer * 3, \
        f"model divergence should grow with depth: 1層={one_layer:.2e} 12層={deep:.2e}"
    # (2) 合成予測（深さ込み・safety 付き上界）は実測を上回る
    predicted = model_tolerance([GraphOp("matmul", K=K)] * 12)
    assert predicted >= deep, \
        f"propagated bound {predicted:.2e} should bound measured {deep:.2e}"
    # (3) 単一カーネル許容を「12 層分」誤って流用すると過小評価になる
    single = expected_gemm_abs_error(K, "float16", scale=1.0)
    assert predicted > single, "model-level tolerance must exceed single-kernel tolerance"


def test_residual_dilutes_model_divergence():
    # 残差トポロジ（skip 接続）は同じ深さでも発散を希釈する（一次モデル）。
    plain = [GraphOp("matmul", K=512) for _ in range(24)]
    resid = [GraphOp("matmul", K=512, residual=True) for _ in range(24)]
    mp = propagate(plain).model_divergence
    mr = propagate(resid).model_divergence
    assert mr < mp * 0.5            # 残差は線形累積よりずっと小さい
    # 単一ブロックでは差が出ない（深さ効果）
    one_p = propagate([GraphOp("matmul", K=512)]).model_divergence
    one_r = propagate([GraphOp("matmul", K=512, residual=True)]).model_divergence
    assert abs(one_p - one_r) < 1e-12


def test_residual_lower_than_plain_numpy():
    # 実 numpy（pre-norm）: 深い残差スタックの相対発散は平坦チェーンより小さい。
    n, K, L = 16, 64, 24
    plain = _run_chain(L, n, K, residual=False, relative=True)
    resid = _run_chain(L, n, K, residual=True, relative=True)
    assert resid < plain, f"residual {resid:.2e} should be < plain {plain:.2e}"


def test_empty_graph():
    assert model_tolerance([]) == 0.0


def test_only_genuine_relative_amplifiers():
    # 相対誤差を増幅するのは reduce/softmax/exp のみ。div/reciprocal/add は相対 ~1。
    assert is_amplifier("reduce") and is_amplifier("exp") and is_amplifier("softmax")
    assert not is_amplifier("div") and not is_amplifier("reciprocal")
    assert not is_amplifier("add") and not is_amplifier("matmul")


def test_empirical_cond_is_data_driven():
    rng = np.random.default_rng(0)
    signed = rng.standard_normal((4, 256))
    positive = np.abs(rng.standard_normal((4, 256)))
    # 符号付き和は相殺で κ≫1、正の和は κ≈1（相殺なし）
    assert empirical_cond(signed, "reduce", axis=1) > 5.0
    assert empirical_cond(positive, "reduce", axis=1) < 1.5
    # max reduction は well-conditioned
    assert empirical_cond(signed, "reduce", axis=1, reduce_kind="max") == 1.0
    # exp の相対条件数は max|x|
    logits = rng.standard_normal((4, 16)) * 3
    assert abs(empirical_cond(logits, "exp") - float(np.abs(logits).max())) < 1e-9
    # div は相対では増幅しない
    assert empirical_cond(signed, "div") == 1.0


def test_empirical_cond_makes_amplification_fire():
    # 静的 cond=1 では amp=1 だが、実測 cond を入れると合成発散が naive 和を超える
    rng = np.random.default_rng(1)
    signed = rng.standard_normal((4, 512))
    kappa = empirical_cond(signed, "reduce", axis=1)
    ops = [GraphOp("matmul", K=256), GraphOp("reduce", cond=kappa)]
    rep = propagate(ops)
    assert rep.model_divergence > rep.naive_sum   # 実測 cond で増幅が発火


def main() -> int:
    ok = True
    tests = [
        test_depth_accumulates_divergence,
        test_naive_per_kernel_underestimates,
        test_ill_conditioned_op_is_dominant,
        test_per_kernel_pass_but_model_diverges_numpy,
        test_residual_dilutes_model_divergence,
        test_residual_lower_than_plain_numpy,
        test_empty_graph,
        test_only_genuine_relative_amplifiers,
        test_empirical_cond_is_data_driven,
        test_empirical_cond_makes_amplification_fire,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: 12 層 transformer 風グラフの合成発散
    ops = ([GraphOp("matmul", K=256), GraphOp("softmax", cond=4.0),
            GraphOp("matmul", K=256), GraphOp("add")] * 3)
    print("\n--- 12-ish op グラフの発散伝播 ---")
    print(propagate(ops).to_text())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
