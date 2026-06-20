"""equivalence 検出器のテスト。擬似ベンダーで数値発散を*捕まえる*ことを実証。

これは GPU シミュレーション（CPU）。実 GPU では同じ compare() を実機出力に適用する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.equivalence import (  # noqa: E402
    DV_DIVERGENT,
    DV_EQUIVALENT,
    DV_LAYOUT,
    classify_divergence,
    compare,
    simulate_vendor_matmul,
)


def _inputs(M=128, N=128, K=512):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    return a, b


def test_identical_is_equivalent():
    a, b = _inputs()
    ref = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    rep = compare(ref, ref.copy(), dtype="float16")
    assert rep.equivalent, rep.to_text()


def test_f32_vs_f32_within_tolerance():
    # 同じ f32 累積・分割違いは微差 → 等価のはず
    a, b = _inputs()
    v1 = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    v2 = simulate_vendor_matmul(a, b, accum="f32", split_k=4)
    rep = compare(v1, v2, dtype="float16")
    assert rep.equivalent, f"f32 split should be equivalent: {rep.to_text()}"


def test_f16_accum_detected_as_divergent():
    # fp16 累積は大 K で誤差が膨らむ → DIVERGENT を検出できるべき
    a, b = _inputs(K=2048)
    good = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    bad = simulate_vendor_matmul(a, b, accum="f16", split_k=64)
    rep = compare(good, bad, dtype="float16")
    # 検出器が機能 = ズレを equivalent と誤判定しない
    assert not rep.equivalent, f"divergence not caught: {rep.to_text()}"
    assert rep.max_abs_err > 1e-2


def test_report_fields():
    a, b = _inputs()
    ref = simulate_vendor_matmul(a, b)
    rep = compare(ref, ref.copy(), "float16")
    assert rep.n_mismatch == 0
    assert rep.n_total == ref.size


def test_uniform_risk_interface():
    # EquivalenceReport も他レポートと同じ risk/max_risk/ok を持つ（検証層の統一）
    from tsugi.report import Risk
    a, b = _inputs()
    ref = simulate_vendor_matmul(a, b)
    eq = compare(ref, ref.copy(), "float16")
    assert eq.ok and eq.risk is Risk.OK and eq.max_risk is Risk.OK
    dv = compare(ref, ref + 10.0, "float16")
    assert not dv.ok and dv.risk is Risk.BLOCK


def test_classify_layout_vs_numerical_divergence():
    # element-wise 不一致を レイアウト不一致(値正しい) vs 真の数値発散 に区別
    rng = np.random.default_rng(0)
    K = 256
    a = rng.standard_normal((64, 64)).astype(np.float32)
    assert classify_divergence(a, a.copy(), K) == DV_EQUIVALENT
    # 転置・置換は値の多重集合を保存 → LAYOUT（数値は正しい・整列バグ）
    assert classify_divergence(a, a.T.copy(), K) == DV_LAYOUT
    shuffled = rng.permutation(a.reshape(-1)).reshape(64, 64).astype(np.float32)
    assert classify_divergence(a, shuffled, K) == DV_LAYOUT
    # 真のスケール発散は multiset も崩す → DIVERGENT
    assert classify_divergence(a, (a * 1.5).astype(np.float32), K) == DV_DIVERGENT
    # 要素数が違えば layout ではありえない → DIVERGENT
    assert classify_divergence(a, a[:, :32].copy(), K) == DV_DIVERGENT


def main() -> int:
    ok = True
    tests = [
        test_identical_is_equivalent,
        test_f32_vs_f32_within_tolerance,
        test_f16_accum_detected_as_divergent,
        test_report_fields,
        test_uniform_risk_interface,
        test_classify_layout_vs_numerical_divergence,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: 擬似 NVIDIA(f32) vs 擬似 AMD(f16累積) の照合
    a, b = _inputs(K=2048)
    nv = simulate_vendor_matmul(a, b, accum="f32")
    amd = simulate_vendor_matmul(a, b, accum="f16", split_k=64)
    print("\n--- 擬似ベンダー等価性照合（K=2048）---")
    print("nvidia(f32) vs amd(f16累積):", compare(nv, amd, "float16").to_text())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
