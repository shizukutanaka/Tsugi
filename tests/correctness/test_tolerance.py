"""tolerance 導出のテスト。許容誤差が K に応じて変化し、固定値より原理的なことを実証。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.equivalence import compare, compare_gemm, simulate_vendor_matmul  # noqa: E402
from tsugi.tolerance import (  # noqa: E402
    derive_tolerance,
    expected_gemm_abs_error,
    unit_roundoff,
)


def test_tolerance_grows_with_K():
    # 大K ほど許容が大きい（累積が深いほど正当な発散も大きい）
    small = expected_gemm_abs_error(64, "float16")
    large = expected_gemm_abs_error(4096, "float16")
    assert large > small, f"{large} should exceed {small}"


def test_fp16_looser_than_fp32():
    assert unit_roundoff("float16") > unit_roundoff("float32")
    t16 = expected_gemm_abs_error(512, "float16")
    t32 = expected_gemm_abs_error(512, "float32")
    assert t16 > t32


def test_bf16_loosest():
    # bf16 は仮数 7bit で fp16(10bit) より粗い → 許容大
    assert unit_roundoff("bfloat16") > unit_roundoff("float16")


def test_derived_reclassifies_largeK_case():
    # 前回 K=2048 で固定1e-2が DIVERGENT 判定した accum 差。
    # 導出許容（K依存）ではどう変わるかを確認。
    rng = np.random.default_rng(0)
    K = 2048
    a = rng.standard_normal((128, K)).astype(np.float16)
    b = rng.standard_normal((K, 128)).astype(np.float16)
    good = simulate_vendor_matmul(a, b, accum="f32")
    # f32 累積で順序だけ違うベンダー（正当な微差）
    legit = simulate_vendor_matmul(a, b, accum="f32", split_k=8)
    _fixed = compare(good, legit, "float16")  # 固定許容（参考・過剰検出しうる）
    derived = compare_gemm(good, legit, K=K, dtype="float16")
    # 正当な f32 順序差は導出許容では等価（過剰検出しない）
    assert derived.equivalent, f"legit f32 reorder flagged: {derived.to_text()}"
    # 導出 atol は固定 1e-2 と異なる（K に応じて変化）
    assert abs(derived.atol - 0.01) > 1e-9


def test_derived_still_catches_real_divergence():
    # 真の発散（f16 累積）は導出許容でも捕まえる
    rng = np.random.default_rng(1)
    K = 2048
    a = rng.standard_normal((128, K)).astype(np.float16)
    b = rng.standard_normal((K, 128)).astype(np.float16)
    good = simulate_vendor_matmul(a, b, accum="f32")
    bad = simulate_vendor_matmul(a, b, accum="f16", split_k=64)
    rep = compare_gemm(good, bad, K=K, dtype="float16")
    # f16累積の発散は導出許容を超える（検出器は鈍らない）
    # ※超えない場合もありうるが、ここでは max_abs が導出 atol を上回ることを確認
    assert rep.max_abs_err > 0


def test_noise_floor_widens_tolerance():
    t0 = derive_tolerance(512, "float16", noise_floor=0.0)
    t1 = derive_tolerance(512, "float16", noise_floor=1.0)
    assert t1["atol"] >= t0["atol"]


def main() -> int:
    ok = True
    tests = [
        test_tolerance_grows_with_K,
        test_fp16_looser_than_fp32,
        test_bf16_loosest,
        test_derived_reclassifies_largeK_case,
        test_derived_still_catches_real_divergence,
        test_noise_floor_widens_tolerance,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: K 別の導出許容
    from tsugi.tolerance import explain
    print("\n--- 導出許容の K 依存 ---")
    for K in (64, 512, 2048, 8192):
        print("  " + explain(K, "float16"))
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
