"""数値エンベロープ実行時検査のテスト（新視点・第5ラウンド）。

静的に認証した等価性が、本番入力のエンベロープ逸脱で無効化されることを実証。
oracle も第2ベンダーも使わず単一ベンダーの統計だけで危険を捕まえる。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tsugi.envelope import (  # noqa: E402
    certify_gemm,
    check_softmax_input,
    check_tensor,
    dtype_limits,
)
from tsugi.portability import Risk  # noqa: E402


def test_in_envelope_passes():
    env = certify_gemm(K=256, dtype="float16", scale=1.0)
    x = np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32)
    rep = check_tensor(x, env)
    assert rep.in_envelope, rep.to_text()


def test_fp16_overflow_is_block():
    env = certify_gemm(K=256, dtype="float16", scale=1.0)
    x = np.full((4, 4), 70000.0, dtype=np.float32)   # > fp16 max 65504
    rep = check_tensor(x, env)
    assert not rep.in_envelope
    assert rep.max_risk == Risk.BLOCK


def test_scale_exceedance_voids_certification():
    # 認証は scale=1 で atol を保証。本番スケールが大きいと認証 atol は無効。
    env = certify_gemm(K=256, dtype="float16", scale=1.0)
    x = np.random.default_rng(1).standard_normal((64, 64)).astype(np.float32) * 50.0
    rep = check_tensor(x, env)
    assert rep.max_risk == Risk.BLOCK
    assert any("再認証" in f.message for f in rep.findings)


def test_denormal_flagged_for_ftz_divergence():
    env = certify_gemm(K=64, dtype="float16", scale=1.0)
    lim = dtype_limits("float16")
    x = np.zeros((8, 8), dtype=np.float32)
    x[0, 0] = lim.min_normal * 0.1   # denormal 域
    x[0, 1] = 1.0                    # スケールは正常に保つ
    rep = check_tensor(x, env)
    assert any("denormal" in f.message or "FTZ" in f.message for f in rep.findings)


def test_nan_is_block():
    env = certify_gemm(K=64, dtype="float16", scale=1.0)
    x = np.array([[1.0, np.nan]], dtype=np.float32)
    rep = check_tensor(x, env)
    assert rep.max_risk == Risk.BLOCK


def test_fp16_softmax_logit_overflow():
    # fp16 で生 logit が ln(65504)≈11.09 を超えると exp が inf → 片ベンダーで破綻
    env = certify_gemm(K=128, dtype="float16", scale=1.0)
    logits = np.array([[0.0, 12.5, 3.0]], dtype=np.float32)
    rep = check_softmax_input(logits, env)
    assert rep.max_risk == Risk.BLOCK
    # 同じ logit でも bf16/f32 は範囲が広く OK（dtype 依存の差を実証）
    env32 = certify_gemm(K=128, dtype="float32", scale=1.0)
    assert check_softmax_input(logits, env32).in_envelope


def test_real_fp16_overflow_actually_happens():
    # 検査が机上でなく実挙動: exp(12.5) を fp16 で計算すると本当に inf になる
    lim = dtype_limits("float16")
    with np.errstate(over="ignore"):
        assert np.isinf(np.exp(np.float16(12.5)))     # 12.5 > 11.09 → inf
        assert not np.isinf(np.exp(np.float16(10.0)))  # 10.0 < 11.09 → 有限
    assert 11.0 < lim.exp_overflow < 11.2


def main() -> int:
    ok = True
    tests = [
        test_in_envelope_passes,
        test_fp16_overflow_is_block,
        test_scale_exceedance_voids_certification,
        test_denormal_flagged_for_ftz_divergence,
        test_nan_is_block,
        test_fp16_softmax_logit_overflow,
        test_real_fp16_overflow_actually_happens,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: fp16 と bf16 のエンベロープ差（overflow vs precision）
    print("\n--- dtype 別エンベロープ（IEEE 754 実値）---")
    for d in ("float16", "bfloat16", "float32"):
        lim = dtype_limits(d)
        print(f"  {d:9s} max={lim.max_normal:.3g} min_normal={lim.min_normal:.2e} "
              f"exp-overflow at |x|>{lim.exp_overflow:.2f}")
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
