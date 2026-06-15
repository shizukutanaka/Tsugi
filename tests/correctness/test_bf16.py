"""bf16 忠実丸めのテスト。oracle が bf16 の精度損失を実際に再現することを検証。

これまで bf16→f32 マップで精度損失を無視していた弱点の修正。tolerance の u=2^-8 と整合。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tsugi  # noqa: E402
from tsugi.dtypes import round_to_bf16  # noqa: E402


def test_bf16_loses_precision_beyond_7bits():
    # bf16 は仮数 7bit。1 + 2^-8 は表現可能、1 + 2^-10 は丸めで失われる
    x = np.array([1.0 + 2.0 ** -10], dtype=np.float32)
    r = round_to_bf16(x)
    assert r[0] != x[0], "bf16 should lose 2^-10 bit"


def test_bf16_exact_values_preserved():
    # bf16 で正確に表現できる値は不変
    for v in (0.0, 1.0, 2.0, 0.5, -4.0):
        x = np.array([v], dtype=np.float32)
        assert round_to_bf16(x)[0] == v, f"{v} should be exact in bf16"


def test_bf16_round_to_nearest_even():
    # 大きい配列で平均的に round-to-nearest（バイアスなし）
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10000).astype(np.float32)
    r = round_to_bf16(x)
    # 丸め誤差の平均はほぼ 0（バイアスのない丸め）
    assert abs(np.mean(r - x)) < 1e-3


def test_dtype_to_uses_bf16_rounding():
    # tsugi.float32 → bfloat16 変換が実際に丸める
    import tsugi.tile as tile
    t = tile.zeros((4, 4), tsugi.float32)
    t.data = np.full((4, 4), 1.0 + 2.0 ** -10, dtype=np.float32)
    b = t.to(tsugi.bfloat16)
    assert not np.allclose(b.data, t.data, atol=0), "to(bf16) must round"


def test_bf16_coarser_than_fp16():
    # 同じ値で bf16 の誤差 >= fp16 の誤差（bf16 は仮数が粗い）
    rng = np.random.default_rng(1)
    x = rng.standard_normal(10000).astype(np.float32)
    err_bf16 = np.max(np.abs(round_to_bf16(x) - x))
    err_fp16 = np.max(np.abs(x.astype(np.float16).astype(np.float32) - x))
    assert err_bf16 >= err_fp16, f"bf16 {err_bf16} should be >= fp16 {err_fp16}"


def main() -> int:
    ok = True
    tests = [
        test_bf16_loses_precision_beyond_7bits,
        test_bf16_exact_values_preserved,
        test_bf16_round_to_nearest_even,
        test_dtype_to_uses_bf16_rounding,
        test_bf16_coarser_than_fp16,
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
