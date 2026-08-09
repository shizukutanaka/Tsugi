"""Tsugi タイル演算・リファレンス Tensor の correctness テスト。

coverage_report.py が報告する未実行関数のうち、純粋計算かつ安全に
検証可能なものを網羅する:
  - tsugi.tile.sqrt / rsqrt / maximum
  - tsugi.runtime_ref.Tensor.__add__ / __iadd__ / __repr__
  - tsugi.dtypes.DType.__repr__ / constexpr.__class_getitem__

GPU 不要。ここが緑 = リファレンス意味論が正しい。
実行: python tests/correctness/test_tile_ops.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tsugi  # noqa: E402
from tsugi import tile  # noqa: E402
from tsugi.runtime_ref import Tensor  # noqa: E402
from tsugi.dtypes import DType, constexpr, float16, float32  # noqa: E402


def test_sqrt_matches_numpy():
    x = tile.zeros((2, 2), float32)
    x.data[:] = np.array([[4.0, 9.0], [16.0, 1.0]], dtype=np.float32)
    out = tile.sqrt(x)
    assert np.allclose(_arr(out), np.sqrt(x.data))


def test_rsqrt_matches_reciprocal_sqrt():
    x = tile.zeros((2, 2), float32)
    x.data[:] = np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32)
    out = tile.rsqrt(x)
    assert np.allclose(_arr(out), 1.0 / np.sqrt(x.data))


def test_maximum_matches_numpy():
    a = tile.zeros((2, 3), float32)
    b = tile.zeros((2, 3), float32)
    a.data[:] = np.array([[1.0, 5.0, 3.0], [9.0, 2.0, 4.0]], dtype=np.float32)
    b.data[:] = np.array([[3.0, 2.0, 7.0], [1.0, 8.0, 0.0]], dtype=np.float32)
    out = tile.maximum(a, b)
    assert np.allclose(_arr(out), np.maximum(a.data, b.data))


def test_tensor_add_overload():
    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([10.0, 20.0, 30.0]))
    c = a + b
    assert isinstance(c, Tensor)
    assert np.allclose(c.data, np.array([11.0, 22.0, 33.0]))


def test_tensor_iadd_in_place():
    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([10.0, 20.0, 30.0]))
    a += b
    assert np.allclose(a.data, np.array([11.0, 22.0, 33.0]))


def test_tensor_repr_roundtrip():
    t = Tensor(np.zeros((2, 3), dtype=np.float32))
    r = repr(t)
    assert "Tensor(" in r and "shape=(2, 3)" in r


def test_dtype_repr():
    assert "float16" in repr(float16)
    assert "float32" in repr(float32)
    # __repr__ はインスタンスメソッド（self 必須）で、各インスタンスから呼ばれる経路を検証
    assert repr(float16) == "tsugi.float16"
    assert isinstance(repr(float32), str)


def test_constexpr_class_getitem():
    # constexpr は型注釈マーカーとして constexpr[int] のように使える。
    # __class_getitem__ が cls を返す経路を検証。
    alias = constexpr.__class_getitem__(int)
    assert alias is constexpr


def _arr(x) -> np.ndarray:
    return x.data if isinstance(x, Tensor) else np.asarray(x)


def main() -> int:
    ok = True
    tests = [
        test_sqrt_matches_numpy,
        test_rsqrt_matches_reciprocal_sqrt,
        test_maximum_matches_numpy,
        test_tensor_add_overload,
        test_tensor_iadd_in_place,
        test_tensor_repr_roundtrip,
        test_dtype_repr,
        test_constexpr_class_getitem,
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
