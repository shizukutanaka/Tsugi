"""Tsugi ``tile`` namespace — タイル演算のリファレンス意味論。

SPEC.md §1.5 / §2.1 の各 op を NumPy で定義する。これが正しさの真値。
GPU では同じ op が tsugi.tile dialect → 各社行列コアへ lowering される。

オフセット規約（well-defined・SPEC と一致）:
  load(ptr, (row_start, col_start), (TILE_R, TILE_C)) は
  ptr[row_start:row_start+TILE_R, col_start:col_start+TILE_C] を返す。
  row_start/col_start は *要素* 単位（ブロック index ではない）。
"""
from __future__ import annotations

import numpy as np

from .dtypes import DType, float32
from .runtime_ref import Tensor, _arr


def zeros(shape: tuple[int, ...], dtype: DType = float32) -> Tensor:
    return Tensor(np.zeros(shape, dtype=dtype.np))


def load(ptr: np.ndarray, offsets: tuple[int, int], shape: tuple[int, int]) -> Tensor:
    """グローバル配列からタイルをロード（境界外はゼロパディング）。"""
    r0, c0 = offsets
    tr, tc = shape
    out = np.zeros((tr, tc), dtype=ptr.dtype)
    r1, c1 = min(r0 + tr, ptr.shape[0]), min(c0 + tc, ptr.shape[1])
    out[: r1 - r0, : c1 - c0] = ptr[r0:r1, c0:c1]
    return Tensor(out)


def store(ptr: np.ndarray, offsets: tuple[int, int], value: Tensor) -> None:
    """タイルをグローバル配列へストア（境界クリップ）。"""
    r0, c0 = offsets
    v = _arr(value)
    tr, tc = v.shape
    r1, c1 = min(r0 + tr, ptr.shape[0]), min(c0 + tc, ptr.shape[1])
    ptr[r0:r1, c0:c1] = v[: r1 - r0, : c1 - c0].astype(ptr.dtype)


def dot(a: Tensor, b: Tensor, acc: Tensor | None = None) -> Tensor:
    """行列積。GPU では行列コア（WMMA/MFMA）へ lowering（ADR-004）。

    accumulate は FP32 で行う（GPU の標準: f16 入力・f32 accum）。
    """
    av = _arr(a).astype(np.float32)
    bv = _arr(b).astype(np.float32)
    out = av @ bv
    if acc is not None:
        out = out + _arr(acc).astype(np.float32)
    return Tensor(out)


def reduce(x: Tensor, axis: int, kind: str = "sum") -> Tensor:
    v = _arr(x)
    if kind == "sum":
        return Tensor(np.sum(v, axis=axis, keepdims=True))
    if kind == "max":
        return Tensor(np.max(v, axis=axis, keepdims=True))
    raise ValueError(f"unknown reduce kind: {kind}")


def exp(x: Tensor) -> Tensor:
    return Tensor(np.exp(_arr(x)))


def sqrt(x: Tensor) -> Tensor:
    return Tensor(np.sqrt(_arr(x)))


def rsqrt(x: Tensor) -> Tensor:
    return Tensor(1.0 / np.sqrt(_arr(x)))


def maximum(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(np.maximum(_arr(a), _arr(b)))
