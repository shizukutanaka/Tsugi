"""Tsugi reference runtime context — CPU/NumPy correctness oracle.

GPU バックエンド（NVPTX/AMDGPU）が一致すべき数値の真値を定義する。
「リファレンス実装先行」（OpenCL 失敗の解毒剤・リサーチ由来）。

ここは *性能* でなく *正しさ* の基準。GPU codegen はこの結果と
max abs error < 1e-2 (FP16) で一致しなければならない（BENCHMARK.md §4）。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .dtypes import DType

# program_id をカーネル本体から読むためのスレッドローカル文脈
_ctx = threading.local()


@dataclass
class _ProgramContext:
    program_ids: tuple[int, ...]


def _current() -> _ProgramContext:
    ctx = getattr(_ctx, "ctx", None)
    if ctx is None:
        raise RuntimeError("program_id() called outside a kernel launch")
    return ctx


def program_id(axis: int) -> int:
    """現在のプログラムインスタンスのブロックID（grid 上の座標）。"""
    return _current().program_ids[axis]


class Tensor:
    """タイル値。NumPy 配列のラッパ。DSL 演算子をオーバーロードする。

    ベンダー非依存の意味論をここで1か所に定義（単一責任・C8）。
    """

    __slots__ = ("data",)

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    # acc += tile.dot(...) を成立させる
    def __iadd__(self, other: "Tensor") -> "Tensor":
        self.data = self.data + _arr(other)
        return self

    def __add__(self, other: "Tensor") -> "Tensor":
        return Tensor(self.data + _arr(other))

    def __mul__(self, other: "Tensor") -> "Tensor":
        return Tensor(self.data * _arr(other))

    def to(self, dtype: DType) -> "Tensor":
        return Tensor(dtype.round(self.data))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype})"


def _arr(x) -> np.ndarray:
    return x.data if isinstance(x, Tensor) else np.asarray(x)


def cdiv(a: int, b: int) -> int:
    """ceil division — grid 計算用。"""
    return -(-a // b)


def launch(kernel_fn, grid: tuple[int, ...], args: tuple, kwargs: dict) -> None:
    """リファレンス実行: grid 上の全プログラムインスタンスを順に呼ぶ。

    GPU では並列。CPU リファレンスでは逐次（正しさの基準なので速度不問）。
    """
    ranges = [range(g) for g in grid]

    def _rec(prefix: tuple[int, ...], depth: int) -> None:
        if depth == len(ranges):
            _ctx.ctx = _ProgramContext(program_ids=prefix)
            try:
                kernel_fn(*args, **kwargs)
            finally:
                _ctx.ctx = None
            return
        for i in ranges[depth]:
            _rec(prefix + (i,), depth + 1)

    _rec((), 0)
