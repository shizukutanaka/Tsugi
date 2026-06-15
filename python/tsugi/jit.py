"""``@tsugi.jit`` — カーネルデコレータと grid 起動構文。

リファレンスモード（CPU/NumPy・本ファイル）では即時実行。
GPU モード（Phase 1+・要 LLVM/MLIR + 実機）では tsugi.tile IR を生成し
SPEC.md §3 パイプラインでコンパイルする（未実装・未検証と明記）。
"""
from __future__ import annotations

from typing import Callable

from .runtime_ref import launch


class _Launcher:
    """kernel[grid](*args) 構文を成立させる。"""

    def __init__(self, fn: Callable, grid: tuple[int, ...]) -> None:
        self._fn = fn
        self._grid = grid

    def __call__(self, *args, **kwargs) -> None:
        launch(self._fn, self._grid, args, kwargs)


class JITKernel:
    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", "kernel")

    def __getitem__(self, grid) -> _Launcher:
        if isinstance(grid, int):
            grid = (grid,)
        return _Launcher(self._fn, tuple(grid))

    def __call__(self, *args, **kwargs):
        # grid 指定なしの直接呼びは単一プログラム(grid=(1,))として実行
        return _Launcher(self._fn, (1,))(*args, **kwargs)


def jit(fn: Callable) -> JITKernel:
    """カーネルを JIT 対象としてマークする。"""
    return JITKernel(fn)
