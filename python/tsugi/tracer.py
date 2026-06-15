"""Tsugi tracer — @tsugi.jit カーネルを tsugi.tile IR へトレースする。

方式: 具体トレース（concrete trace）。実値（NumPy）で計算しながら同時に IR を記録する。
利点:
  - IR が必ずリファレンス実行と一致する（IR が正しいことが自明）
  - MLIR 風テキストが得られる（GPU lowering の入力）
  - Python の for ループは具体値で展開される（MLIR の素朴な初期形）

トレース対象は 1 プログラムインスタンス（program_id 固定）の命令列。
GPU では grid 全体に展開されるが、IR 構造の検証にはこれで十分。
"""
from __future__ import annotations

import threading

import numpy as np

from . import ir
from .runtime_ref import _arr

_trace = threading.local()


class _Tracer:
    def __init__(self, kernel_name: str) -> None:
        self.kernel = ir.Kernel(name=kernel_name)
        self._counter = 0

    def fresh(self, type_str: str) -> ir.Value:
        v = ir.Value(name=f"%{self._counter}", type=type_str)
        self._counter += 1
        return v

    def emit(self, kind: str, operands, attrs, result_type: str | None) -> ir.Value | None:
        res = self.fresh(result_type) if result_type else None
        self.kernel.body.append(
            ir.Op(kind=kind, operands=list(operands), attrs=dict(attrs), result=res)
        )
        return res


class SymTensor:
    """トレース中の値。NumPy 実体（検証用）と IR 値（記録用）を両方持つ。"""

    __slots__ = ("data", "val")

    def __init__(self, data: np.ndarray, val: ir.Value) -> None:
        self.data = data
        self.val = val

    def _tt(self) -> str:
        s = "x".join(str(d) for d in self.data.shape)
        dt = {np.dtype("float16"): "f16", np.dtype("float32"): "f32"}.get(
            self.data.dtype, str(self.data.dtype)
        )
        return f"tensor<{s}x{dt}>"

    def __iadd__(self, other: "SymTensor") -> "SymTensor":
        t = _tracer()
        res = t.emit("add", [self.val, other.val], {}, self._tt())
        self.data = self.data + _arr(other)
        self.val = res
        return self

    def to(self, dtype) -> "SymTensor":
        t = _tracer()
        data = dtype.round(self.data)
        s = "x".join(str(d) for d in data.shape)
        dt = "f16" if dtype.np == np.float16 else "f32"
        res = t.emit("cast", [self.val], {"to": dtype.name}, f"tensor<{s}x{dt}>")
        return SymTensor(data, res)


def _tracer() -> _Tracer:
    t = getattr(_trace, "t", None)
    if t is None:
        raise RuntimeError("trace op called outside trace()")
    return t


# --- トレース版 tile op（IR を積みつつ実値を計算） -----------------------

class _TraceTile:
    @staticmethod
    def zeros(shape, dtype=None) -> SymTensor:
        from .dtypes import float32
        dtype = dtype or float32
        data = np.zeros(shape, dtype=dtype.np)
        s = "x".join(str(d) for d in shape)
        dt = "f16" if dtype.np == np.float16 else "f32"
        res = _tracer().emit("zeros", [], {"shape": list(shape)}, f"tensor<{s}x{dt}>")
        return SymTensor(data, res)

    @staticmethod
    def load(ptr, offsets, shape) -> SymTensor:
        r0, c0 = offsets
        tr, tc = shape
        out = np.zeros((tr, tc), dtype=ptr.dtype)
        r1, c1 = min(r0 + tr, ptr.shape[0]), min(c0 + tc, ptr.shape[1])
        out[: r1 - r0, : c1 - c0] = ptr[r0:r1, c0:c1]
        s = f"{tr}x{tc}"
        dt = "f16" if ptr.dtype == np.float16 else "f32"
        res = _tracer().emit("load", [], {"offset": list(offsets)}, f"tensor<{s}x{dt}>")
        return SymTensor(out, res)

    @staticmethod
    def store(ptr, offsets, value: SymTensor) -> None:
        r0, c0 = offsets
        v = value.data
        tr, tc = v.shape
        r1, c1 = min(r0 + tr, ptr.shape[0]), min(c0 + tc, ptr.shape[1])
        ptr[r0:r1, c0:c1] = v[: r1 - r0, : c1 - c0].astype(ptr.dtype)
        _tracer().emit("store", [value.val], {"offset": list(offsets)}, None)

    @staticmethod
    def dot(a: SymTensor, b: SymTensor, acc: SymTensor | None = None) -> SymTensor:
        av = a.data.astype(np.float32)
        bv = b.data.astype(np.float32)
        out = av @ bv
        operands = [a.val, b.val]
        if acc is not None:
            out = out + acc.data.astype(np.float32)
            operands.append(acc.val)
        s = "x".join(str(d) for d in out.shape)
        res = _tracer().emit("dot", operands, {}, f"tensor<{s}xf32>")
        return SymTensor(out, res)


def trace(kernel, args: tuple, kwargs: dict, program_ids=(0, 0)) -> ir.Module:
    """カーネルを 1 プログラムインスタンスとしてトレースし IR を返す。

    kernel: JITKernel（プレーンな fn でも可）
    """
    from . import runtime_ref, tile as tile_mod
    from .jit import JITKernel

    fn = kernel._fn if isinstance(kernel, JITKernel) else kernel
    name = getattr(fn, "__name__", "kernel")
    t = _Tracer(name)
    _trace.t = t

    # tile モジュールの関数を一時的にトレース版へ差し替え（共有モジュールを直接patch）
    saved = {k: getattr(tile_mod, k) for k in ("zeros", "load", "store", "dot")}
    tile_mod.zeros = _TraceTile.zeros
    tile_mod.load = _TraceTile.load
    tile_mod.store = _TraceTile.store
    tile_mod.dot = _TraceTile.dot
    runtime_ref._ctx.ctx = runtime_ref._ProgramContext(program_ids=program_ids)
    try:
        fn(*args, **kwargs)
    finally:
        for k, v in saved.items():
            setattr(tile_mod, k, v)
        runtime_ref._ctx.ctx = None
        _trace.t = None

    return ir.Module(kernels=[t.kernel])
