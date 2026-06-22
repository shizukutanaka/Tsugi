"""Tsugi — 継ぎ — GPU ベンダーを接合する移植検証層.

公開 API。現状はリファレンス実装（CPU/NumPy・正しさの真値）が動作する。
GPU バックエンド（NVPTX/AMDGPU）はこのリファレンスと一致するよう実装される
（Phase 1+・要 LLVM/MLIR + 実機・未検証の経路は「未検証」と明記）。

    import tsugi
    from tsugi import tile

    @tsugi.jit
    def k(...): ...

    k[grid](...)
"""
from __future__ import annotations

from . import (
    calibration,
    decision,
    envelope,
    equivalence,
    feasibility,
    ir,
    nondeterminism,
    occupancy,
    oracle_check,
    portability,
    propagation,
    provenance,
    rollout,
    tile,
    tolerance,
)
from .audit import Audit, audit, audit_cross_vendor, audit_runtime
from .compile import compile  # noqa: A004
from .dtypes import bfloat16, constexpr, float16, float32, int8, int32
from .jit import jit
from .runtime_ref import cdiv, program_id
from .tracer import trace

__all__ = [
    "tile",
    "ir",
    "portability",
    "equivalence",
    "occupancy",
    "tolerance",
    "feasibility",
    "propagation",
    "envelope",
    "calibration",
    "nondeterminism",
    "decision",
    "rollout",
    "oracle_check",
    "provenance",
    "audit",
    "audit_runtime",
    "audit_cross_vendor",
    "Audit",
    "jit",
    "trace",
    "compile",
    "program_id",
    "cdiv",
    "constexpr",
    "float16",
    "bfloat16",
    "float32",
    "int32",
    "int8",
]

__version__ = "0.2.0"
