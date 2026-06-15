"""Tsugi IR — tsugi.tile dialect の in-memory 表現と MLIR 風テキスト出力。

SPEC.md §2.1 / src/tsugi/ir/TsugiTileOps.td と対応。tracer がこれを生成し、
GPU バックエンド（Phase1+・要 LLVM/MLIR + 実機）がこれを各社へ lowering する。

ここは構造のみ（CPU で動作・検証可能）。実 codegen は別（未検証と明記）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Value:
    """SSA 値。名前と型を持つ。"""

    name: str          # 例 "%0"
    type: str          # 例 "tensor<128x32xf16>"

    def __str__(self) -> str:
        return self.name


@dataclass
class Op:
    """tsugi.tile op。kind は dialect の op 名（dot/load/store/zeros/reduce/...）。"""

    kind: str
    operands: list[Value] = field(default_factory=list)
    attrs: dict[str, object] = field(default_factory=dict)
    result: Value | None = None

    def to_mlir(self) -> str:
        ops = ", ".join(str(o) for o in self.operands)
        attr = ""
        if self.attrs:
            kv = ", ".join(f"{k} = {v!r}" for k, v in self.attrs.items())
            attr = f" {{{kv}}}"
        res = f"{self.result} = " if self.result else ""
        rtype = f" : {self.result.type}" if self.result else ""
        return f"  {res}tsugi_tile.{self.kind} {ops}{attr}{rtype}"


@dataclass
class Kernel:
    name: str
    params: list[str] = field(default_factory=list)
    body: list[Op] = field(default_factory=list)

    def to_mlir(self) -> str:
        lines = [f"tsugi_tile.kernel @{self.name}({', '.join(self.params)}) {{"]
        lines += [op.to_mlir() for op in self.body]
        lines.append("}")
        return "\n".join(lines)


@dataclass
class Module:
    kernels: list[Kernel] = field(default_factory=list)

    def to_mlir(self) -> str:
        return "\n\n".join(k.to_mlir() for k in self.kernels)

    def op_kinds(self) -> list[str]:
        return [op.kind for k in self.kernels for op in k.body]
