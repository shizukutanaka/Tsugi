"""Tsugi compile — 上流パイプラインの統合エントリ（frontend→IR→各社写像）。

tsugi.compile(kernel, args, target) で DSL から各社向け lowering plan までを
1 関数で通す。これが「コンパイラ上流」の完成形 API。

重要（主張と実装の一致）: これは dry-run。実機械語（PTX/AMDGCN）の生成と
GPU 実行には LLVM/MLIR + 実機が必要で、本関数はそこまでは行わない。
emit_machine_code=True は未実装として明示的に NotImplementedError を投げる。
"""
from __future__ import annotations

from dataclasses import dataclass

from . import ir
from .lowering import coverage, lowering_plan
from .tracer import trace

VALID_TARGETS = ("nvidia", "amd_cdna", "amd_rdna", "spirv")


@dataclass
class CompiledArtifact:
    """上流コンパイル結果。IR と lowering plan を保持（dry-run）。"""

    target: str
    module: ir.Module
    plan: list[str]

    @property
    def mlir(self) -> str:
        return self.module.to_mlir()

    @property
    def plan_text(self) -> str:
        return "\n".join(self.plan)

    def __repr__(self) -> str:
        cov, total = coverage(self.target)
        return (f"CompiledArtifact(target={self.target}, "
                f"ops={len(self.module.op_kinds())}, coverage={cov}/{total})")


def compile(kernel, args: tuple, *, target: str = "nvidia",  # noqa: A001
            program_ids=(0, 0), emit_machine_code: bool = False) -> CompiledArtifact:
    """DSL カーネルを target 向けに上流コンパイルする（dry-run）。

    target: nvidia | amd_cdna | amd_rdna | spirv
    """
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")
    if emit_machine_code:
        raise NotImplementedError(
            "machine-code emission (PTX/AMDGCN) requires LLVM/MLIR + GPU. "
            "本 sandbox では未対応。実機で VendorLowering.cpp の実装が必要。"
        )

    module = trace(kernel, args, {}, program_ids=program_ids)
    plan = lowering_plan(module, target)
    return CompiledArtifact(target=target, module=module, plan=plan)
