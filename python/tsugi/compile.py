"""Tsugi compile — 上流パイプラインの統合エントリ（frontend→IR→各社写像）。

tsugi.compile(kernel, args, target) で DSL から各社向け lowering plan までを
1 関数で通す。これが「コンパイラ上流」の完成形 API。

重要（主張と実装の一致）: 既定は dry-run（IR＋lowering plan）。
`emit_machine_code=True` を渡すと `codegen` 層が **実 PTX / 実 AMDGCN テキスト**を
生成し、ベンダーのアセンブラ（ptxas / llvm-mc）が受理するかまで確かめる。
ただし **実行は依然として未検証**（実機が要る）。生成物の検証レベルは
`CompiledArtifact.level`（codegen.VERIFY_LEVELS）が正直に告げる。
"""
from __future__ import annotations

from dataclasses import dataclass

from . import ir
from .codegen import AssembleResult, EmitResult, assemble, emit
from .lowering import coverage, lowering_plan
from .tracer import trace

VALID_TARGETS = ("nvidia", "amd_cdna", "amd_rdna", "spirv")


@dataclass
class CompiledArtifact:
    """上流コンパイル結果。IR と lowering plan を保持（dry-run）。"""

    target: str
    module: ir.Module
    plan: list[str]
    emitted: EmitResult | None = None      # 実アセンブリ（emit_machine_code=True 時）
    assembly: AssembleResult | None = None  # ベンダーアセンブラの判定

    @property
    def mlir(self) -> str:
        return self.module.to_mlir()

    @property
    def plan_text(self) -> str:
        return "\n".join(self.plan)

    @property
    def asm(self) -> str | None:
        """生成した実アセンブリ（PTX / AMDGCN）。dry-run なら None。"""
        return self.emitted.text if self.emitted is not None else None

    @property
    def level(self) -> str:
        """この生成物の検証レベル（codegen.VERIFY_LEVELS・L3 には到達しない）。"""
        from .codegen import VERIFY_LEVELS
        if self.assembly is None:
            return VERIFY_LEVELS[0]        # 未生成
        return self.assembly.level

    def __repr__(self) -> str:
        cov, total = coverage(self.target)
        tail = f", level={self.level}" if self.emitted is not None else ""
        return (f"CompiledArtifact(target={self.target}, "
                f"ops={len(self.module.op_kinds())}, coverage={cov}/{total}{tail})")


def compile(kernel, args: tuple, *, target: str = "nvidia",  # noqa: A001
            program_ids=(0, 0), emit_machine_code: bool = False,
            arch: str | None = None) -> CompiledArtifact:
    """DSL カーネルを target 向けに上流コンパイルする。

    target: nvidia | amd_cdna | amd_rdna | spirv
    emit_machine_code: True で実アセンブリ（PTX/AMDGCN）を生成しアセンブルまで行う。
        spirv は codegen 未対応（lowering plan のみ）。**実行は未検証**。
    """
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")

    module = trace(kernel, args, {}, program_ids=program_ids)
    plan = lowering_plan(module, target)
    art = CompiledArtifact(target=target, module=module, plan=plan)
    if emit_machine_code:
        if target == "spirv":
            raise NotImplementedError(
                "SPIR-V codegen は未実装（lowering plan のみ）。"
                "nvidia / amd_cdna / amd_rdna は emit_machine_code に対応する。"
            )
        art.emitted = emit(module, target=target, arch=arch)
        art.assembly = assemble(art.emitted.text, target=target,
                                arch=art.emitted.arch)
    return art
