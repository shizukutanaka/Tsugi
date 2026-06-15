"""Tsugi lowering plan — tsugi.tile IR op → 各社 intrinsic の対応表（設計の実体化）。

実 codegen ではない。「どの op がどの NVVM/ROCDL intrinsic に落ちるか」を
データとして表現し、テキスト出力する。これが実機での MLIR lowering 実装の仕様になる。
ADR-004（Vulkan coopmat 非依存・MLIR intrinsic 直叩き）を機械可読にしたもの。
"""
from __future__ import annotations

from . import ir

# tsugi.tile op → (NVIDIA NVVM, AMD ROCDL) intrinsic の対応（ADR-004 / SPEC §2.3）
VENDOR_LOWERING: dict[str, dict[str, str]] = {
    "dot": {
        "nvidia": "nvvm.wmma.mma.sync",
        "amd_cdna": "rocdl.mfma.f32.16x16x16f16",
        "amd_rdna": "rocdl.wmma.f16.16x16x16.f16",
        "spirv": "OpCooperativeMatrixMulAddKHR",  # フォールバックのみ
    },
    "load": {
        "nvidia": "ld.global + cp.async (staging)",
        "amd_cdna": "global_load + ds_write (LDS)",
        "amd_rdna": "global_load + ds_write (LDS)",
        "spirv": "OpLoad",
    },
    "store": {
        "nvidia": "st.global",
        "amd_cdna": "global_store",
        "amd_rdna": "global_store",
        "spirv": "OpStore",
    },
    "zeros": {
        "nvidia": "register init (mov 0)",
        "amd_cdna": "register init (v_mov 0)",
        "amd_rdna": "register init (v_mov 0)",
        "spirv": "OpConstantComposite",
    },
    "add": {
        "nvidia": "add.f32 / fma",
        "amd_cdna": "v_add_f32 / v_fma",
        "amd_rdna": "v_add_f32 / v_fma",
        "spirv": "OpFAdd",
    },
    "cast": {
        "nvidia": "cvt.f16.f32",
        "amd_cdna": "v_cvt_f16_f32",
        "amd_rdna": "v_cvt_f16_f32",
        "spirv": "OpFConvert",
    },
}


def lowering_plan(module: ir.Module, target: str) -> list[str]:
    """IR の各 op を target 向け intrinsic へ写像した計画を返す（テキスト）。

    target: nvidia | amd_cdna | amd_rdna | spirv
    未対応 op は明示（嘘をつかない）。
    """
    plan: list[str] = []
    for kernel in module.kernels:
        plan.append(f"; lowering @{kernel.name} → {target}")
        for op in kernel.body:
            mapping = VENDOR_LOWERING.get(op.kind, {})
            intrinsic = mapping.get(target, "<UNSUPPORTED — needs implementation>")
            res = f"{op.result} = " if op.result else ""
            plan.append(f"  {res}{op.kind:6s} → {intrinsic}")
    return plan


def coverage(target: str) -> tuple[int, int]:
    """target が対応する op 数 / 全 op 数。"""
    total = len(VENDOR_LOWERING)
    covered = sum(
        1 for m in VENDOR_LOWERING.values()
        if target in m and not m[target].startswith("<")
    )
    return covered, total
