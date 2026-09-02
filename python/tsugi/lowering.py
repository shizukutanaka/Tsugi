"""Tsugi lowering plan — tsugi.tile IR op → 各社 intrinsic の対応表（設計の実体化）。

ここは *設計の意図*（どの op がどの NVVM/ROCDL intrinsic に落ちるか）を人手の表として
持つ層。ADR-004（Vulkan coopmat 非依存・MLIR intrinsic 直叩き）を機械可読にしたもの。

実 codegen は `tsugi.codegen`（PTX/AMDGCN テキストを生成し ptxas / llvm-mc に
アセンブルさせる）。本表の主張が本当かは codegen 層がベンダーのアセンブラに問う——
表は設計の宣言、codegen はその検証、という役割分担。
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
    "sub": {
        "nvidia": "sub.f32 / fma.rn.f32",
        "amd_cdna": "v_sub_f32 / v_fma_f32",
        "amd_rdna": "v_sub_f32 / v_fma_f32",
        "spirv": "OpFSub",
    },
    "mul": {
        "nvidia": "mul.f32 / fma.rn.f32",
        "amd_cdna": "v_mul_f32 / v_fma_f32",
        "amd_rdna": "v_mul_f32 / v_fma_f32",
        "spirv": "OpFMul",
    },
    "div": {
        "nvidia": "div.rn.f32 (rcp.approx + Newton)",
        "amd_cdna": "v_rcp_f32 + Newton refine",
        "amd_rdna": "v_rcp_f32 + Newton refine",
        "spirv": "OpFDiv",
    },
    "max": {
        "nvidia": "max.f32",
        "amd_cdna": "v_max_f32",
        "amd_rdna": "v_max_f32",
        "spirv": "OpExtInst FMax (GLSL.std.450)",
    },
    "exp": {  # 直接 exp 命令は無く 2^x ベース（×log2e）が共通実装
        "nvidia": "ex2.approx.f32 (×log2e)",
        "amd_cdna": "v_exp_f32 (2^x, ×log2e)",
        "amd_rdna": "v_exp_f32 (2^x, ×log2e)",
        "spirv": "OpExtInst Exp (GLSL.std.450)",
    },
    "sqrt": {
        "nvidia": "sqrt.rn.f32",
        "amd_cdna": "v_sqrt_f32",
        "amd_rdna": "v_sqrt_f32",
        "spirv": "OpExtInst Sqrt (GLSL.std.450)",
    },
    "rsqrt": {
        "nvidia": "rsqrt.approx.f32",
        "amd_cdna": "v_rsq_f32",
        "amd_rdna": "v_rsq_f32",
        "spirv": "OpExtInst InverseSqrt (GLSL.std.450)",
    },
    "reduce": {  # クロスレーン縮約（単一命令でなくシャッフル系パターン）
        "nvidia": "shfl.sync.bfly + redux.sync (warp reduce)",
        "amd_cdna": "ds_swizzle / v_permlane + ds reduce",
        "amd_rdna": "ds_swizzle / v_permlane + ds reduce",
        "spirv": "OpGroupNonUniformFAdd/FMax",
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


def unlowered_ops(target: str) -> set[str]:
    """DSL が emit しうるのに target 向け lowering が未定義な op を返す（嘘をつかない）。

    tracer.EMITTABLE_OPS を唯一の出所として lowering テーブルと突き合わせる。空集合なら
    その target は DSL の全 op を網羅。新 op 追加時に lowering を忘れると非空になり、
    不変条件（verify.py）が drift を検出する。
    """
    from .tracer import EMITTABLE_OPS
    missing: set[str] = set()
    for op in EMITTABLE_OPS:
        intrinsic = VENDOR_LOWERING.get(op, {}).get(target, "")
        if not intrinsic or intrinsic.startswith("<"):
            missing.add(op)
    return missing


def coverage(target: str) -> tuple[int, int]:
    """target が対応する op 数 / 全 op 数。"""
    total = len(VENDOR_LOWERING)
    covered = sum(
        1 for m in VENDOR_LOWERING.values()
        if target in m and not m[target].startswith("<")
    )
    return covered, total
