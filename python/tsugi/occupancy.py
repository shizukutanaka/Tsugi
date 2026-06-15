"""Tsugi occupancy — ベンダー別の占有率推定（検証層の拡張）。

同じタイル構成が NVIDIA と AMD で *異なる占有率* になることを計算で示す。
これがクロスベンダー移植の落とし穴の一つ（性能が片方だけ崩れる）。

重要（主張と実装の一致）: 以下の HW 定数は一次情報源（NVIDIA Hopper Tuning Guide /
AMD ROCm GPU arch specs / GPUOpen）の代表アーキ実値。出典は docs/SOURCES.md。
別アーキは constants を上書きして使う（H100/MI300X/RX7900XTX 以外は要差し替え）。
"""
from __future__ import annotations

from dataclasses import dataclass

from .autotune import TileConfig


@dataclass(frozen=True)
class HwModel:
    """演算ユニット（NVIDIA SM / AMD CU）あたりの代表的リソース。要実機確認。"""

    name: str
    warp_size: int          # NVIDIA warp=32 / AMD wavefront=64
    regs_per_unit: int      # レジスタファイル総数
    smem_per_unit: int      # 共有メモリ/LDS バイト
    max_warps_per_unit: int # 同時常駐 warp/wavefront 上限


# 一次情報源の実値（代表アーキ）。出典は docs/SOURCES.md 参照。
# regs_per_unit = 演算ユニット(SM/CU)あたりの 32bit レジスタ総数
# smem_per_unit = 共有メモリ/LDS バイト
# max_warps_per_unit = 同時常駐 warp/wavefront 上限
HW: dict[str, HwModel] = {
    # NVIDIA H100 (Hopper, CC 9.0): NVIDIA Hopper Tuning Guide
    #   warp=32 / 64K 32bit regs/SM / smem 228KB/SM (carveout) / 64 warps/SM
    "nvidia":   HwModel("nvidia(H100/Hopper)", 32, 65536, 228 * 1024, 64),
    # AMD MI300X (CDNA3, gfx942): ROCm GPU arch specs + rocprofiler-compute
    #   wavefront=64 / VGPR 512KiB/CU=131072 32bit slots / LDS 64KiB/CU / 32 wavefronts/CU
    "amd_cdna": HwModel("amd_cdna(MI300X/CDNA3)", 64, 131072, 64 * 1024, 32),
    # AMD RX 7900 XTX (RDNA3, gfx1100): GPUOpen Occupancy + RDNA3 ISA
    #   wave32 / 1536 VGPR/SIMD ×32lane ×2SIMD/CU=98304 slots / LDS 64KiB(usable)/CU / 32 waves/CU
    "amd_rdna": HwModel("amd_rdna(RX7900XTX/RDNA3)", 32, 98304, 64 * 1024, 32),
}


@dataclass
class OccupancyEstimate:
    vendor: str
    occupancy: float          # 0.0–1.0
    limited_by: str           # "registers" | "shared_mem" | "warps"
    threads_per_block: int
    smem_bytes: int

    def to_text(self) -> str:
        return (f"{self.vendor:16s} occ={self.occupancy:5.0%} "
                f"limited_by={self.limited_by:11s} "
                f"threads={self.threads_per_block} smem={self.smem_bytes//1024}KB")


def estimate(cfg: TileConfig, vendor: str, *,
             regs_per_thread: int = 64, dtype_bytes: int = 2) -> OccupancyEstimate:
    """タイル構成のベンダー別占有率を推定する。

    regs_per_thread: 1 スレッドあたりレジスタ数（概算・GEMM は 64〜128 が典型）
    """
    hw = HW.get(vendor)
    if hw is None:
        raise ValueError(f"unknown vendor: {vendor}")

    threads = cfg.num_warps * hw.warp_size
    smem = (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n) * dtype_bytes * cfg.num_stages

    # 各制約から常駐可能なブロック数を算出
    warps_per_block = cfg.num_warps
    by_warps = hw.max_warps_per_unit // max(1, warps_per_block)
    by_regs = hw.regs_per_unit // max(1, threads * regs_per_thread)
    by_smem = hw.smem_per_unit // max(1, smem)

    blocks = min(by_warps, by_regs, by_smem)
    limiter = min(
        (by_warps, "warps"), (by_regs, "registers"), (by_smem, "shared_mem")
    )[1]

    resident_warps = blocks * warps_per_block
    occ = min(1.0, resident_warps / hw.max_warps_per_unit)
    return OccupancyEstimate(
        vendor=vendor, occupancy=occ, limited_by=limiter,
        threads_per_block=threads, smem_bytes=smem,
    )


def cross_vendor_occupancy(cfg: TileConfig,
                           vendors=("nvidia", "amd_cdna", "amd_rdna")) -> dict[str, OccupancyEstimate]:
    return {v: estimate(cfg, v) for v in vendors}


def occupancy_gap(cfg: TileConfig, a: str = "nvidia", b: str = "amd_cdna") -> float:
    """2 ベンダー間の占有率差（移植時に性能が片方だけ崩れる指標）。"""
    ea, eb = estimate(cfg, a), estimate(cfg, b)
    return abs(ea.occupancy - eb.occupancy)
