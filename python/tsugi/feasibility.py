"""Tsugi feasibility — 起動可能性の検証（ソクラテス問答・第3ラウンドの新視点）。

盲点: occupancy は「同じ構成でも占有率がベンダー間で違う」を *連続量* として示す。
だが occ=0% を portability は「ただ遅い」WARN として扱っていた。これは誤り。
occ=0% は「遅い」のではなく「*起動できない*」= 動かない。連続（占有率）と
離散の関門（起動可否）を混同していた。

per-block のハード上限（共有メモリ/LDS・threads・regs/thread）はベンダーで違う。
NVIDIA Hopper の 227KB/block smem を前提にチューニングした構成は、AMD CDNA の
LDS 64KiB/workgroup に *物理的に収まらず* カーネルが launch すらしない。
これは性能差でなく「1ソース・2ベンダー」の約束そのものの破綻。

ゆえに feasibility は等価性・占有率より *上流* の categorical ゲート。
「正しく・速く動くか」を問う前に「そもそも動くか」を問う。

HW 上限の出典は docs/SOURCES.md（一次情報源・主張と実装の一致）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .autotune import TileConfig


@dataclass(frozen=True)
class HwLimits:
    """per-block / per-thread のハード上限（超えると launch / compile が失敗する）。

    occupancy.HW は「常駐数を決める容量」、こちらは「越えたら起動不能の壁」。
    両者は別物（容量 ≠ 起動可否）。
    """

    name: str
    warp_size: int
    max_smem_per_block: int   # 1 ブロック/workgroup が使える共有メモリ/LDS 上限（バイト）
    max_threads_per_block: int
    max_regs_per_thread: int


# 一次情報源の per-block 上限（代表アーキ）。出典は docs/SOURCES.md。
# 重要な差分: smem/LDS per block が NVIDIA 227KB vs AMD 64KB（起動可否を分ける主因）。
LIMITS: dict[str, HwLimits] = {
    # NVIDIA H100 (Hopper, CC 9.0): Hopper Tuning Guide
    #   block あたり最大 227KB smem（opt-in dynamic）/ 1024 threads/block / 255 regs/thread
    "nvidia":   HwLimits("nvidia(H100/Hopper)", 32, 227 * 1024, 1024, 255),
    # AMD MI300X (CDNA3, gfx942): ROCm GPU arch specs
    #   LDS 64KiB/workgroup / 1024 work-items/workgroup / VGPR 256/lane
    "amd_cdna": HwLimits("amd_cdna(MI300X/CDNA3)", 64, 64 * 1024, 1024, 256),
    # AMD RX 7900 XTX (RDNA3, gfx1100): GPUOpen / RDNA3 ISA
    #   LDS 64KiB usable/workgroup / 1024 work-items / VGPR 256/lane(wave32)
    "amd_rdna": HwLimits("amd_rdna(RX7900XTX/RDNA3)", 32, 64 * 1024, 1024, 256),
}


@dataclass
class ResourceCheck:
    resource: str   # "shared_mem" | "threads" | "registers"
    required: int
    limit: int

    @property
    def fits(self) -> bool:
        return self.required <= self.limit

    def to_text(self) -> str:
        mark = "ok " if self.fits else "OVER"
        return (f"[{mark}] {self.resource:11s} "
                f"required={self.required} limit={self.limit}")


@dataclass
class FeasibilityReport:
    vendor: str
    checks: list[ResourceCheck] = field(default_factory=list)

    @property
    def launchable(self) -> bool:
        return all(c.fits for c in self.checks)

    @property
    def blockers(self) -> list[ResourceCheck]:
        return [c for c in self.checks if not c.fits]

    def to_text(self) -> str:
        status = "LAUNCHABLE" if self.launchable else "NOT-LAUNCHABLE"
        lines = [f"feasibility → {self.vendor} ({status})"]
        for c in self.checks:
            lines.append("  " + c.to_text())
        return "\n".join(lines)


def check(cfg: TileConfig, vendor: str, *,
          regs_per_thread: int = 64, dtype_bytes: int = 2) -> FeasibilityReport:
    """タイル構成が vendor で *起動できるか* を判定する（占有率の前段ゲート）。

    occupancy.estimate が occ を計算するのと同じ資源式を使うが、ここでは
    「常駐数」でなく「per-block のハード上限を越えていないか」を問う。
    越えていれば launch/compile が失敗する（= 移植ブロッカー）。
    """
    lim = LIMITS.get(vendor)
    if lim is None:
        raise ValueError(f"unknown vendor: {vendor}")

    threads = cfg.num_warps * lim.warp_size
    smem = (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n) * dtype_bytes * cfg.num_stages

    rep = FeasibilityReport(vendor=vendor)
    rep.checks.append(ResourceCheck("shared_mem", smem, lim.max_smem_per_block))
    rep.checks.append(ResourceCheck("threads", threads, lim.max_threads_per_block))
    rep.checks.append(ResourceCheck("registers", regs_per_thread, lim.max_regs_per_thread))
    return rep


def cross_vendor_feasibility(cfg: TileConfig,
                             vendors=("nvidia", "amd_cdna", "amd_rdna"),
                             **kw) -> dict[str, FeasibilityReport]:
    return {v: check(cfg, v, **kw) for v in vendors}


def first_vendor_only(cfg: TileConfig,
                      a: str = "nvidia", b: str = "amd_cdna", **kw) -> list[str]:
    """片方のベンダーでだけ起動できる（= 単一ソース約束の破綻）リソースを抽出。

    「NVIDIA で動くから OK」と思った構成が AMD で *起動すらしない* 罠を可視化する。
    """
    ra, rb = check(cfg, a, **kw), check(cfg, b, **kw)
    out: list[str] = []
    for ca, cb in zip(ra.checks, rb.checks):
        if ca.fits != cb.fits:
            ok, ng = (a, b) if ca.fits else (b, a)
            ng_check = ca if not ca.fits else cb
            out.append(
                f"{ca.resource}: {ok} は起動可・{ng} は起動不能 "
                f"（required={ng_check.required} > {ng} 上限 {ng_check.limit}）")
    return out
