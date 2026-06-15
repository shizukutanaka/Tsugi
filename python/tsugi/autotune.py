"""Tsugi autotuning — タイル構成の探索（SPEC.md §4）。

各ベンダー独立にタイルサイズ・レイアウト・ステージ数を探索する。
リファレンス（本ファイル）は探索ロジックのみ（CPU 実行可）。実 GPU 計測との
接続は Phase 3（要実機）。warp(NVIDIA=32) と wavefront(AMD=64) の差を吸収する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product


@dataclass(frozen=True)
class TileConfig:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int

    def key(self) -> str:
        return f"m{self.block_m}_n{self.block_n}_k{self.block_k}_s{self.num_stages}_w{self.num_warps}"


# ベンダー別の妥当性制約（warp/wavefront サイズの差を反映）
_VENDOR_LANES = {"nvidia": 32, "amd": 64}


@dataclass
class SearchSpace:
    block_m: tuple[int, ...] = (32, 64, 128)
    block_n: tuple[int, ...] = (32, 64, 128)
    block_k: tuple[int, ...] = (32, 64)
    num_stages: tuple[int, ...] = (2, 3, 4)
    num_warps: tuple[int, ...] = (4, 8)
    pruned: list[str] = field(default_factory=list)

    def candidates(self, vendor: str, shared_mem_bytes: int = 64 * 1024) -> list[TileConfig]:
        """妥当な構成のみ列挙（共有メモリ上限・lane 整合でプルーン）。"""
        lanes = _VENDOR_LANES.get(vendor)
        if lanes is None:
            raise ValueError(f"unknown vendor: {vendor}")
        out: list[TileConfig] = []
        for bm, bn, bk, ns, nw in product(
            self.block_m, self.block_n, self.block_k, self.num_stages, self.num_warps
        ):
            cfg = TileConfig(bm, bn, bk, ns, nw)
            # 共有メモリ概算: (BM*BK + BK*BN) * 2bytes(f16) * stages
            smem = (bm * bk + bk * bn) * 2 * ns
            if smem > shared_mem_bytes:
                self.pruned.append(cfg.key())
                continue
            # threads = num_warps * lanes が BM をカバーできるか（粗い妥当性）
            if nw * lanes < bm:
                self.pruned.append(cfg.key())
                continue
            out.append(cfg)
        return out


def grid_search(vendor: str, space: SearchSpace | None = None) -> list[TileConfig]:
    """v0.1: 全候補を列挙（実計測は Phase 3）。将来ベイズ最適化へ。"""
    space = space or SearchSpace()
    return space.candidates(vendor)
