"""occupancy 推定のテスト。同一構成がベンダー間で占有率が異なることを実証。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.autotune import TileConfig  # noqa: E402
from tsugi.occupancy import (  # noqa: E402
    cross_vendor_occupancy,
    estimate,
    occupancy_gap,
)


def test_occupancy_in_range():
    cfg = TileConfig(64, 64, 32, 3, 4)
    for v in ("nvidia", "amd_cdna", "amd_rdna"):
        e = estimate(cfg, v)
        assert 0.0 <= e.occupancy <= 1.0, f"{v}: {e.occupancy}"


def test_large_tile_drops_occupancy():
    # 大タイル + 多ステージ → 共有メモリ制約で占有率低下
    small = estimate(TileConfig(32, 32, 32, 2, 4), "amd_cdna")
    large = estimate(TileConfig(128, 128, 64, 4, 8), "amd_cdna")
    assert large.occupancy <= small.occupancy, \
        f"large should not exceed small: {large.occupancy} vs {small.occupancy}"


def test_cross_vendor_difference_exists():
    # 妥当な常駐構成で NVIDIA と AMD の占有率が変わる（warp32/64・LDS差）
    cfg = TileConfig(64, 64, 32, 3, 4)
    occ = cross_vendor_occupancy(cfg)
    vals = {v: e.occupancy for v, e in occ.items()}
    assert len(set(vals.values())) > 1, f"no cross-vendor occupancy difference: {vals}"


def test_occupancy_gap_nonneg():
    cfg = TileConfig(64, 64, 32, 3, 4)
    gap = occupancy_gap(cfg, "nvidia", "amd_cdna")
    assert gap >= 0.0


def test_unknown_vendor_raises():
    try:
        estimate(TileConfig(32, 32, 32, 2, 4), "intel")
        raise AssertionError("should raise")
    except ValueError:
        pass


def main() -> int:
    ok = True
    tests = [
        test_occupancy_in_range,
        test_large_tile_drops_occupancy,
        test_cross_vendor_difference_exists,
        test_occupancy_gap_nonneg,
        test_unknown_vendor_raises,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: 大タイル構成のベンダー別占有率
    cfg = TileConfig(64, 64, 32, 3, 4)
    print(f"\n--- 構成 {cfg.key()} の占有率（一次情報源の実値・docs/SOURCES.md）---")
    for _v, e in cross_vendor_occupancy(cfg).items():
        print("  " + e.to_text())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
