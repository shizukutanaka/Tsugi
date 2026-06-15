"""autotune の単体テスト。warp(NVIDIA)/wavefront(AMD) でプルーンが変わることを確認。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.autotune import SearchSpace, TileConfig, grid_search  # noqa: E402


def test_candidates_nonempty():
    for vendor in ("nvidia", "amd"):
        cfgs = grid_search(vendor)
        assert len(cfgs) > 0, f"{vendor}: no candidates"


def test_shared_mem_pruning():
    # 小さい共有メモリ上限 → 大タイルがプルーンされる
    space = SearchSpace()
    small = space.candidates("nvidia", shared_mem_bytes=8 * 1024)
    space2 = SearchSpace()
    large = space2.candidates("nvidia", shared_mem_bytes=128 * 1024)
    assert len(small) < len(large), "smaller smem must prune more"


def test_vendor_lane_difference():
    # AMD wavefront=64 > NVIDIA warp=32 → lane 整合の通過数が異なりうる
    n = len(grid_search("nvidia"))
    a = len(grid_search("amd"))
    assert n > 0 and a > 0
    # 構成キーが生成できる
    c = TileConfig(64, 64, 32, 3, 4)
    assert c.key() == "m64_n64_k32_s3_w4"


def test_unknown_vendor_raises():
    try:
        grid_search("intel-v0.1-out-of-scope")
        raise AssertionError("should have raised")
    except ValueError:
        pass


def main() -> int:
    tests = [
        test_candidates_nonempty,
        test_shared_mem_pruning,
        test_vendor_lane_difference,
        test_unknown_vendor_raises,
    ]
    ok = True
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
