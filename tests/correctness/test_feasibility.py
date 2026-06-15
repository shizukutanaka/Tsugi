"""feasibility 検証のテスト（新視点・第3ラウンド）。

「占有率が低い」と「そもそも起動できない」を区別する。NVIDIA で起動する構成が
AMD では LDS 上限超過で起動不能になる罠を categorical に検出することを実証。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi import portability  # noqa: E402
from tsugi.autotune import TileConfig  # noqa: E402
from tsugi.feasibility import (  # noqa: E402
    check,
    cross_vendor_feasibility,
    first_vendor_only,
)
from tsugi.ir import Module  # noqa: E402


def test_small_config_launchable_everywhere():
    cfg = TileConfig(32, 32, 32, 2, 4)
    for v in ("nvidia", "amd_cdna", "amd_rdna"):
        assert check(cfg, v).launchable, f"{v} should launch small config"


def test_large_smem_breaks_amd_not_nvidia():
    # (128*64 + 64*128)*2*4 = 131072 bytes = 128KB
    #   NVIDIA Hopper 227KB/block → 起動可、AMD CDNA LDS 64KB → 起動不能
    cfg = TileConfig(128, 128, 64, 4, 8)
    assert check(cfg, "nvidia").launchable, "NVIDIA (227KB) should launch"
    amd = check(cfg, "amd_cdna")
    assert not amd.launchable, "AMD CDNA (64KB LDS) must NOT launch"
    assert any(b.resource == "shared_mem" for b in amd.blockers)


def test_first_vendor_only_flags_single_source_break():
    cfg = TileConfig(128, 128, 64, 4, 8)
    only = first_vendor_only(cfg, "nvidia", "amd_cdna")
    assert only, "should flag a vendor-asymmetric launch failure"
    assert any("shared_mem" in o for o in only)


def test_unfeasible_is_BLOCK_not_just_WARN():
    # 回帰防止: occ=0% を「遅い WARN」でなく「起動不能 BLOCK」に分類する
    cfg = TileConfig(128, 128, 64, 4, 8)
    rep = portability.analyze(Module(kernels=[]), "amd_cdna", cfg=cfg)
    assert rep.max_risk == portability.Risk.BLOCK, \
        f"unfeasible config must be BLOCK, got {rep.max_risk.name}"
    assert any(f.op == "launch" for f in rep.findings)
    # NVIDIA では同じ構成が BLOCK にならない（起動はする）
    nv = portability.analyze(Module(kernels=[]), "nvidia", cfg=cfg)
    assert nv.max_risk < portability.Risk.BLOCK, "NVIDIA should be launchable"


def test_unknown_vendor_raises():
    try:
        check(TileConfig(32, 32, 32, 2, 4), "intel")
        raise AssertionError("should raise")
    except ValueError:
        pass


def main() -> int:
    ok = True
    tests = [
        test_small_config_launchable_everywhere,
        test_large_smem_breaks_amd_not_nvidia,
        test_first_vendor_only_flags_single_source_break,
        test_unfeasible_is_BLOCK_not_just_WARN,
        test_unknown_vendor_raises,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: NVIDIA でチューニングした構成が AMD で起動不能になる実例
    cfg = TileConfig(128, 128, 64, 4, 8)
    print(f"\n--- 構成 {cfg.key()} の起動可能性（per-block 上限・docs/SOURCES.md）---")
    for _v, f in cross_vendor_feasibility(cfg).items():
        for line in f.to_text().splitlines():
            print("  " + line)
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
