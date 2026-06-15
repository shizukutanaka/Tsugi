"""統合ファサード tsugi.audit のテスト（検証層を 1 判定に束ねる）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.audit import audit  # noqa: E402
from tsugi.portcheck import _demo_module  # noqa: E402
from tsugi.report import Risk  # noqa: E402


def test_audit_aggregates_all_static_phases():
    mod, block, cfg = _demo_module()
    a = audit(mod, cfg, block_dims=block)
    names = {p.name.split()[0] for p in a.phases}
    # 静的層（portability/feasibility/occupancy/numerics）＋ runtime チェックリスト
    assert {"portability", "feasibility", "occupancy", "numerics", "runtime"} <= names


def test_audit_verdict_from_static_only():
    # デモ構成は AMD で起動不能 → 静的判定は BLOCK（移植ブロッカー）
    mod, block, cfg = _demo_module()
    a = audit(mod, cfg, block_dims=block)
    assert a.max_risk == Risk.BLOCK
    assert not a.portable


def test_runtime_phase_excluded_from_verdict():
    # runtime 層は実機データ待ちゆえ判定に影響しない（静的層のみで verdict）
    mod, block, cfg = _demo_module()
    a = audit(mod, cfg, block_dims=block)
    rt = [p for p in a.phases if p.when == "runtime"]
    assert len(rt) == 1
    assert all(p.when == "static" for p in a.static_phases)
    assert a.max_risk == max(p.max_risk for p in a.static_phases)


def test_audit_text_has_lifecycle_and_verdict():
    mod, block, cfg = _demo_module()
    txt = audit(mod, cfg, block_dims=block).to_text()
    assert "起動不能" in txt          # feasibility BLOCK
    assert "導出許容" in txt          # numerics
    assert "要実機データ" in txt      # runtime チェックリスト
    assert "判定（静的層）" in txt


def test_audit_without_cfg_still_runs_portability():
    # 構成なしでも移植性と（dot があれば）数値目安は出る
    mod, block, cfg = _demo_module()
    a = audit(mod, None, block_dims=block)
    names = {p.name.split()[0] for p in a.phases}
    assert "portability" in names
    assert "feasibility" not in names   # cfg 無しでは起動可能性は判定不可


def main() -> int:
    ok = True
    tests = [
        test_audit_aggregates_all_static_phases,
        test_audit_verdict_from_static_only,
        test_runtime_phase_excluded_from_verdict,
        test_audit_text_has_lifecycle_and_verdict,
        test_audit_without_cfg_still_runs_portability,
    ]
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
