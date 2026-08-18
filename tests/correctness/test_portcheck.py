"""portcheck のユーザーカーネル読込テスト。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.portcheck import _demo_module, _load_user_module, report  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "user_kernel.py"


def test_report_returns_audit_exit_code_not_collapsed():
    """CLI の終了コードは CI ゲート契約（OK/INFO=0・WARN=1・BLOCK=2）に忠実であること。

    従来 `report()` は `0 if portable else 1` で BLOCK(2) と WARN(1) を 1 に潰していた。
    CI が「exit>=2 でのみ失敗」設定だと BLOCK が素通りする（プロセス層の偽OK）。
    デモ構成は AMD で起動不能＝BLOCK なので、終了コードは 2 でなければならない。
    """
    import io
    from contextlib import redirect_stdout

    from tsugi.audit import audit

    mod, block, cfg = _demo_module()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = report(mod, block, cfg)
    a = audit(mod, cfg, block_dims=block)
    assert a.max_risk.name == "BLOCK"          # デモは起動不能 BLOCK
    assert rc == a.exit_code == 2, f"BLOCK が exit=2 でなく {rc}（WARN と潰れている＝偽OK）"


def test_demo_flags_launch_block_for_amd():
    # 新視点3: NVIDIA で起動する構成が AMD で起動不能（BLOCK）と報告されること
    import io
    from contextlib import redirect_stdout
    mod, block, cfg = _demo_module()
    buf = io.StringIO()
    with redirect_stdout(buf):
        report(mod, block, cfg)
    out = buf.getvalue()
    assert "BLOCK" in out and "起動" in out, "launch-feasibility BLOCK missing"
    assert "起動不能" in out


def test_load_user_kernel():
    mod, block, cfg = _load_user_module(str(EXAMPLE))
    assert "dot" in mod.op_kinds()
    assert block == (48,)


def test_missing_contract_raises():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("x = 1\n")
        bad = f.name
    try:
        _load_user_module(bad)
        raise AssertionError("should raise")
    except RuntimeError:
        pass




def test_report_includes_tolerance_for_accumulation(capsys=None):
    import io
    from contextlib import redirect_stdout
    mod, block, cfg = _demo_module()
    buf = io.StringIO()
    with redirect_stdout(buf):
        report(mod, block, cfg)
    out = buf.getvalue()
    assert "導出許容" in out or "atol" in out, "tolerance guidance missing"


def main() -> int:
    ok = True
    for t in (test_report_returns_audit_exit_code_not_collapsed, test_demo_flags_launch_block_for_amd, test_load_user_kernel, test_missing_contract_raises, test_report_includes_tolerance_for_accumulation):
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
