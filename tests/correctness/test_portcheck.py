"""portcheck のユーザーカーネル読込テスト。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.portcheck import _demo_module, _load_user_module, report  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "user_kernel.py"


def test_demo_module_reports():
    mod, block, cfg = _demo_module()
    rc = report(mod, block, cfg)
    assert rc in (0, 1)


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
    for t in (test_demo_module_reports, test_demo_flags_launch_block_for_amd, test_load_user_kernel, test_missing_contract_raises, test_report_includes_tolerance_for_accumulation):
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
