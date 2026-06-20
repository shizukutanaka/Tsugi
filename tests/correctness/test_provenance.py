"""証明書の陳腐化検出テスト（verdict を環境フィンガープリントに束ねる）。

検証は point-in-time。環境(driver/library/compiler)が変われば検証済み等価は無効化
されうる。stale 検出で「一度認証=永遠に有効」の誤りを防ぐ。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.provenance import (  # noqa: E402
    certify,
    changed_fields,
    env_fingerprint,
    fingerprint_hash,
    is_stale,
)


def test_same_env_not_stale():
    c = certify("portable", rocm="6.0", driver="550.0")
    assert not is_stale(c, rocm="6.0", driver="550.0")


def test_stack_upgrade_makes_cert_stale():
    c = certify("equivalent", rocm="6.0", driver="550.0")
    assert is_stale(c, rocm="6.0", driver="560.0")     # driver 更新で stale
    assert is_stale(c, rocm="6.1", driver="550.0")     # rocm 更新で stale


def test_changed_fields_pinpoints_drift():
    c = certify("portable", rocm="6.0", driver="550.0", compiler="llvm-18")
    diff = changed_fields(c, rocm="6.1", driver="550.0", compiler="llvm-18")
    assert diff == {"rocm": ("6.0", "6.1")}            # 変化したフィールドのみ
    assert changed_fields(c, rocm="6.0", driver="550.0", compiler="llvm-18") == {}


def test_fingerprint_is_order_independent():
    h1 = fingerprint_hash({"a": "1", "b": "2"})
    h2 = fingerprint_hash({"b": "2", "a": "1"})
    assert h1 == h2                                     # キー順に依存しない
    assert "numpy" in env_fingerprint()                # 基本フィールドを捕捉


def main() -> int:
    ok = True
    tests = [
        test_same_env_not_stale,
        test_stack_upgrade_makes_cert_stale,
        test_changed_fields_pinpoints_drift,
        test_fingerprint_is_order_independent,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    print(certify("portable", rocm="6.0").to_text())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
