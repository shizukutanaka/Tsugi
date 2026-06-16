"""GPU クロスベンダー監査の契約テスト。

- GPU があれば: 実カーネル出力で audit_two_vendors を回す（codegen 完成まで SKIP）。
- GPU が無くても: ハーネスの配線（ノイズ実測 → 監査）を擬似 run で CPU 検証する。
  これにより「実機で何が起きるべきか」の契約が常に実行可能なまま保たれる。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from harness import audit_two_vendors, gpu_vendors  # noqa: E402


def _noisy_vendor(base, eps, vendor_seed):
    """擬似 GPU run: seed ごとに run-to-run ノイズが変わる出力を返す。"""
    def run(seed):
        rng = np.random.default_rng(vendor_seed * 10_000 + seed)
        return base + eps * rng.standard_normal(base.shape).astype(np.float32)
    return run


def test_harness_wires_noise_into_audit_for_equivalent_vendors():
    base = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    rep = audit_two_vendors(_noisy_vendor(base, 1e-4, 1),
                            _noisy_vendor(base, 1e-4, 2), K=256, n_runs=12)
    assert rep.portable          # 真に等価（ノイズのみ）→ ブロッカー無し


def test_harness_flags_real_divergence():
    base = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    good = _noisy_vendor(base, 1e-4, 1)
    bug = _noisy_vendor(base * 1.05, 1e-4, 2)   # 5% 系統スケールバグ
    rep = audit_two_vendors(good, bug, K=256, n_runs=12)
    assert not rep.portable      # 系統バグは BLOCK


def main() -> int:
    print("=== GPU cross-vendor audit contract ===")
    vendors = gpu_vendors()
    if not vendors:
        print("SKIP: no GPU detected — 実カーネル照合は実機が必要（未検証）。")
        print("      ハーネス配線は擬似 run で CPU 検証する（以下）。")

    ok = True
    for t in (test_harness_wires_noise_into_audit_for_equivalent_vendors,
              test_harness_flags_real_divergence):
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
