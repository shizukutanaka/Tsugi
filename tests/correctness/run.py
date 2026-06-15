"""Tsugi correctness エントリポイント。

2段構え:
  1. リファレンス（CPU/NumPy）correctness — GPU 不要・常に実行・正しさの真値
  2. GPU 両ベンダー数値照合 — 実機があれば実行・無ければ正直に skip

実行: python tests/correctness/run.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _available_vendors() -> list[str]:
    vendors: list[str] = []
    try:
        import torch  # noqa: F401
        import torch.cuda
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            vendors.append("amd" if ("amd" in name or "radeon" in name
                                     or "instinct" in name) else "nvidia")
    except Exception:  # noqa: BLE001
        pass
    return vendors


def main() -> int:
    rc = 0
    print("=== [1] reference correctness (CPU/NumPy oracle) ===")
    for t in ("test_reference.py", "test_autotune.py", "test_tracer.py", "test_lowering.py", "test_compile.py", "test_portability.py", "test_equivalence.py", "test_occupancy.py", "test_portcheck.py", "test_tolerance.py", "test_bf16.py", "test_feasibility.py", "test_propagation.py", "test_envelope.py", "test_calibration.py", "test_nondeterminism.py", "test_decision.py"):
        r = subprocess.run([sys.executable, str(HERE / t)], check=False)
        rc |= r.returncode

    print("\n=== [2] GPU two-vendor correctness ===")
    vendors = _available_vendors()
    if not vendors:
        print("SKIP: no GPU detected (LLVM/MLIR + NVIDIA/AMD hardware required). "
              "未検証 — 実機での実行が必要。")
    else:
        print(f"detected vendors: {vendors}")
        print("kernel execution vs reference: not yet implemented (Phase1, requires GPU)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
