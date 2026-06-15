"""Tsugi 不変条件チェッカ（CI verify・ハーネスの verify パターン）。

プロジェクトの不変条件を機械的に検証する。CI で fail on error。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / "python"

INVARIANTS: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    INVARIANTS.append((name, cond))
    print(f"[{'OK ' if cond else 'ERR'}] {name}")


def _grep(pattern: str, *globs: str) -> list[str]:
    hits = []
    self_path = Path(__file__).resolve()
    for g in globs:
        for p in ROOT.rglob(g):
            if "build" in p.parts or "__pycache__" in p.parts:
                continue
            if p.resolve() == self_path:  # verify 自身のスキャンパターンを除外
                continue
            try:
                for ln in p.read_text(encoding="utf-8").splitlines():
                    if pattern in ln:
                        hits.append(f"{p}: {ln.strip()}")
            except Exception:  # noqa: BLE001
                pass
    return hits


def main() -> int:
    # 1. 課金コード不在（絶対禁止）
    billing = [h for h in _grep("stripe", "*.py") + _grep("Stripe", "*.py")
               if "ADR" not in h and "FAQ" not in h]
    check("no billing/Stripe code", not billing)

    # 2. バイナリ CUDA 変換の不在（ADR-002）
    zluda = [h for h in _grep("zluda", "*.py") if "FAQ" not in h and "ADR" not in h]
    check("no binary CUDA translation (ADR-002)", not zluda)

    # 3. PII パターン不在
    pii = [h for h in _grep("@gmail", "*.py") + _grep("@yahoo", "*.py")]
    check("no PII in code", not pii)

    # 4. リファレンス correctness 全通過
    r = subprocess.run([sys.executable, str(ROOT / "tests/correctness/test_reference.py")],
                       capture_output=True, text=True)
    check("reference correctness PASS", r.returncode == 0)

    # 5. tracer / lowering / autotune 全通過
    for t in ("test_tracer.py", "test_lowering.py", "test_autotune.py"):
        rr = subprocess.run([sys.executable, str(ROOT / "tests/correctness" / t)],
                            capture_output=True, text=True)
        check(f"{t} PASS", rr.returncode == 0)

    # 6. dot は行列コア intrinsic へ写像（ADR-004）
    sys.path.insert(0, str(PY))
    from tsugi.lowering import VENDOR_LOWERING
    check("dot→wmma (NVIDIA, ADR-004)", "wmma" in VENDOR_LOWERING["dot"]["nvidia"])
    check("dot→mfma (AMD CDNA, ADR-004)", "mfma" in VENDOR_LOWERING["dot"]["amd_cdna"])

    # 7. machine-code emission は正直に未実装
    import tsugi
    try:
        tsugi.compile(lambda: None, (), emit_machine_code=True)
        check("machine-code honestly unimplemented", False)
    except NotImplementedError:
        check("machine-code honestly unimplemented", True)


    # 8. equivalence 検出器が発散を捕まえる（新視点の柱）
    import numpy as np
    from tsugi.equivalence import compare, simulate_vendor_matmul
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 1024)).astype(np.float16)
    b = rng.standard_normal((1024, 64)).astype(np.float16)
    good = simulate_vendor_matmul(a, b, accum="f32")
    bad = simulate_vendor_matmul(a, b, accum="f16", split_k=64)
    check("equivalence detects f16-accum divergence", not compare(good, bad, "float16").equivalent)
    check("equivalence accepts identical", compare(good, good.copy(), "float16").equivalent)


    # 9. occupancy: 同一構成がベンダー間で差を持つ（移植落とし穴の検出）
    from tsugi.autotune import TileConfig
    from tsugi.occupancy import cross_vendor_occupancy
    occ = cross_vendor_occupancy(TileConfig(64, 64, 32, 3, 4))
    vals = {v: e.occupancy for v, e in occ.items()}
    check("occupancy differs across vendors", len(set(vals.values())) > 1)


    # 10. tolerance: 導出許容が K 依存（固定値でない・新視点）
    from tsugi.tolerance import expected_gemm_abs_error
    check("derived tolerance grows with K",
          expected_gemm_abs_error(4096, "float16") > expected_gemm_abs_error(64, "float16"))

    # 11. feasibility: 起動不能を BLOCK に分類する（占有率と区別・新視点）
    from tsugi import portability
    from tsugi.feasibility import check as feasible_check
    from tsugi.ir import Module
    nv_tuned = TileConfig(128, 128, 64, 4, 8)   # NVIDIA で起動・AMD で起動不能
    check("feasibility: NVIDIA launchable", feasible_check(nv_tuned, "nvidia").launchable)
    check("feasibility: AMD CDNA NOT launchable", not feasible_check(nv_tuned, "amd_cdna").launchable)
    check("unfeasible config classified as BLOCK (not WARN)",
          portability.analyze(Module(kernels=[]), "amd_cdna", cfg=nv_tuned).max_risk
          == portability.Risk.BLOCK)

    # 12. propagation: per-kernel 等価 ⇏ per-model 等価（深さで発散が累積・新視点4）
    from tsugi.propagation import GraphOp, model_tolerance
    single = model_tolerance([GraphOp("matmul", K=256)])
    deep = model_tolerance([GraphOp("matmul", K=256)] * 12)
    check("model-level divergence exceeds single-kernel (composition)", deep > single)
    amp = model_tolerance([GraphOp("matmul", K=128), GraphOp("softmax", cond=8.0)])
    flat = model_tolerance([GraphOp("matmul", K=128), GraphOp("softmax", cond=1.0)])
    check("ill-conditioned op amplifies divergence", amp > flat)

    failed = [n for n, c in INVARIANTS if not c]
    print(f"\n{'VERIFY PASS' if not failed else 'VERIFY FAIL'}: "
          f"{len(INVARIANTS) - len(failed)}/{len(INVARIANTS)} invariants")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
