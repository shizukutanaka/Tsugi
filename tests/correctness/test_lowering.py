"""lowering plan のテスト。matmul IR が NVIDIA/AMD intrinsic へ写像されることを確認。

実 codegen ではない（それは要 LLVM/MLIR + 実機）。op→intrinsic の対応が
全 op で埋まっていること（穴がないこと）を検証する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tsugi  # noqa: E402
from tsugi import tile  # noqa: E402
from tsugi.lowering import VENDOR_LOWERING, coverage, lowering_plan  # noqa: E402


@tsugi.jit
def matmul_kernel(a, b, c, M, N, K, BM, BN, BK):
    pid_m = tsugi.program_id(0)
    pid_n = tsugi.program_id(1)
    acc = tile.zeros((BM, BN), tsugi.float32)
    for k in range(0, K, BK):
        ta = tile.load(a, (pid_m * BM, k), (BM, BK))
        tb = tile.load(b, (k, pid_n * BN), (BK, BN))
        acc = tile.dot(ta, tb, acc)
    tile.store(c, (pid_m * BM, pid_n * BN), acc.to(tsugi.float16))


def _ir():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 64)).astype(np.float16)
    b = rng.standard_normal((64, 64)).astype(np.float16)
    c = np.zeros((64, 64), dtype=np.float16)
    return tsugi.trace(matmul_kernel, (a, b, c, 64, 64, 64, 32, 32, 32), {}, (0, 0))


def test_no_unsupported_for_v01_targets():
    mod = _ir()
    for target in ("nvidia", "amd_cdna", "amd_rdna"):
        plan = lowering_plan(mod, target)
        unsupported = [ln for ln in plan if "UNSUPPORTED" in ln]
        assert not unsupported, f"{target} has unsupported ops: {unsupported}"


def test_coverage_full_for_v01():
    for target in ("nvidia", "amd_cdna", "amd_rdna"):
        cov, total = coverage(target)
        assert cov == total, f"{target}: {cov}/{total} ops covered"


def test_dot_maps_to_matrix_core():
    # ADR-004: dot は各社行列コア intrinsic へ
    assert "wmma" in VENDOR_LOWERING["dot"]["nvidia"]
    assert "mfma" in VENDOR_LOWERING["dot"]["amd_cdna"]


def test_lowering_synced_to_dsl_op_vocabulary():
    # DSL が emit しうる全 op に実ターゲット lowering がある（spec が DSL に同期・drift 検出）
    from tsugi.lowering import unlowered_ops
    from tsugi.tracer import EMITTABLE_OPS
    for target in ("nvidia", "amd_cdna", "amd_rdna", "spirv"):
        assert not unlowered_ops(target), \
            f"{target} missing lowering for: {unlowered_ops(target)}"
    # softmax/norm 系の超越関数も網羅されていること（回帰の番人）
    for op in ("exp", "reduce", "rsqrt", "div"):
        assert op in EMITTABLE_OPS and op in VENDOR_LOWERING


def main() -> int:
    ok = True
    for t in (test_no_unsupported_for_v01_targets, test_coverage_full_for_v01,
              test_dot_maps_to_matrix_core, test_lowering_synced_to_dsl_op_vocabulary):
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考出力: NVIDIA / AMD への lowering plan
    mod = _ir()
    for target in ("nvidia", "amd_cdna"):
        print(f"\n--- lowering plan: {target} ---")
        print("\n".join(lowering_plan(mod, target)))
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
