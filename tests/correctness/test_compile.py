"""tsugi.compile の e2e テスト（上流パイプライン統合）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tsugi  # noqa: E402
from tsugi import tile  # noqa: E402


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


def _args():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 64)).astype(np.float16)
    b = rng.standard_normal((64, 64)).astype(np.float16)
    c = np.zeros((64, 64), dtype=np.float16)
    return (a, b, c, 64, 64, 64, 32, 32, 32)


def test_compile_nvidia_and_amd():
    for target in ("nvidia", "amd_cdna"):
        art = tsugi.compile(matmul_kernel, _args(), target=target)
        assert art.target == target
        assert "tsugi_tile.dot" in art.mlir
        assert "UNSUPPORTED" not in art.plan_text


def test_invalid_target_raises():
    try:
        tsugi.compile(matmul_kernel, _args(), target="bogus")
        raise AssertionError("should raise")
    except ValueError:
        pass


def test_machine_code_honestly_unimplemented():
    try:
        tsugi.compile(matmul_kernel, _args(), emit_machine_code=True)
        raise AssertionError("should raise NotImplementedError")
    except NotImplementedError:
        pass


def main() -> int:
    ok = True
    for t in (test_compile_nvidia_and_amd, test_invalid_target_raises,
              test_machine_code_honestly_unimplemented):
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    art = tsugi.compile(matmul_kernel, _args(), target="nvidia")
    print(f"\n{art!r}")
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
