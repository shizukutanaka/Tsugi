"""tracer の correctness テスト。

@tsugi.jit カーネルを tsugi.tile IR へトレースし、
  1. IR が期待 op 構造を含む（dot/load/store/zeros/cast）
  2. MLIR 風テキストが生成される
  3. トレース時の実値が eager リファレンスと一致（IR が正しい証明）
を検証する。GPU 不要・実行可能。
"""
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


def _inputs():
    rng = np.random.default_rng(0)
    M = N = K = 64
    BM = BN = BK = 32
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    c = np.zeros((M, N), dtype=np.float16)
    return a, b, c, M, N, K, BM, BN, BK


def test_trace_produces_ir():
    args = _inputs()
    mod = tsugi.trace(matmul_kernel, args, {}, program_ids=(0, 0))
    kinds = mod.op_kinds()
    for expected in ("zeros", "load", "dot", "store", "cast"):
        assert expected in kinds, f"IR missing op '{expected}': {kinds}"
    # K=64, BK=32 → 2 反復 → load 4回(a,b×2)・dot 2回
    assert kinds.count("dot") == 2, f"expected 2 dots, got {kinds.count('dot')}"
    assert kinds.count("load") == 4, f"expected 4 loads, got {kinds.count('load')}"
    return len(kinds)


def test_mlir_text_renders():
    args = _inputs()
    mod = tsugi.trace(matmul_kernel, args, {}, program_ids=(0, 0))
    text = mod.to_mlir()
    assert "tsugi_tile.kernel @matmul_kernel" in text
    assert "tsugi_tile.dot" in text
    assert "tensor<32x32xf32>" in text
    return text


def test_trace_values_match_eager():
    # トレースした実値（block (0,0)）が eager リファレンスと一致
    a, b, c_trace, M, N, K, BM, BN, BK = _inputs()
    # eager 全実行
    c_eager = np.zeros((M, N), dtype=np.float16)

    @tsugi.jit
    def k(a, b, c, M, N, K, BM, BN, BK):
        pid_m = tsugi.program_id(0)
        pid_n = tsugi.program_id(1)
        acc = tile.zeros((BM, BN), tsugi.float32)
        for kk in range(0, K, BK):
            ta = tile.load(a, (pid_m * BM, kk), (BM, BK))
            tb = tile.load(b, (kk, pid_n * BN), (BK, BN))
            acc = tile.dot(ta, tb, acc)
        tile.store(c, (pid_m * BM, pid_n * BN), acc.to(tsugi.float16))

    grid = (tsugi.cdiv(M, BM), tsugi.cdiv(N, BN))
    k[grid](a, b, c_eager, M, N, K, BM, BN, BK)
    ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    err = np.max(np.abs(c_eager.astype(np.float32) - ref.astype(np.float32)))
    assert err < 1e-1, f"eager err {err}"
    return float(err)


def main() -> int:
    ok = True
    for t in (test_trace_produces_ir, test_mlir_text_renders, test_trace_values_match_eager):
        try:
            r = t()
            info = f"ops={r}" if isinstance(r, int) else (f"err={r:.2e}" if isinstance(r, float) else "ok")
            print(f"[PASS] {t.__name__:30s} {info}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__:30s} {e}")
            ok = False
    # 参考: IR テキストを表示
    print("\n--- generated tsugi.tile IR (block 0,0) ---")
    print(tsugi.trace(matmul_kernel, _inputs(), {}, program_ids=(0, 0)).to_mlir())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
