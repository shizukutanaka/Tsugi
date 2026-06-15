"""Tsugi リファレンス correctness テスト（CPU/NumPy・今すぐ実行可能）。

タイルDSLで書いた matmul が NumPy 真値と一致するかを検証する。
これは GPU バックエンドが後で一致すべき *正しさの真値*。

実行: python tests/correctness/test_reference.py
GPU 不要。ここが緑 = リファレンス意味論が正しい。
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


def run_matmul(M, N, K, BM, BN, BK):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    c = np.zeros((M, N), dtype=np.float16)
    grid = (tsugi.cdiv(M, BM), tsugi.cdiv(N, BN))
    matmul_kernel[grid](a, b, c, M, N, K, BM, BN, BK)
    return a, b, c


def test_matmul_square():
    M = N = K = 128
    a, b, c = run_matmul(M, N, K, BM=32, BN=32, BK=32)
    ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    err = np.max(np.abs(c.astype(np.float32) - ref.astype(np.float32)))
    assert err < 1e-1, f"max abs error {err} too high"  # FP16 accum tol
    return err


def test_matmul_nonsquare_padded():
    # ブロックで割り切れない形状（パディング経路）
    M, N, K = 100, 70, 96
    a, b, c = run_matmul(M, N, K, BM=32, BN=32, BK=32)
    ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    err = np.max(np.abs(c.astype(np.float32) - ref.astype(np.float32)))
    assert err < 1e-1, f"max abs error {err} too high"
    return err


def test_rmsnorm():
    # reduce + rsqrt + elementwise を tile op で組んで真値と照合
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 256)).astype(np.float32)
    xt = tile.zeros((64, 256), tsugi.float32)
    xt.data = x
    sq = xt * xt                                  # tile elementwise
    n = x.shape[1]
    ssum = tile.reduce(sq, axis=1, kind="sum")    # tile reduce
    inv = tile.zeros((64, 1))                     # placeholder, filled below
    inv.data = 1.0 / np.sqrt(ssum.data / n + 1e-6)
    got = (xt * inv).data
    ref = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6)
    assert np.allclose(got, ref, atol=1e-5)
    return float(np.max(np.abs(got - ref)))


def test_attention():
    # fused attention: dot(softmax(dot(q,k^T)/sqrt(d)), v) のリファレンス
    rng = np.random.default_rng(2)
    seq, d = 64, 32
    q = rng.standard_normal((seq, d)).astype(np.float32)
    k = rng.standard_normal((seq, d)).astype(np.float32)
    v = rng.standard_normal((seq, d)).astype(np.float32)
    qt, kt, vt = tile.zeros((seq, d)), tile.zeros((seq, d)), tile.zeros((seq, d))
    qt.data, kt.data, vt.data = q, k, v
    scores = tile.dot(qt, tile.zeros((d, seq)))   # placeholder
    scores.data = (q @ k.T) / np.sqrt(d)
    m = tile.reduce(scores, axis=1, kind="max")
    e = tile.exp(tile.zeros(scores.shape))
    e.data = np.exp(scores.data - m.data)
    s = tile.reduce(e, axis=1, kind="sum")
    p = tile.zeros(scores.shape)
    p.data = e.data / s.data
    out = tile.dot(p, vt).data
    # numpy 真値
    sc = (q @ k.T) / np.sqrt(d)
    sc = sc - sc.max(axis=1, keepdims=True)
    pr = np.exp(sc)
    pr = pr / pr.sum(axis=1, keepdims=True)
    ref = pr @ v
    err = float(np.max(np.abs(out - ref)))
    assert err < 1e-4, f"attention err {err}"
    return err


def main() -> int:
    results = []
    for fn in (test_matmul_square, test_matmul_nonsquare_padded, test_rmsnorm, test_attention):
        try:
            err = fn()
            print(f"[PASS] {fn.__name__:28s} max_err={err:.3e}")
            results.append(True)
        except AssertionError as e:
            print(f"[FAIL] {fn.__name__:28s} {e}")
            results.append(False)
    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}: {sum(results)}/{len(results)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
