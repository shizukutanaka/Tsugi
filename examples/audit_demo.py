"""Tsugi audit デモ — 静的 audit() と実行時 audit_runtime() の両 facade を一望する。

実行: python examples/audit_demo.py
GPU 不要。クロスベンダー出力は CPU でシミュレートする（明示）。実機では
シミュレート部分を実 GPU カーネルの出力に置き換えるだけで同じ audit が回る。
"""
from __future__ import annotations

import numpy as np

import tsugi
from tsugi import tile
from tsugi.audit import audit, audit_runtime
from tsugi.envelope import certify_gemm
from tsugi.equivalence import simulate_vendor_matmul


@tsugi.jit
def matmul(a, b, c, M, N, K, BM, BN, BK):
    pm, pn = tsugi.program_id(0), tsugi.program_id(1)
    acc = tile.zeros((BM, BN), tsugi.float32)
    for k in range(0, K, BK):
        acc = tile.dot(tile.load(a, (pm * BM, k), (BM, BK)),
                       tile.load(b, (k, pn * BN), (BK, BN)), acc)
    tile.store(c, (pm * BM, pn * BN), acc.to(tsugi.float16))


def main() -> int:
    # --- 1. 静的監査: traced IR ＋構成だけで移植リスク/起動可能性/数値目安を判定 ---
    bm = bn = 128
    bk = 64                          # NVIDIA 前提の大 smem 構成（AMD で起動不能の罠）
    a = np.zeros((256, 256), np.float16)
    mod = tsugi.trace(matmul, (a, a.copy(), a.copy(), 256, 256, 256, bm, bn, bk),
                      {}, (0, 0))
    from tsugi.autotune import TileConfig
    cfg = TileConfig(block_m=bm, block_n=bn, block_k=bk, num_stages=4, num_warps=8)

    print("################  静的 audit（デプロイ前・GPU 不要）  ################")
    static = audit(mod, cfg, block_dims=(bm,))
    print(static.to_text())

    # --- 2. 実行時監査: 実機/実データのクロスベンダー出力を束ねて判定 ---
    # ここでは CPU で「2 つのベンダー」をシミュレート（実機では実出力を渡す）。
    K = 256
    rng = np.random.default_rng(0)
    lhs = rng.standard_normal((128, K)).astype(np.float16)
    rhs = rng.standard_normal((K, 128)).astype(np.float16)
    nvidia = simulate_vendor_matmul(lhs, rhs, accum="f32", split_k=1)
    amd_ok = simulate_vendor_matmul(lhs, rhs, accum="f32", split_k=8)   # 真に等価（順序差）
    amd_bug = amd_ok * 1.005                                            # 0.5% 系統バグ

    # タスク用 logit（同じ発散がタスク判断に与える影響）
    logits = rng.standard_normal((2000, 1000)).astype(np.float32)
    logits_b = logits + 1e-2 * rng.standard_normal(logits.shape).astype(np.float32)

    env = certify_gemm(K, "float16", scale=float(np.sqrt(np.mean(nvidia.astype(np.float64) ** 2))))

    print("\n############  実行時 audit_runtime（健全な AMD）  ############")
    good = audit_runtime(nvidia, amd_ok, K, env=env, noise_floor=1e-3,
                         logits_a=logits, logits_b=logits_b, flip_budget=0.05)
    print(good.to_text())

    print("\n############  実行時 audit_runtime（0.5% 系統バグの AMD）  ############")
    bad = audit_runtime(nvidia, amd_bug, K, env=env, noise_floor=1e-3)
    print(bad.to_text())

    print("\n要点: 静的監査は実機前に移植ブロッカーを、実行時監査は max_abs の盲点に"
          "隠れる系統バグ（0.5%）を捕まえる。実機では simulate を実 GPU 出力に置換するだけ。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
