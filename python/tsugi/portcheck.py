"""tsugi.portcheck — クロスベンダー移植性レポート CLI（今すぐ使える製品面）。

GPU codegen 完成を待たずに価値を出す（新視点の戦略的含意）。
DSL カーネルをトレースし、各ベンダーの移植リスクと挙動差を報告する。

使い方:
    python -m tsugi.portcheck            # 内蔵デモ（matmul）
    python -m tsugi.portcheck mykernel.py  # ユーザーカーネル（@tsugi.jit + KERNEL_ARGS）
"""
from __future__ import annotations

import sys

import numpy as np

from . import compile as _compile  # noqa: F401  (公開 API 確認用)
from .portability import analyze, cross_vendor_diff

TARGETS = ("nvidia", "amd_cdna", "amd_rdna")


def _load_user_module(path: str):
    """ユーザーの .py を読み込み (module, block_dims) を返す。

    契約: ファイルは以下を定義する。
      kernel     : @tsugi.jit カーネル
      make_args(): トレース用の引数 tuple を返す関数
      BLOCK_DIMS : (任意) 占有率/warp 解析用の block 次元 tuple
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("user_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "kernel") or not hasattr(mod, "make_args"):
        raise RuntimeError(
            f"{path} は kernel(@tsugi.jit) と make_args() を定義する必要がある")

    import tsugi
    args = mod.make_args()
    traced = tsugi.trace(mod.kernel, args, {}, (0, 0))
    block = getattr(mod, "BLOCK_DIMS", None)
    cfg = getattr(mod, "TILE_CONFIG", None)
    return traced, block, cfg


def _demo_module():
    import tsugi
    from tsugi import tile
    from tsugi.autotune import TileConfig

    # NVIDIA の大きな smem を前提にチューニングした構成（AMD では起動不能になる罠）
    bm = bn = 128
    bk = 64

    @tsugi.jit
    def matmul(a, b, c, M, N, K, BM, BN, BK):
        pm, pn = tsugi.program_id(0), tsugi.program_id(1)
        acc = tile.zeros((BM, BN), tsugi.float32)
        for k in range(0, K, BK):
            acc = tile.dot(tile.load(a, (pm * BM, k), (BM, BK)),
                           tile.load(b, (k, pn * BN), (BK, BN)), acc)
        tile.store(c, (pm * BM, pn * BN), acc.to(tsugi.float16))

    a = np.zeros((256, 256), np.float16)
    mod = tsugi.trace(
        matmul, (a, a.copy(), a.copy(), 256, 256, 256, bm, bn, bk), {}, (0, 0))
    cfg = TileConfig(block_m=bm, block_n=bn, block_k=bk, num_stages=4, num_warps=8)
    return mod, (bm,), cfg


def report(module, block_dims, cfg=None) -> int:
    print("=== Tsugi portability report ===\n")
    worst = 0
    for t in TARGETS:
        rep = analyze(module, t, block_dims=block_dims, cfg=cfg)
        print(rep.to_text())
        print()
        worst = max(worst, int(rep.max_risk))
    # 起動可能性（占有率より上流のゲート）— 構成があれば categorical に判定
    if cfg is not None:
        from .feasibility import cross_vendor_feasibility, first_vendor_only
        print(f"--- 起動可能性（構成 {cfg.key()}）---")
        for _v, f in cross_vendor_feasibility(cfg).items():
            print("  " + f.to_text().replace("\n", "\n  "))
        only = first_vendor_only(cfg, "nvidia", "amd_cdna")
        if only:
            print("  ! 単一ソース約束の破綻（片方でしか起動しない）:")
            for o in only:
                print(f"      {o}")
        print()

    diffs = cross_vendor_diff(module, ("nvidia", "amd_cdna"))
    if diffs:
        print("--- ベンダー間で挙動差の疑い ---")
        for d in diffs:
            print(f"  ! {d}")
    # 累積を伴う matmul があれば導出許容の目安を併記（検証層の統合）
    n_dots = sum(1 for k in module.kernels for op in k.body if op.kind == "dot")
    if n_dots >= 2:
        from .envelope import certify_gemm
        from .tolerance import explain
        K_est = n_dots * 32  # 反復数×典型 BK の粗い推定
        print("\n--- 数値等価性の目安（導出許容）---")
        print("  " + explain(K_est, "float16"))
        print("  → 実 GPU 比較は equivalence.compare_gemm(nv_out, amd_out, K) で照合")
        # 認証の前提を明示（この保証が有効な動作範囲）。本番入力の逸脱は要 runtime 検査。
        env = certify_gemm(K_est, "float16", scale=1.0)
        print("\n--- 認証エンベロープ（この保証が有効な前提）---")
        print("  " + env.to_text())
        print("  → 本番入力がこの範囲を逸脱したら envelope.check_tensor で実行時検出（oracle 不要）")
    print(f"\n判定: {'移植可（要注意点あり）' if worst < 3 else '移植ブロッカーあり'}")
    return 0 if worst < 3 else 1


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv:
        try:
            mod, block, cfg = _load_user_module(argv[0])
            print(f"[portcheck] loaded user kernel: {argv[0]}\n")
            return report(mod, block, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[portcheck] ユーザーカーネル読込失敗: {e}")
            print("[portcheck] 契約: kernel(@tsugi.jit) + make_args() を定義")
            return 2
    mod, block, cfg = _demo_module()
    return report(mod, block, cfg)


if __name__ == "__main__":
    sys.exit(main())
