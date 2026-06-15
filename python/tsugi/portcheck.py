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
from .portability import cross_vendor_diff

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
    """統合ファサード tsugi.audit に委譲して 1 レポートにまとめる（重複排除）。"""
    from .audit import audit

    print("=== Tsugi portability report ===\n")
    # ベンダー間の挙動差の疑い（audit に無い per-pair 差分）を先に併記
    diffs = cross_vendor_diff(module, ("nvidia", "amd_cdna"))
    if diffs:
        print("--- ベンダー間で挙動差の疑い ---")
        for d in diffs:
            print(f"  ! {d}")
        print()
    a = audit(module, cfg, block_dims=block_dims)
    print(a.to_text())
    return 0 if a.portable else 1


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
