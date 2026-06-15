"""Tsugi tile DSL — matmul 最小例（SPEC.md §1.2）。
実行には LLVM/MLIR + NVIDIA/AMD GPU が必要。"""
import tsugi
from tsugi import tile


@tsugi.jit
def matmul(a, b, c, M, N, K,
           BM: tsugi.constexpr, BN: tsugi.constexpr, BK: tsugi.constexpr):
    pid_m, pid_n = tsugi.program_id(0), tsugi.program_id(1)
    acc = tile.zeros((BM, BN), tsugi.float32)
    for k in range(0, K, BK):
        acc += tile.dot(tile.load(a, (pid_m, k), (BM, BK)),
                        tile.load(b, (k, pid_n), (BK, BN)))
    tile.store(c, (pid_m, pid_n), acc.to(tsugi.float16))
