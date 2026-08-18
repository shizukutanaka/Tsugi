"""ユーザーカーネル例（portcheck 契約）。
python -m tsugi.portcheck examples/user_kernel.py で解析される。"""
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "python"))

import tsugi
from tsugi import tile

BLOCK_DIMS = (48,)   # わざと wavefront(64) 非倍数 → AMD で WARN を誘発


@tsugi.jit
def kernel(a, b, c, M, N, K, BM, BN, BK):
    pm, pn = tsugi.program_id(0), tsugi.program_id(1)
    acc = tile.zeros((BM, BN), tsugi.float32)
    for k in range(0, K, BK):
        acc = tile.dot(tile.load(a, (pm * BM, k), (BM, BK)),
                       tile.load(b, (k, pn * BN), (BK, BN)), acc)
    tile.store(c, (pm * BM, pn * BN), acc.to(tsugi.float16))


def make_args():
    a = np.zeros((96, 96), np.float16)
    return (a, a.copy(), a.copy(), 96, 96, 96, 48, 48, 48)
