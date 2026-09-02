"""デバイス（GPU）テンソルを受け取れる `asarray`——検証層の入口を 1 箇所に畳む。

## なぜ要るか

Tsugi の検証層は「**両ベンダーの実機出力**を突き合わせる」ためにある。つまり渡される
テンソルは当然 GPU 上にあり、`np.asarray` は CUDA/HIP テンソルに対して素の
TypeError（`can't convert cuda:0 device type tensor to numpy`）を投げる。

これは 1 箇所の不具合ではなく **種類** だった: `audit_runtime`・`audit_cross_vendor`
（`docs/GPU-BRINGUP.md` が実機初日の最初のコマンドとして指示する関数）に加え、
`equivalence.compare` / `decision.compare_decisions` / `envelope.check_tensor` /
`calibration.check_systematic` など、テンソルを取る公開関数がすべて同じ壊れ方をしていた。

各所に変換を撒くのでなく、**層が使う `asarray` そのものを 1 つ差し替える**。
（Musk 第 3 段階: 部品を足すのでなく、既にある部品を正しくする。）

## 契約

`numpy.asarray` の上位互換。差分は「`.detach().cpu().numpy()` を持つ入力を先に
NumPy 化する」ことだけで、それ以外の挙動・シグネチャは同じ。
"""
from __future__ import annotations

import numpy as np


def _to_host(x):
    """デバイス上のテンソルをホストへ移す（できなければそのまま返す）。

    torch を import しない（任意依存であり、検証層は torch 無しでも動く）。
    `.detach()` → `.cpu()` → `.numpy()` を持つならそれを使う、という duck-typing。
    """
    if x is None or isinstance(x, np.ndarray):
        return x
    for step in ("detach", "cpu"):
        fn = getattr(x, step, None)
        if callable(fn):
            try:
                x = fn()
            except Exception:  # noqa: BLE001 — 変換できなければ元の経路に任せる
                return x
    to_numpy = getattr(x, "numpy", None)
    if callable(to_numpy):
        try:
            return to_numpy()
        except Exception:  # noqa: BLE001
            return x
    return x


def asarray(x, dtype=None, **kwargs):
    """`numpy.asarray` の device-aware 版（None は素通し）。

    デバイステンソルの **列** も受ける（run スタックは `list[tensor]` で渡されうる）。
    要素ごとにホストへ移してから積む。
    """
    if x is None:
        return None
    host = _to_host(x)
    if isinstance(host, (list, tuple)) and host and not isinstance(host[0], (int, float)):
        host = [_to_host(e) for e in host]
    return np.asarray(host, dtype=dtype, **kwargs)
