"""Tsugi IR の CPU リファレンス意味論（NumPy 解釈器）。

## なぜ要るか

`codegen` は「その命令がその arch に存在し、意図どおり符号化され、ローダの形をしている」
までを第三者のツールで裏づけた。しかし **IR が何を意味するか**——とりわけ FX から降下
してきた IR がユーザーのモデルと同じ計算を表すか——はどこでも検証していなかった。
アセンブラは意味を知らないし、LLVM に問うたのは命令選択だけである。

ここは IR に**実行可能な意味**を与える。これにより
「PyTorch のモデル出力」対「降下した IR を評価した出力」という**意味論の照合**が
CPU だけで成立する（`tests/correctness/test_fxlower.py`）。降下の分解
（softmax/layer_norm/gelu…）が正しいかは、これでしか確かめられない。

## 何を保証しないか

これは *IR の意味* であって *生成した機械語の意味* ではない。レーン割当・行列コアの
レイアウト・実機の丸めはここに現れない（それは L3・要実機）。
すなわち本モジュールが一致を示しても「GPU で同じ数値が出る」ことにはならない。
"""
from __future__ import annotations

import numpy as np

from . import ir


def _reduce(x: np.ndarray, axis: int, kind: str) -> np.ndarray:
    """keepdims 縮約。IR の reduce は後続の elementwise とブロードキャストで結合する。"""
    if kind == "max":
        return np.max(x, axis=axis, keepdims=True)
    if kind == "mean":
        return np.mean(x, axis=axis, keepdims=True)
    return np.sum(x, axis=axis, keepdims=True)


def evaluate(module: ir.Module, inputs: list[np.ndarray]) -> list[np.ndarray]:
    """IR を NumPy で評価する。`load` は inputs を順に消費する。

    `store` された値を出力として返す（複数なら順に）。`load` が inputs を使い切ったら
    最後の入力を再利用する（重みを持たない単入力グラフの評価を想定）。
    **入力の束縛が曖昧な IR（重みが graph 外にある matmul 等）には使わないこと**——
    それは意味論の検証にならない。
    """
    outs: list[np.ndarray] = []
    for kernel in module.kernels:
        env: dict[str, np.ndarray] = {}
        n_load = 0

        def val(v):
            return env[v.name]

        for op in kernel.body:
            k, a = op.kind, op.attrs
            if k == "load":
                x = inputs[min(n_load, len(inputs) - 1)]
                n_load += 1
                r = np.asarray(x, dtype=np.float64)
            elif k == "zeros":
                base = next(iter(env.values()), None)
                shape = base.shape if base is not None else (1,)
                r = np.full(shape, float(a.get("fill", 0.0)), dtype=np.float64)
            elif k in ("add", "sub", "mul", "div", "max"):
                x, y = val(op.operands[0]), val(op.operands[1])
                r = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
                     "div": np.divide, "max": np.maximum}[k](x, y)
            elif k == "exp":
                r = np.exp(val(op.operands[0]))
            elif k == "sqrt":
                r = np.sqrt(val(op.operands[0]))
            elif k == "rsqrt":
                r = 1.0 / np.sqrt(val(op.operands[0]))
            elif k == "reduce":
                r = _reduce(val(op.operands[0]), int(a.get("axis", -1)),
                            str(a.get("kind", "sum")))
            elif k == "cast":
                r = val(op.operands[0]).astype(np.float16).astype(np.float64)
            elif k == "dot":
                x, y = val(op.operands[0]), val(op.operands[1])
                acc = val(op.operands[2]) if len(op.operands) > 2 else 0.0
                r = x @ y + acc
            elif k == "store":
                outs.append(val(op.operands[0]))
                continue
            else:
                raise NotImplementedError(f"interp: unsupported IR op {k!r}")
            if op.result is not None:
                env[op.result.name] = r
    return outs
