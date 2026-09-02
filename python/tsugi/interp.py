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


def evaluate(module: ir.Module, inputs, *, dot=None) -> list[np.ndarray]:
    """IR を NumPy で評価する。

    `dot` を渡すと `dot` op の意味論を差し替えられる（`dot(x, y) -> ndarray`）。
    これで **「ベンダー A の matmul」「ベンダー B の matmul」** を同じ IR に当てて
    走らせられる——`equivalence.simulate_vendor_matmul` を渡せば、静的な天井でなく
    *実測* のクロスベンダー発散が CPU で得られる（第 62 回）。既定は正確な `@`。

    `inputs` が dict なら `load` の `binding` 属性（`"input:x"` / `"param:weight"`）で
    束縛する——生成カーネルの引数が何のテンソルかを IR 自身が持っているので、
    重みを含むグラフでも意味論を照合できる。list なら出現順に消費する。

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
                desc = a.get("binding")
                if isinstance(inputs, dict):
                    if desc not in inputs:
                        raise KeyError(
                            f"interp: 束縛 {desc!r} が inputs に無い "
                            f"（利用可能: {sorted(inputs)}）")
                    x = inputs[desc]
                else:
                    x = inputs[min(n_load, len(inputs) - 1)]
                n_load += 1
                r = np.asarray(x, dtype=np.float64)
            elif k == "zeros":
                # スカラーで返しブロードキャストに任せる。IR の zeros は「定数」と
                # 「dot のアキュムレータ」にしか使われず、どちらも被演算子の形に従う。
                # 形を推測すると（例: 直前の値の shape）dot の結果形と食い違う。
                r = np.float64(a.get("fill", 0.0))
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
                if a.get("rhs_transposed"):
                    y = y.T          # linear(x,W,b) = x·Wᵀ + b
                acc = val(op.operands[2]) if len(op.operands) > 2 else 0.0
                r = (dot(x, y) if dot is not None else x @ y) + acc
            elif k == "store":
                outs.append(val(op.operands[0]))
                continue
            else:
                raise NotImplementedError(f"interp: unsupported IR op {k!r}")
            if op.result is not None:
                env[op.result.name] = r
    return outs
