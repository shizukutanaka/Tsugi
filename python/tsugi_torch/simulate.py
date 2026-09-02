"""CPU 上の 2 ベンダー模倣 — 静的な *天井* でなく *実測* の跨ベンダー発散を出す。

## なぜ要るか（第 62 回・本ラウンド最大の発見）

`audit_fx` が返す `model_divergence` / `task_flip_bound` を、同じモデルを CPU で
「2 ベンダー」として走らせた実測と突き合わせたところ:

| model | 静的 δ | 予測 flip | 実測 δ（典型） | 実測 δ（最悪クラス） | 実測 flip |
|---|---|---|---|---|---|
| MLP64 | 9.0e-2 | 35.5% | 5.2e-5 | 4.4e-4 | 0.00% |
| deep8 | 2.4e-1 | 66.3% | 3.4e-4 | 9.8e-4 | 0.00% |

静的値は **最悪クラスの実測の 200×、典型の 1700× 上**。原因は構造的で、伝播モデルは
`u(fp16)`（格納 dtype の丸め）を発散単位にしているが、両ベンダーが f32 で累積するなら
跨ベンダー差は `u(f32)` スケール（2¹³≈8192× 小さい）。**静的値は「許容の天井」であって
「予測」ではない**。天井から導いた flip 上界は真ではあっても無情報で、楔ユーザーへの
警告が毎回「フリップ率 ≤ 40〜80%」ではノイズとして無視される（偽BLOCK 100% の道具）。

## 何をするか

降下した IR（`fxlower.fx_to_ir`）を `interp.evaluate` で走らせるとき、`dot` の意味論を
`equivalence.simulate_vendor_matmul` に差し替える。既知の発散クラスごとに
「ベンダー A」「ベンダー B」を作り、出力の差と判断フリップ率を **実測** する:

- `order`  : 累積順序の差（f32 累積・split_k 違い）——最も一般的・最小
- `f16acc` : 片方が f16 で累積——既知の最悪クラス
- `tf32`   : 片方が TF32 入力（f32 モデルで効く。f16 入力では効かない）
- `rtz`    : 片方が RTZ 丸め——系統バイアス

## 何を保証しないか（黙らない）

模倣は **既知の発散クラスだけ** を含む。実機固有の要因（超越関数の実装差・FMA
contraction の有無・レイアウト依存の縮約順）は含まないので、実測値は実機発散の
**下界**、静的天井は **上界** である。両方を報告し、実機照合（`audit_cross_vendor`）で
更新することを前提にする。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: 発散クラス → (ベンダー A の matmul 設定, ベンダー B の matmul 設定)
DIVERGENCE_CLASSES: dict[str, tuple[dict, dict]] = {
    "order":  ({"accum": "f32", "split_k": 1}, {"accum": "f32", "split_k": 8}),
    "f16acc": ({"accum": "f32", "split_k": 1}, {"accum": "f16", "split_k": 1}),
    "tf32":   ({"accum": "f32", "split_k": 1},
               {"accum": "f32", "split_k": 1, "input_precision": "tf32"}),
    "rtz":    ({"accum": "f32", "split_k": 1},
               {"accum": "f32", "split_k": 1, "input_rounding": "rtz"}),
}


@dataclass
class ClassResult:
    name: str
    rel_divergence: float        # max|a−b| / max|a|
    flip_rate: float             # argmax が変わったサンプル率（点推定）
    flip_rate_ub: float          # Wilson 上側限界（小標本で 0 を過信しない）
    n: int


@dataclass
class SimulationReport:
    classes: list[ClassResult] = field(default_factory=list)
    n_samples: int = 0
    dtype: str = "float16"

    @property
    def worst(self) -> ClassResult | None:
        return max(self.classes, key=lambda c: c.flip_rate_ub, default=None)

    @property
    def typical(self) -> ClassResult | None:
        return next((c for c in self.classes if c.name == "order"), None)

    def to_lines(self) -> list[str]:
        out = [f"CPU 2 ベンダー模倣（n={self.n_samples}・入力 {self.dtype}・既知の発散クラス）:"]
        for c in self.classes:
            out.append(f"  {c.name:7s}: 相対発散 {c.rel_divergence:.2e}  "
                       f"フリップ率 {c.flip_rate * 100:.3f}%（上界 {c.flip_rate_ub * 100:.3f}%）")
        out.append("  ※ 模倣は既知クラスのみ。実機固有の要因（超越関数実装・FMA 融合・"
                   "レイアウト依存の縮約順）は含まないので、これは実機発散の *下界*。")
        return out


def _bindings_from_module(gm: Any, lm, inputs) -> tuple[dict[str, np.ndarray] | None, str]:
    """束縛記述子を実テンソルへ引く。引けなければ `(None, 理由)`。

    **同じテンソルを複数の記述子に当てない**（第 62 回の発見）。`torch.compile` が渡す
    dynamo グラフでは重みが引数へ *持ち上げられる* ため、束縛は

        ['input:L_fn_modules_0_parameters_weight_', …, 'input:L_args_0_', …]

    のように `input:` が複数になり、`named_parameters()` は **空**になる。ここで
    代表入力 1 本を全記述子へ配ると、重みの位置に活性が入った計算を「実測」と称して
    報告することになる——静的仮定を残すより悪い。よって位置対応が取れるとき
    （`inputs` が記述子と同数の列）だけ束縛し、取れなければ諦める。
    """
    params: dict[str, Any] = {}
    try:
        for name, p in gm.named_parameters():
            params[name] = p.detach().cpu().numpy()
    except Exception:  # noqa: BLE001 — stand-in グラフ等（重み無し）
        params = {}
    descs = list(lm.report.bindings)
    ins = [d for d in descs if d.startswith("input:")]
    seq = list(inputs) if isinstance(inputs, (list, tuple)) else None

    if seq is not None:
        if len(seq) != len(descs):
            return None, (f"位置対応が取れない（グラフの引数 {len(descs)} 本に対し "
                          f"与えられた入力 {len(seq)} 本）")
        return ({d: np.asarray(v, dtype=np.float64) for d, v in zip(descs, seq)}, "")

    if len(ins) != 1:
        return None, (f"活性の入力が一意でない（`input:` 記述子 {len(ins)} 本）——"
                      "dynamo は重みを引数へ持ち上げるので、代表入力 1 本では"
                      "どれが活性か決まらない。全引数を順に渡すこと")
    binds: dict[str, np.ndarray] = {}
    for d in descs:
        if d.startswith("input:"):
            binds[d] = np.asarray(inputs, dtype=np.float64)
        else:
            key = d.split(":", 1)[1]
            if key not in params:
                return None, f"重み {key!r} をモジュールから引けない"
            binds[d] = params[key]
    return binds, ""


def refusal_reason(gm: Any, inputs) -> str:
    """模倣を諦めた理由を人が読める 1 行で返す（空文字なら諦めていない）。"""
    from .fxlower import fx_to_ir
    try:
        lm = fx_to_ir(gm)
    except Exception as exc:                # noqa: BLE001
        return f"降下できない: {type(exc).__name__}"
    if lm.report.partial:
        return f"降下が partial（表せない op {sorted(set(lm.report.unsupported))}）"
    return _bindings_from_module(gm, lm, inputs)[1]


def simulate_cross_vendor(gm: Any, x, *, dtype: str = "float16",
                          classes: tuple[str, ...] = ("order", "f16acc", "tf32", "rtz"),
                          confidence: float = 0.95) -> SimulationReport | None:
    """FX グラフを CPU で「2 ベンダー」として走らせ、発散クラスごとに実測する。

    `x` は代表入力 1 本（グラフの引数が活性 1 本だけのとき）か、**グラフの引数と同順・
    同数の列**（`torch.compile` の dynamo グラフのように重みが引数へ持ち上げられている
    とき）。降下が partial（表せない op がある）か、束縛が一意に決まらないときは
    **None**（模倣できないことを黙って 0 で埋めない）。理由は `refusal_reason`。
    """
    from tsugi.arrays import asarray
    from tsugi.decision import decision_flips
    from tsugi.equivalence import simulate_vendor_matmul
    from tsugi.interp import evaluate
    from tsugi.rollout import flip_rate_upper_bound

    from .fxlower import fx_to_ir

    lm = fx_to_ir(gm)
    if lm.report.partial:
        return None
    if not isinstance(x, (list, tuple)):
        x = asarray(x, dtype=np.float64)
    binds, _why = _bindings_from_module(gm, lm, x)
    if binds is None:
        return None
    lo = np.float16 if dtype == "float16" else np.float32

    def vendor(cfg: dict):
        def _dot(a, b):
            return simulate_vendor_matmul(np.asarray(a).astype(lo),
                                          np.asarray(b).astype(lo),
                                          **cfg).astype(np.float64)
        return _dot

    rep = SimulationReport(dtype=dtype)
    for name in classes:
        cfg_a, cfg_b = DIVERGENCE_CLASSES[name]
        out_a = evaluate(lm.module, binds, dot=vendor(cfg_a))[-1]
        # 標本数は **出力の行数**。入力側から数えると、dynamo のように重みが引数へ
        # 持ち上げられたグラフで重みの行数を標本数と誤って報告する。
        rep.n_samples = int(out_a.shape[0]) if out_a.ndim else 1
        out_b = evaluate(lm.module, binds, dot=vendor(cfg_b))[-1]
        denom = float(np.max(np.abs(out_a))) + 1e-30
        rel = float(np.max(np.abs(out_a - out_b))) / denom
        if out_a.ndim >= 2 and out_a.shape[-1] > 1:
            flips = decision_flips(out_a, out_b)
            n = int(flips.size)
            k = int(flips.sum())
            fr = k / n if n else 0.0
            ub = flip_rate_upper_bound(k, n, confidence=confidence) if n else 0.0
        else:
            n, fr, ub = 0, 0.0, 0.0
        rep.classes.append(ClassResult(name, rel, fr, ub, n))
    return rep
