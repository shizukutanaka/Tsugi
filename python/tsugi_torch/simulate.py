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

- `order`  : 累積順序の差（f32 累積・split_k 違い・fp16 格納）——最も一般的・最小
- `f16acc` : 片方が f16 で累積（fp16 格納）
- `tf32`   : 片方が TF32 入力（**f32 格納でしか発現しない**）
- `rtz`    : 同じ TF32 精度ポリシーの下で RNE 対 RTZ（**f32 格納**・系統バイアス）

クラスごとに格納 dtype を固定するのは、そうしないと *発現しないクラスを 0 と報告する*
から。当初 `tf32`/`rtz` を fp16 格納で回して両方とも「相対発散 0.00e+00」と出していたが、
これは「差が無い」ではなく「その差が表現できない」だった（fp16 の仮数 10 bit は TF32 と
同じ・`input_precision="ieee"` は丸めモード指定ごと捨てる）。正しい格納で測ると
`rtz` が **最大の発散クラス**（9.3e-4 > f16acc 4.6e-4）である。**偽OK を直す道具の中に
偽OK があった。**

## 何を保証しないか（黙らない）

模倣は **既知の発散クラスだけ** を含む。実機固有の要因（超越関数の実装差・FMA
contraction の有無・レイアウト依存の縮約順）は含まないので、実測値は実機発散の
**下界**である。実機照合（`audit_cross_vendor`）で更新することを前提にする。

## 「天井」は何の天井か（追補4・自分の見出し数値への問答）

上表の 200×/1700× は、実測を **スケール正規化** `max|Δ|/max|a|` で測ったときの比で
ある。ところが `equivalence.compare` が使う正準の相対誤差は **要素ごと**
`max(|Δ|/(|a|+1e-12))` で、こちらで測ると同じ差が 2〜7 と出る——**静的値 3.6e-2 を
上回る**。つまり静的値は「あらゆる意味での上界」ではない:

| クラス | スケール正規化 | 要素ごと | 静的値 3.6e-2 との関係 |
|---|---|---|---|
| order  | 2.2e-5 | 1.8e-3 | 下回る |
| f16acc | 4.6e-4 | 2.6e0  | **上回る** |
| tf32   | 4.1e-4 | 5.7e0  | **上回る** |
| rtz    | 9.3e-4 | 7.0e0  | **上回る** |

要素ごとの値は分母が 0 に近い要素（GELU/LayerNorm の出力）に支配されるので、
伝播モデル（op ごとの相対条件数を掛け合わせる *典型スケール* の量）と比べる相手として
適切なのはスケール正規化の方である。しかし **「天井」と一言で呼ぶと、要素ごとの
意味でも上界だと読まれる**。よって両方を報告し、どちらと比べているかを明記する。
**見出しの数値それ自体にも同じ問いを向けること。**
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: 発散クラス → (ベンダー A の設定, ベンダー B の設定, 格納 dtype)
#:
#: **格納 dtype をクラスごとに固定するのは、そうしないと *発現しないクラスを 0 と
#: 報告する* から**（第 62 回・自分で書いた模倣に対するソクラテス問答で発覚）:
#:
#: - `tf32` は入力を 10 仮数へ丸めるが、fp16 の仮数はもともと 10 bit。fp16 格納では
#:   丸めが恒等写像になり、常に「相対発散 0.00e+00」と出る。
#: - `rtz` も `input_precision="ieee"` のままだと `truncate_to_tensorcore` が
#:   仮数 23 bit で即 return し、丸めモードの指定ごと捨てられる。やはり常に 0。
#:
#: どちらも「測ったが差が無かった」ではなく「**その差がこの設定では表現できない**」。
#: 0 と表示すれば読み手は「TF32 起因の発散は無い」と読む——偽OK である。よって
#: TF32 系は f32 格納に固定し、`rtz` は *同じ精度ポリシーの下で丸めモードだけ*
#: 変える（RNE 対 RTZ）ことで、丸めモード差を単独で分離する。
DIVERGENCE_CLASSES: dict[str, tuple[dict, dict, str]] = {
    "order":  ({"accum": "f32", "split_k": 1},
               {"accum": "f32", "split_k": 8}, "float16"),
    "f16acc": ({"accum": "f32", "split_k": 1},
               {"accum": "f16", "split_k": 1}, "float16"),
    "tf32":   ({"accum": "f32", "split_k": 1},
               {"accum": "f32", "split_k": 1, "input_precision": "tf32"}, "float32"),
    "rtz":    ({"accum": "f32", "split_k": 1, "input_precision": "tf32",
                "input_rounding": "rne"},
               {"accum": "f32", "split_k": 1, "input_precision": "tf32",
                "input_rounding": "rtz"}, "float32"),
}

#: クラスが *表現可能* であるための前提（満たさないなら測らず「非適用」と言う）。
CLASS_REQUIRES: dict[str, str] = {
    "tf32": "f32 格納（fp16 の仮数 10 bit は TF32 と同じで丸めが恒等になる）",
    "rtz": "f32 格納＋縮小仮数の精度ポリシー（ieee のままだと丸めモードが捨てられる）",
}


@dataclass
class ClassResult:
    name: str
    rel_divergence: float        # max|a−b| / max|a|（**スケール正規化**・堅牢）
    flip_rate: float             # argmax が変わったサンプル率（点推定）
    flip_rate_ub: float          # Wilson 上側限界（小標本で 0 を過信しない）
    n: int
    #: `equivalence.compare` と同じ **要素ごと** の相対誤差の最大値
    #: max(|a−b| / (|a|+1e-12))。分母が 0 に近い要素（GELU/LayerNorm の出力など）で
    #: 発散するため、スケール正規化より桁で大きくなる。どちらか一方だけを
    #: 「相対発散」と呼ぶと、比較相手を取り違える（第 62 回・追補4）。
    max_rel_elementwise: float = 0.0
    storage: str = "float16"     # このクラスを測った格納 dtype
    applicable: bool = True      # False なら「差が無い」でなく「表現できない」
    why: str = ""                # 非適用の理由（黙って 0 を出さない）
    task: str = "classification"  # フリップをどのタスク意味論で測ったか


@dataclass
class SimulationReport:
    classes: list[ClassResult] = field(default_factory=list)
    n_samples: int = 0
    task: str = "classification"

    @property
    def measured(self) -> list[ClassResult]:
        """実際に発現しうるクラスだけ（非適用を混ぜると 0 が最悪値を薄める）。"""
        return [c for c in self.classes if c.applicable]

    @property
    def worst(self) -> ClassResult | None:
        # 上界が並ぶ（小標本ではフリップ 0 が揃い、上界が同値になる）ので相対発散で
        # 決着させる。ここを怠ると「最悪クラス」が登録順の先頭になり、実際に一番
        # 大きいクラスの名前がレポートに出ない。
        return max(self.measured,
                   key=lambda c: (c.flip_rate_ub, c.rel_divergence), default=None)

    @property
    def typical(self) -> ClassResult | None:
        return next((c for c in self.measured if c.name == "order"), None)

    def to_lines(self) -> list[str]:
        out = [f"CPU 2 ベンダー模倣（n={self.n_samples}・既知の発散クラス・"
               f"フリップは task={self.task} の意味論）:"]
        for c in self.classes:
            if not c.applicable:
                out.append(f"  {c.name:7s}: **非適用** — {c.why}"
                           "（0 と報告すると「発散なし」と読まれる）")
                continue
            out.append(f"  {c.name:7s}[{c.storage[-2:]}]: 相対発散 {c.rel_divergence:.2e}"
                       f"（要素ごと {c.max_rel_elementwise:.2e}）  "
                       f"フリップ率 {c.flip_rate * 100:.3f}%（上界 {c.flip_rate_ub * 100:.3f}%）")
        out.append("  ※ 模倣は既知クラスのみ。実機固有の要因（超越関数実装・FMA 融合・"
                   "レイアウト依存の縮約順）は含まないので、これは実機発散の *下界*。")
        out.append("  ※ 「相対発散」はスケール正規化 max|Δ|/max|a|。括弧内は "
                   "`equivalence.compare` と同じ **要素ごと** の最大相対誤差で、"
                   "分母が 0 に近い要素（GELU/LayerNorm 出力）で桁が上がる。"
                   "静的伝播の値と比べてよいのは前者のみ（後者は上回りうる）。")
        if self.task == "classification":
            out.append("  ※ フリップは **argmax（多クラス分類）前提**。回帰・バイナリ・"
                       "ランキング・サンプリングのモデルでは argmax が固定され "
                       "flip=0 に張りつく（静かな誤用）——`task=` を指定すること。")
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


def simulate_cross_vendor(gm: Any, x, *,
                          classes: tuple[str, ...] = ("order", "f16acc", "tf32", "rtz"),
                          storage: tuple[str, ...] = ("float16", "float32"),
                          task: str = "classification",
                          task_kwargs: dict | None = None,
                          confidence: float = 0.95) -> SimulationReport | None:
    """FX グラフを CPU で「2 ベンダー」として走らせ、発散クラスごとに実測する。

    `x` は代表入力 1 本（グラフの引数が活性 1 本だけのとき）か、**グラフの引数と同順・
    同数の列**（`torch.compile` の dynamo グラフのように重みが引数へ持ち上げられている
    とき）。降下が partial（表せない op がある）か、束縛が一意に決まらないときは
    **None**（模倣できないことを黙って 0 で埋めない）。理由は `refusal_reason`。
    """
    from tsugi.arrays import asarray
    from tsugi.decision import compare_task, decision_flips
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
    def vendor(cfg: dict, store: str):
        lo = np.float16 if store == "float16" else np.float32

        def _dot(a, b):
            return simulate_vendor_matmul(np.asarray(a).astype(lo),
                                          np.asarray(b).astype(lo),
                                          **cfg).astype(np.float64)
        return _dot

    rep = SimulationReport(task=task)
    for name in classes:
        cfg_a, cfg_b, store = DIVERGENCE_CLASSES[name]
        if store not in storage:
            # そのクラスが発現する格納 dtype を測っていない。0 でなく **非適用**と言う。
            rep.classes.append(ClassResult(
                name, 0.0, 0.0, 0.0, 0, storage=store, applicable=False, task=task,
                why=f"{store} 格納を測っていない: " + CLASS_REQUIRES.get(name, "前提未充足")))
            continue
        out_a = evaluate(lm.module, binds, dot=vendor(cfg_a, store))[-1]
        # 標本数は **出力の行数**。入力側から数えると、dynamo のように重みが引数へ
        # 持ち上げられたグラフで重みの行数を標本数と誤って報告する。
        rep.n_samples = int(out_a.shape[0]) if out_a.ndim else 1
        out_b = evaluate(lm.module, binds, dot=vendor(cfg_b, store))[-1]
        denom = float(np.max(np.abs(out_a))) + 1e-30
        rel = float(np.max(np.abs(out_a - out_b))) / denom
        # 同じ差を **2 つの尺度**で測る。どちらか一方だけを「相対発散」と呼ぶと、
        # 比較相手（静的伝播の値）と別の量を突き合わせることになる（追補4）。
        _e = np.abs(out_a - out_b) / (np.abs(out_a) + 1e-12)
        rel_elem = float(np.nanmax(_e)) if _e.size else 0.0
        if task != "classification":
            # 非分類タスクに argmax を当てると flip=0 に張りつく（新視点11 の静かな
            # 誤用）。decision 層は既にタスク別の意味論を持っているので、模倣も
            # そちらへ委ねる——同じ罠を新しい道具で踏み直さない。
            tr = compare_task(out_a, out_b, task=task, confidence=confidence,
                              **(task_kwargs or {}))
            n, fr, ub = int(tr.n), float(tr.flip_rate), float(tr.flip_rate_ub)
        elif out_a.ndim >= 2 and out_a.shape[-1] > 1:
            flips = decision_flips(out_a, out_b)
            n = int(flips.size)
            k = int(flips.sum())
            fr = k / n if n else 0.0
            ub = flip_rate_upper_bound(k, n, confidence=confidence) if n else 0.0
        else:
            n, fr, ub = 0, 0.0, 0.0
        # 2 設定が **ビット同一** を返したら、それは「差が無かった」ではなく
        # 「この設定では差が表現できない」可能性が高い（第 62 回に tf32/rtz で実際に
        # 起きた）。0 と報告すると偽OKなので、非適用として名指しする。
        if rel == 0.0 and name in CLASS_REQUIRES:
            rep.classes.append(ClassResult(
                name, 0.0, 0.0, 0.0, n, storage=store, applicable=False, task=task,
                why=f"{store} 格納でも差がビット同一 — 前提: "
                    + CLASS_REQUIRES[name]))
            continue
        rep.classes.append(ClassResult(name, rel, fr, ub, n, storage=store, task=task,
                                       max_rel_elementwise=rel_elem))
    return rep
