"""Tsugi attribution — 発散帰属（新視点12）。

ソクラテス式問答:

  Q1. equivalence/decision/rollout はすべて *出力層* の最終発散を測る。
      「移植が壊れた」とき、開発者は次に何をするか？ → 手動でデバッグする。
      どの層/op で発散が起きたか、中間テンソルを順番に印刷しながら探す。
      これは O(L) の作業で深いモデルでは数時間かかる。

  Q2. propagation（新視点4）は「どの op が dominant amplifier か」を *理論的に* 予測する。
      だが実測データでそれを確認する手段はあるか？ → 無い。propagation は上界推定であって、
      実データでの因果分析ではない。理論モデルが正しいか、実際にどこで発散が爆発するかは
      出力だけを見ても分からない。

  Q3. ならば中間テンソルを各層で比較すれば何が分かるか？ →
      - **onset（発散の始まり）**: threshold を超える最初の層 = ここより後は汚染されている。
      - **spike（最大増幅）**: 発散の増分が最大の層 = propagation の dominant amplifier を実測で照合。
      - **binary search**: 第 L 層が同じなら原因は L より後ろ。違えば前。O(log L) で絞れる。

  Q4. これは propagation の何を補うか？ → propagation は *理論的な上界*（保守）。attribution は
      *実データでの因果特定*（観測）。propagation.dominant() が「softmax が dominant」と言ったとき、
      attribution.spike が layer=7 の softmax を示せば理論が確かめられる。示さなければ理論のモデル
      誤差が可視化される。**理論（propagation）と実験（attribution）の接続点。**

  Q5. これは什么（ごく普通の）デバッグ実践を *体系化* する。開発者が「とりあえず中間層を印刷」
      していたものを: 定量的 API（onset/spike/全層 divergence プロファイル）にし、
      audit 経路から呼べるようにする（修正の証拠にもなる）。

実装の要点:
  - `layers_a`/`layers_b` は各層を表す callable のリスト。各呼び出しは前層の出力を受け取る。
  - 入力 x は両ベンダーで共有（入力差異を排除し、計算差異だけを測る）。
  - 共有 CPU 参照で使うと「CPU 参照 vs GPU 実行」の per-layer 差異を可視化でき、
    実機 GPU 結果をデバッグする最短経路になる（未来の実機検証への橋）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .report import FindingReport, Risk


def layer_divergences(layers_a, layers_b, x, *, relative: bool = True) -> list[float]:
    """各層を通過した後の 2 ベンダー発散を測る（prefix scan）。

    `layers_a`/`layers_b`: 各層を表す callable のリスト。layer_i(prev_output) -> output。
    `x`: 初期入力（両ベンダーで共有 — 入力差を除き計算差だけを測る）。
    `relative=True`: max|a-b| / (max|a| + ε)。`False`: max 絶対差。

    返り値: len(layers_a) の発散リスト。インデックス i は第 i 層出力での発散。
    長さが合わない場合は短い方に揃える（ユーザーの誤指定を許容）。
    """
    xa = np.asarray(x, dtype=np.float64)
    xb = xa.copy()
    divs = []
    for la, lb in zip(layers_a, layers_b):
        xa = np.asarray(la(xa), dtype=np.float64)
        xb = np.asarray(lb(xb), dtype=np.float64)
        num = float(np.max(np.abs(xa - xb))) if xa.size else 0.0
        if relative:
            divs.append(num / (float(np.max(np.abs(xa))) + 1e-30) if xa.size else 0.0)
        else:
            divs.append(num)
    return divs


def find_onset(divs: list[float], threshold: float) -> int | None:
    """発散が threshold を超える最初の層インデックス（None = 全層で超えない）。

    binary search を模倣: divs が単調でなくても使えるが、単調な場合は
    bisect で O(log L) に落とせる。非単調（発散が下がることがある）場合は線形。
    """
    for i, d in enumerate(divs):
        if d > threshold:
            return i
    return None


def find_spike(divs: list[float]) -> int | None:
    """発散の *増分*（Δdiv = divs[i] − divs[i-1]）が最大の層インデックス。

    propagation の dominant amplifier（最大増幅層）を実測で同定する。
    divs が空または 1 要素の場合は None。
    """
    if len(divs) <= 1:
        return None if not divs else 0
    deltas = [divs[0]] + [divs[i] - divs[i - 1] for i in range(1, len(divs))]
    return int(np.argmax(deltas))


@dataclass
class AttributionReport(FindingReport):
    """per-layer 発散プロファイルと onset/spike の所見。"""

    layer_names: list[str] = field(default_factory=list)
    divs: list[float] = field(default_factory=list)
    onset: int | None = None        # threshold を最初に超える層（None = 全層安全）
    spike: int | None = None        # 発散増分が最大の層（dominant amplifier）
    tol: float = 0.0

    @property
    def n_layers(self) -> int:
        return len(self.divs)

    @property
    def final_divergence(self) -> float:
        return self.divs[-1] if self.divs else 0.0

    @property
    def spike_name(self) -> str:
        if self.spike is None or self.spike >= len(self.layer_names):
            return f"layer[{self.spike}]"
        return self.layer_names[self.spike]

    @property
    def onset_name(self) -> str:
        if self.onset is None:
            return "(none)"
        if self.onset >= len(self.layer_names):
            return f"layer[{self.onset}]"
        return self.layer_names[self.onset]

    def to_text(self) -> str:  # type: ignore[override]
        return super().to_text(
            header=(f"attribution ({self.n_layers} layers, tol={self.tol:.2e}, "
                    f"final_div={self.final_divergence:.2e}, "
                    f"onset={self.onset_name}, spike={self.spike_name})"),
            empty="(all layers within tolerance — divergence origin not found)")


def attribute(layers_a, layers_b, x, *, tol: float, names=None,
              relative: bool = True) -> AttributionReport:
    """per-layer 発散スキャンで発散の onset と spike を特定する。

    onset: 「ここより前は clean、ここから汚染」の境界 → 疑うべき op を絞る。
    spike: 「ここで発散が最も増幅された」層 → propagation の dominant() との照合。

    tol: 単一カーネル等価の許容（tolの超過をonset として定義）。
    names: 層名のリスト（指定するとレポートが読みやすい）。
    """
    divs = layer_divergences(layers_a, layers_b, x, relative=relative)
    onset = find_onset(divs, tol)
    spike = find_spike(divs)
    n = len(divs)
    if names is not None:
        raw = list(names)
        _names = (raw + [f"layer[{i}]" for i in range(len(raw), n)])[:n]
    else:
        _names = [f"layer[{i}]" for i in range(n)]

    rep = AttributionReport(
        layer_names=_names, divs=divs, onset=onset, spike=spike, tol=tol,
    )

    if not divs:
        rep.add(Risk.INFO, "attribution", "レイヤーが空 — 測定対象が無い")
        return rep

    if onset is None:
        rep.add(Risk.OK, "attribution",
                f"全 {len(divs)} 層で発散 ≤ tol {tol:.2e}（移植 clean）")
    else:
        final = divs[-1]
        spike_delta = (divs[spike] - (divs[spike - 1] if spike > 0 else 0.0)) if spike is not None else 0.0
        risk = Risk.BLOCK if final > tol * 10 else Risk.WARN
        rep.add(risk, "attribution",
                f"発散 onset={rep.onset_name} (div={divs[onset]:.2e}) | "
                f"dominant spike={rep.spike_name} (Δ={spike_delta:.2e}) | "
                f"final={final:.2e} — 疑うべき実装: {rep.spike_name}")

    # propagation の理論予測との比較メモ（呼び出し側が propagation dominant と照合できる）
    if spike is not None and onset is not None and spike != onset:
        rep.add(Risk.INFO, "attribution",
                f"spike ({rep.spike_name}) と onset ({rep.onset_name}) が異なる: "
                "最初に contaminate した層と最大増幅層は別 — propagation モデルの精度確認推奨")

    return rep


@dataclass
class DiagnosisReport(FindingReport):
    """attribution + blame の統合診断レポート。

    「どの層か」（onset/spike）と「どちらのベンダーか」（spike 層での blame）を1回で返す。
    spike_closer = "A" → vendor B を直す / "B" → vendor A を直す / "TIED" → 方向不明。
    """

    layer_names: list[str] = field(default_factory=list)
    divs: list[float] = field(default_factory=list)
    onset: int | None = None
    spike: int | None = None
    tol: float = 0.0
    spike_dist_a: float = 0.0    # spike 層での A の oracle 距離
    spike_dist_b: float = 0.0    # spike 層での B の oracle 距離
    spike_closer: str = "TIED"   # "A" / "B" / "TIED"（spike 層の責帰）

    @property
    def spike_name(self) -> str:
        if self.spike is None or self.spike >= len(self.layer_names):
            return f"layer[{self.spike}]"
        return self.layer_names[self.spike]

    @property
    def onset_name(self) -> str:
        if self.onset is None:
            return "(none)"
        if self.onset >= len(self.layer_names):
            return f"layer[{self.onset}]"
        return self.layer_names[self.onset]

    def to_text(self) -> str:  # type: ignore[override]
        blamed = "B" if self.spike_closer == "A" else ("A" if self.spike_closer == "B" else "?")
        return super().to_text(
            header=(f"diagnosis ({len(self.divs)} layers, tol={self.tol:.2e}, "
                    f"onset={self.onset_name}, spike={self.spike_name}, "
                    f"spike_closer={self.spike_closer}→fix vendor {blamed})"),
            empty="(all layers within tolerance — no divergence found)")


def diagnose(layers_a, layers_b, layers_oracle, x, *, tol: float, names=None,
             relative: bool = True) -> DiagnosisReport:
    """attribution + blame を 1 回で実行する統合診断。

    「どの層か」（onset/spike）と「その層でどちらのベンダーが oracle から遠いか（責帰）」を
    同時に返す。layers_oracle が None の場合は attribution のみ（blame なし）。

    spike_closer="A" → vendor B の実装を優先修正。
    spike_closer="B" → vendor A の実装を優先修正。
    spike_closer="TIED" → 差が小さく方向不明（両実装を疑う）。

    layers_oracle: oracle（CPU float64 参照）の各層 callable。なければ blame スキップ。
    """
    from .blame import layer_blame

    divs = layer_divergences(layers_a, layers_b, x, relative=relative)
    onset = find_onset(divs, tol)
    spike = find_spike(divs)
    n = len(divs)
    if names is not None:
        raw = list(names)
        _names = (raw + [f"layer[{i}]" for i in range(len(raw), n)])[:n]
    else:
        _names = [f"layer[{i}]" for i in range(n)]

    rep = DiagnosisReport(
        layer_names=_names, divs=divs, onset=onset, spike=spike, tol=tol,
    )

    if not divs:
        rep.add(Risk.INFO, "diagnosis", "レイヤーが空 — 測定対象が無い")
        return rep

    if onset is None:
        rep.add(Risk.OK, "diagnosis",
                f"全 {len(divs)} 層で発散 ≤ tol {tol:.2e}（移植 clean）")
    else:
        final = divs[-1]
        spike_delta = (divs[spike] - (divs[spike - 1] if spike > 0 else 0.0)) if spike is not None else 0.0
        risk = Risk.BLOCK if final > tol * 10 else Risk.WARN
        rep.add(risk, "diagnosis",
                f"onset={rep.onset_name} (div={divs[onset]:.2e}) | "
                f"spike={rep.spike_name} (Δ={spike_delta:.2e}) | "
                f"final={final:.2e}")

    # blame: spike 層での per-layer oracle 距離を比較（layers_oracle が必要）
    if layers_oracle is not None and spike is not None:
        blame_dists = layer_blame(layers_a, layers_b, layers_oracle, x, relative=relative)
        if spike < len(blame_dists):
            da, db = blame_dists[spike]
            rep.spike_dist_a = da
            rep.spike_dist_b = db
            eps = 1e-30
            ratio = max(da, db) / (min(da, db) + eps)
            if ratio < 2.0:
                rep.spike_closer = "TIED"
            elif da < db:
                rep.spike_closer = "A"
            else:
                rep.spike_closer = "B"

            blamed = "B" if rep.spike_closer == "A" else ("A" if rep.spike_closer == "B" else "?")
            rep.add(Risk.INFO, "diagnosis",
                    f"spike 層 {rep.spike_name} の責帰: vendor {rep.spike_closer} が oracle に近い "
                    f"(A={da:.2e}/B={db:.2e}) → vendor {blamed} の実装を優先修正")
    elif layers_oracle is None:
        rep.add(Risk.INFO, "diagnosis",
                "layers_oracle=None — blame スキップ（oracle なし）")

    return rep


def bisect_onset(fn_prefix_a, fn_prefix_b, x, n_layers: int, *,
                 tol: float, relative: bool = True) -> int | None:
    """onset を binary search で O(log L) で特定する（層数が多い時の効率化）。

    `fn_prefix_a(i, x)`: x を先頭 i 層だけ流した A の出力を返す callable。
    onset がない場合は None。`fn_prefix_*` は prefix i ごとに独立に呼び出す。

    layer_divergences（全層を全部流す）より呼び出し回数が少ないが、
    prefix を独立に計算できる構造（ステートレスな前向き計算）が必要。
    """
    if n_layers <= 0:
        return None

    def div_at(i: int) -> float:
        xa = np.asarray(fn_prefix_a(i, x), dtype=np.float64)
        xb = np.asarray(fn_prefix_b(i, x), dtype=np.float64)
        num = float(np.max(np.abs(xa - xb))) if xa.size else 0.0
        if relative:
            return num / (float(np.max(np.abs(xa))) + 1e-30) if xa.size else 0.0
        return num

    # 最終層で既に clean なら onset なし
    if div_at(n_layers - 1) <= tol:
        return None
    # 最初の層でもう dirty ならそこが onset
    if div_at(0) > tol:
        return 0

    lo, hi = 0, n_layers - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if div_at(mid) > tol:
            hi = mid
        else:
            lo = mid
    return hi
