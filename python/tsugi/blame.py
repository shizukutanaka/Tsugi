"""Tsugi blame — ベンダー責帰（新視点13）。

ソクラテス式問答:

  Q1. cross-vendor BLOCK が出た。どちらのベンダーの実装が間違っているか？
      equivalence/attribution は「A vs B の差」を測る。しかし「どちらが真値に近いか」は
      現在の情報では分からない。開発者は両実装をデバッグするしかなく O(2L) の作業になる。

  Q2. oracle（CPU float64 参照）との距離を比較すれば何が分かるか？
      dist_a = max|a − oracle| / (max|oracle| + ε)（相対距離）。
      dist_a < dist_b → B が oracle から遠い → B（AMD）の実装を優先修正。
      dist_b < dist_a → A が oracle から遠い → A（NVIDIA）の実装を優先修正。
      dist_a ≈ dist_b → 両方同程度に乖離（oracle 自身を疑う、または両実装を見直す）。

  Q3. oracle_check（視点メタ）と何が違うか？
      - oracle_check: "A ≈ B ≈ oracle か"。A ≈ B でも両方 oracle と不一致（shared mode）を検出。
      - blame: "A と B の *相対* 正確性" を比較し「どちらを修正するか」の方向を提供。
      相補的: oracle_check は「共有モード障害の有無」、blame は「責帰の割り当て」。

  Q4. attribution（視点12）と組み合わせると何が完成するか？
      attribution.spike = "layer7_attn"（どの層か）
      blame.closer = "B"（どちらが oracle に近いか＝どちらを直すか）
      → 完全な診断: "layer 7 attention の vendor B 実装が oracle から遠い → B を直せ"
      「出力差を検出 → どの層か（attribution）→ どちらのベンダーか（blame）」の
      診断チェーンが閉じる。

  Q5. ratio（max/min 距離比）が示す情報は？
      ratio が大きい → 責任が一方に集中（方向が明確）。
      ratio ≈ 1 → 両方同程度に間違っている（前提の見直し推奨）。
      oracle_check.verify_oracle を先に呼んで oracle 自身の健全性を担保してから使うこと推奨。

実装の要点:
  - float64 演算を oracle として仮定（oracle_check.verify_oracle で事前確認推奨）。
  - relative=True: scale 不変（絶対値の大きなテンソルで誤判定しない）。
  - blame は *観測的比較* であって因果特定ではない。真因（アルゴリズム誤り / 累積誤差 /
    HW 仕様差）は attribution と組み合わせて推論する。
  - `layer_blame` が per-layer で (dist_a, dist_b) を返し attribution.spike の層で
    blame を確認するクロスチェックを提供する。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .report import FindingReport, Risk

# tol の何倍を超えたら WARN → BLOCK に格上げするか。
# 論拠: tolerance.derive_tolerance がすでに safety·√K·u の安全マージンを含む（保守的）。
# そのさらに 10× 超過は「安全マージンを考慮してもなお大幅に超えている = 系統的な誤り」を意味し、
# WARN（要注意）でなく BLOCK（修正必須）と判断する。calibration の検出限界スケールと整合。
_BLOCK_DIST_RATIO: float = 10.0


def accuracy_relative(out, oracle, *, eps: float = 1e-30) -> float:
    """output の oracle に対する相対距離。max|out − oracle| / (max|oracle| + ε)。"""
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(oracle, dtype=np.float64)
    if not out.size:
        return 0.0
    num = float(np.max(np.abs(out - ref)))
    denom = float(np.max(np.abs(ref))) + eps
    return num / denom


@dataclass
class BlameReport(FindingReport):
    """どちらのベンダーが oracle に近いか（責帰）を報告する。"""

    dist_a: float = 0.0        # A の oracle 相対距離
    dist_b: float = 0.0        # B の oracle 相対距離
    closer: str = "TIED"       # "A" / "B" / "TIED"（ratio < ratio_threshold）
    ratio: float = 1.0         # max(dist_a, dist_b) / (min(dist_a, dist_b) + ε)
    tol: float = 0.0
    ratio_threshold: float = 2.0

    def to_text(self) -> str:  # type: ignore[override]
        return super().to_text(
            header=(f"blame (dist_a={self.dist_a:.2e}, dist_b={self.dist_b:.2e}, "
                    f"closer={self.closer}, ratio={self.ratio:.1f}, tol={self.tol:.2e})"),
            empty="(both vendors within tolerance — blame inconclusive)")


def compare_accuracy(a, b, oracle, *, tol: float, ratio_threshold: float = 2.0,
                     relative: bool = True) -> BlameReport:
    """A と B それぞれの oracle 距離を比較し、どちらが oracle に近いか（責帰）を報告。

    closer="A": A が oracle に近い → vendor B の実装を優先修正。
    closer="B": B が oracle に近い → vendor A の実装を優先修正。
    closer="TIED": 距離比が ratio_threshold 未満 → 差が小さく方向不明。

    oracle_check.verify_oracle で oracle の健全性を事前確認してから呼ぶこと推奨。
    attribution.spike（どの層）と組み合わせて "どの層の、どちらのベンダーの実装か" を特定。
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    ref = np.asarray(oracle, dtype=np.float64)

    if relative:
        dist_a = accuracy_relative(a_arr, ref)
        dist_b = accuracy_relative(b_arr, ref)
    else:
        dist_a = float(np.max(np.abs(a_arr - ref))) if a_arr.size else 0.0
        dist_b = float(np.max(np.abs(b_arr - ref))) if b_arr.size else 0.0

    eps = 1e-30
    ratio = max(dist_a, dist_b) / (min(dist_a, dist_b) + eps)

    if ratio < ratio_threshold:
        closer = "TIED"
    elif dist_a < dist_b:
        closer = "A"
    else:
        closer = "B"

    rep = BlameReport(
        dist_a=dist_a, dist_b=dist_b, closer=closer,
        ratio=ratio, tol=tol, ratio_threshold=ratio_threshold,
    )

    if not a_arr.size:
        rep.add(Risk.INFO, "blame", "空テンソル — 比較対象なし")
        return rep

    a_ok = dist_a <= tol
    b_ok = dist_b <= tol

    if a_ok and b_ok:
        rep.add(Risk.OK, "blame",
                f"両ベンダーとも oracle 距離 ≤ tol ({tol:.2e}) — 責帰不要")
        return rep

    if a_ok and not b_ok:
        risk = Risk.BLOCK if dist_b > tol * _BLOCK_DIST_RATIO else Risk.WARN
        rep.add(risk, "blame",
                f"A は oracle 内 (dist={dist_a:.2e} ≤ tol) / "
                f"B は超過 (dist={dist_b:.2e}) → vendor B の実装を優先修正")
        return rep

    if b_ok and not a_ok:
        risk = Risk.BLOCK if dist_a > tol * _BLOCK_DIST_RATIO else Risk.WARN
        rep.add(risk, "blame",
                f"B は oracle 内 (dist={dist_b:.2e} ≤ tol) / "
                f"A は超過 (dist={dist_a:.2e}) → vendor A の実装を優先修正")
        return rep

    # 両方 tol 超
    if closer == "TIED":
        risk = Risk.WARN
        rep.add(risk, "blame",
                f"A({dist_a:.2e}) と B({dist_b:.2e}) が同程度に oracle から乖離 "
                f"(ratio={ratio:.1f} < {ratio_threshold}) — 両実装を見直す "
                "（oracle_check.verify_oracle で oracle の健全性を確認推奨）")
    else:
        blame_side = "B" if closer == "A" else "A"
        farther = dist_b if closer == "A" else dist_a
        nearer = dist_a if closer == "A" else dist_b
        risk = Risk.BLOCK if farther > tol * _BLOCK_DIST_RATIO else Risk.WARN
        rep.add(risk, "blame",
                f"closer=vendor {closer} (dist={nearer:.2e}) / "
                f"farther=vendor {blame_side} (dist={farther:.2e}, ratio={ratio:.1f}x) "
                f"→ vendor {blame_side} の実装を優先修正")

    return rep


def layer_blame(layers_a, layers_b, layers_oracle, x, *,
                relative: bool = True) -> list[tuple[float, float]]:
    """各層で (dist_a_i, dist_b_i) を返す（attribution.spike の層で blame を確認するクロスチェック）。

    layers_oracle: oracle（CPU float64 参照）の各層 callable。通常は両ベンダーの理想実装。
    返り値: [(dist_a_0, dist_b_0), ..., (dist_a_{L-1}, dist_b_{L-1})]

    usage:
        dists = layer_blame(layers_a, layers_b, layers_oracle, x)
        spike = find_spike([d[0] + d[1] for d in dists])  # 合計が最大の層
        print(f"layer {spike}: A={dists[spike][0]:.2e}, B={dists[spike][1]:.2e}")
    """
    xa = np.asarray(x, dtype=np.float64)
    xb = xa.copy()
    xref = xa.copy()
    result: list[tuple[float, float]] = []

    for la, lb, lref in zip(layers_a, layers_b, layers_oracle):
        xa = np.asarray(la(xa), dtype=np.float64)
        xb = np.asarray(lb(xb), dtype=np.float64)
        xref = np.asarray(lref(xref), dtype=np.float64)
        if relative:
            da = accuracy_relative(xa, xref)
            db = accuracy_relative(xb, xref)
        else:
            da = float(np.max(np.abs(xa - xref))) if xa.size else 0.0
            db = float(np.max(np.abs(xb - xref))) if xb.size else 0.0
        result.append((da, db))

    return result
