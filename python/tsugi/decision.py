"""Tsugi decision — タスクレベル等価性（判断は数値でなく決定で測る）。

盲点: 全 7 視点は「数値がどれだけ違うか」(abs/rel/bias/noise) を測ってきた。だが
開発者が出荷するのは *タスクの判断* —— 分類の argmax、LM の選択トークン、検出のしきい値。
数値発散は判断の代理にすぎず、両者は decouple している:

  - **スケール不変**: logit を 10 倍すれば abs 誤差も 10 倍だが argmax は不変
    → 判断フリップ率は同一（max_abs は判断を測れていない）。
  - **数値等価 ⇏ タスク等価**: 巨大な数値発散でもマージンが大きければフリップ 0、
    微小な発散でも near-tie ならフリップする。

判断が変わるか否かは局所 **マージン（top1−top2）** が支配する。発散 δ が判断を覆すには
マージン < 2δ が *必要*（十分ではない）。ゆえにフリップ率 ≤ P(margin < 2δ) という
保守的な上界が成り立ち、タスク相対の許容は固定 atol でなく **マージン分布** である。

本当に問うべきは「数値がどれだけ違うか」でなく「何 % の予測がベンダー間で変わるか」。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .report import FindingReport, Risk


def margin(logits: np.ndarray) -> np.ndarray:
    """各サンプルの判断マージン = top1 − top2（最後の軸をクラス軸とみなす）。"""
    x = np.asarray(logits, dtype=np.float64)
    part = np.partition(x, -2, axis=-1)
    return part[..., -1] - part[..., -2]


def decision_flips(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ベンダー間で argmax（判断）が変わったサンプルの真偽配列。"""
    return np.argmax(a, axis=-1) != np.argmax(b, axis=-1)


def flip_rate(a: np.ndarray, b: np.ndarray) -> float:
    """判断フリップ率（ユーザーに見える差・スケール不変）。"""
    f = decision_flips(a, b)
    return float(np.mean(f)) if f.size else 0.0


def topk_flip_rate(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    """top-k 候補 *集合* がベンダー間で変わるサンプル率（生成タスク向け）。

    LLM は top-k/top-p から次トークンを選ぶので、argmax だけでなく候補集合の一致が重要。
    集合の比較（順序は問わない）。k=1 は argmax フリップ率に一致する。
    候補がどれか境界（rank k と k+1）を跨ぐと flip = より大きな摂動を要し、生成の
    実効的な「選択肢の安定性」を測る。
    """
    af = np.asarray(a)
    bf = np.asarray(b)
    n, c = af.shape[0], af.shape[-1]
    kk = min(k, c)
    ta = np.argpartition(af, -kk, axis=-1)[:, -kk:]
    tb = np.argpartition(bf, -kk, axis=-1)[:, -kk:]
    ta.sort(axis=-1)
    tb.sort(axis=-1)
    return float(np.mean(np.any(ta != tb, axis=-1))) if n else 0.0


def _nucleus_mask(logits: np.ndarray, p: float, temperature: float) -> np.ndarray:
    """top-p（nucleus）集合のメンバシップ真偽（vocab 軸）。softmax(logit/temp) 降順で
    累積確率が p に達するまでの最小集合（境界トークンを含む）。"""
    x = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-9)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    pr = e / e.sum(axis=-1, keepdims=True)
    order = np.argsort(-pr, axis=-1)
    sorted_pr = np.take_along_axis(pr, order, axis=-1)
    csum = np.cumsum(sorted_pr, axis=-1)
    keep_sorted = (csum - sorted_pr) < p   # この位置より前の累積 < p なら含む（境界含む）
    mask = np.zeros_like(keep_sorted)
    np.put_along_axis(mask, order, keep_sorted, axis=-1)
    return mask


def nucleus_flip_rate(a: np.ndarray, b: np.ndarray, p: float = 0.9,
                      temperature: float = 1.0) -> float:
    """top-p（nucleus）候補 *集合* がベンダー間で変わるサンプル率（生成タスク向け）。

    最新 LLM は nucleus サンプリングが主流。集合サイズは可変で *確率依存* なので、
    argmax/top-k 集合と違い **スケール不変でない**（logit スケール=温度で nucleus が伸縮）。
    これは温度設定がベンダー間一致に効くことを意味する（honest な区別）。
    """
    if np.asarray(a).shape[0] == 0:
        return 0.0
    ma = _nucleus_mask(a, p, temperature)
    mb = _nucleus_mask(b, p, temperature)
    return float(np.mean(np.any(ma != mb, axis=-1)))


def divergence_rms(a: np.ndarray, b: np.ndarray) -> float:
    """ベンダー間 logit 差の典型値 δ（RMS）。"""
    af = np.asarray(a, dtype=np.float64)
    bf = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((af - bf) ** 2)))


def residual_divergence_rms(a: np.ndarray, b: np.ndarray) -> float:
    """argmax を *動かす* 成分だけの δ（per-sample アフィン系統成分を除いた残差 RMS）。

    クロスベンダー発散は系統（相関）成分と乱雑成分を持つ（arXiv:2511.00025）。系統成分の
    うち per-sample の **スケール α と切片 c**（b ≈ α·a + c）は argmax を保存する（順序不変）
    ので判断を覆さない。各サンプルで α,c を最小二乗 fit して除いた残差が、実際にフリップを
    起こす成分。total δ でなくこれを使うと bound が正確（系統発散の過大評価を排す）。
    """
    af = np.asarray(a, dtype=np.float64)
    bf = np.asarray(b, dtype=np.float64)
    ac = af - af.mean(axis=-1, keepdims=True)
    bc = bf - bf.mean(axis=-1, keepdims=True)
    alpha = (ac * bc).sum(axis=-1, keepdims=True) / ((ac * ac).sum(axis=-1, keepdims=True) + 1e-30)
    resid = bc - alpha * ac
    return float(np.sqrt(np.mean(resid ** 2)))


def decompose_divergence(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """発散を「argmax 保存的な系統成分」と「フリップを起こす残差」に分解する。

    systematic_frac=1 なら数値的に大きくても判断は不変（タスク等価）。
    """
    total = divergence_rms(a, b)
    residual = residual_divergence_rms(a, b)
    return {"total": total, "residual": residual,
            "systematic_frac": 1.0 - residual / (total + 1e-30)}


def predicted_flip_bound(ref_logits: np.ndarray, delta: float) -> float:
    """発散 δ が与える判断フリップ率の保守的上界 = P(margin < 2δ)。

    数値の床（calibration）・ノイズの床（nondeterminism）を *タスク影響* に翻訳する橋。
    フリップには margin<2δ が必要ゆえ上界。実フリップ率はこれ以下に収まる。
    """
    m = margin(ref_logits)
    return float(np.mean(m < 2.0 * delta)) if m.size else 0.0


def flip_bound_from_divergence(ref_logits: np.ndarray, rel_divergence: float) -> float:
    """*相対* 発散（propagation のモデル発散）を *タスク* フリップ率上界へ翻訳する。

    propagation は相対発散 δ_rel を返す。logit に効く絶対発散は δ_abs = δ_rel·scale。
    これを predicted_flip_bound に通すことで、第2ベンダーを走らせる前に、静的な
    op グラフ＋代表的な logit 分布だけからタスク影響（判断フリップ率の上界）を予測できる。
    視点4（propagation）→ 視点8（decision）をつなぐ橋。
    """
    x = np.asarray(ref_logits, dtype=np.float64)
    scale = float(np.sqrt(np.mean(x ** 2)) + 1e-30)
    return predicted_flip_bound(ref_logits, rel_divergence * scale)


@dataclass
class DecisionReport(FindingReport):
    flip_rate: float = 0.0
    n: int = 0
    flipped_margin_median: float = 0.0
    overall_margin_median: float = 0.0
    predicted_bound: float = 0.0
    systematic_frac: float = 0.0
    topk: int = 1
    topk_flip_rate: float = 0.0

    def to_text(self) -> str:  # type: ignore[override]
        tk = (f", top-{self.topk} set flip={self.topk_flip_rate * 100:.2f}%"
              if self.topk > 1 else "")
        return super().to_text(
            header=(f"decision equivalence: flip_rate={self.flip_rate * 100:.2f}% "
                    f"(n={self.n}, bound≤{self.predicted_bound * 100:.2f}%, "
                    f"systematic={self.systematic_frac * 100:.0f}%{tk}, "
                    f"flipped-margin {self.flipped_margin_median:.3g} "
                    f"vs overall {self.overall_margin_median:.3g})"),
            empty="(no decision flips — task-equivalent)")


def compare_decisions(a: np.ndarray, b: np.ndarray, *, flip_budget: float = 0.0,
                      ref: np.ndarray | None = None, topk: int = 1) -> DecisionReport:
    """タスクレベルの等価判定（数値でなく判断のフリップで測る）。

    flip_budget: 許容する判断フリップ率（タスク予算・例 0.001 = 0.1%）。
    ref: マージン基準の logit（既定 a）。
    topk: >1 なら生成タスク向けに top-k 候補集合フリップ率も併記する。
    bound は *残差*（argmax 保存的な系統成分を除いた成分）で評価し系統発散の過大評価を排す。
    """
    flips = decision_flips(a, b)
    ref_logits = a if ref is None else ref
    m = margin(ref_logits)
    fm = m[flips]
    decomp = decompose_divergence(a, b)
    rep = DecisionReport(
        flip_rate=flip_rate(a, b),
        n=int(np.argmax(a, axis=-1).size),
        flipped_margin_median=float(np.median(fm)) if fm.size else 0.0,
        overall_margin_median=float(np.median(m)) if m.size else 0.0,
        predicted_bound=predicted_flip_bound(ref_logits, decomp["residual"]),
        systematic_frac=decomp["systematic_frac"],
        topk=topk,
        topk_flip_rate=topk_flip_rate(a, b, topk) if topk > 1 else flip_rate(a, b),
    )
    if rep.flip_rate > flip_budget:
        risk = Risk.BLOCK if rep.flip_rate > max(10 * flip_budget, 0.01) else Risk.WARN
        rep.add(risk, "task",
                f"判断フリップ率 {rep.flip_rate * 100:.2f}% > 予算 {flip_budget * 100:.2f}% "
                "→ ベンダー間でユーザーに見える予測が変わる")
    elif rep.flip_rate > 0.0:
        rep.add(Risk.INFO, "task",
                f"判断フリップ {rep.flip_rate * 100:.2f}%（予算内）・near-tie に集中")
    # 健全性: フリップは低マージン(near-tie)の裾に集中するはず。確信領域で起きるなら異常。
    if fm.size and rep.overall_margin_median > 0 and \
            rep.flipped_margin_median > 0.5 * rep.overall_margin_median:
        rep.add(Risk.WARN, "task",
                "フリップが near-tie 裾に集中していない → 確信予測まで変化・系統的発散を疑う")
    # 数値的に大きくても argmax 保存的な系統発散ならタスクは等価（calibration の系統検出と対）
    if rep.systematic_frac > 0.9 and rep.flip_rate == 0.0:
        rep.add(Risk.INFO, "task",
                f"発散の {rep.systematic_frac * 100:.0f}% は argmax 保存的な系統成分（スケール/シフト）"
                "→ 数値的に大きくても判断は不変（タスク等価）")
    return rep
