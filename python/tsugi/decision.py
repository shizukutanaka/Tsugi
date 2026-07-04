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

# --- 閾値定数 (Q5: magic number 排除) ---
# フリップしたサンプルの margin 中央値が全体中央値の何割を超えたら「near-tie に集中していない」
# と判断するか（0.5 = 50%）。確信サンプルまでフリップしているなら系統的発散を疑う。
_NEAR_TIE_MARGIN_FRAC: float = 0.5
# 予算の何倍を超えたら WARN → BLOCK に格上げするか（10×）。
# 予算 flip_budget がゼロ（完全無フリップ要求）の場合は _FLIP_BLOCK_MIN の絶対値で判断。
_FLIP_BLOCK_RATIO: float = 10.0
# flip_budget=0 の場合の BLOCK 最小フリップ率（1% = 実用上無視できない規模）。
_FLIP_BLOCK_MIN: float = 0.01


def margin(logits: np.ndarray) -> np.ndarray:
    """各サンプルの判断マージン = top1 − top2（最後の軸をクラス軸とみなす）。"""
    x = np.asarray(logits, dtype=np.float64)
    part = np.partition(x, -2, axis=-1)
    return part[..., -1] - part[..., -2]


def decision_flips(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ベンダー間で argmax（判断）が変わったサンプルの真偽配列。"""
    return np.argmax(a, axis=-1) != np.argmax(b, axis=-1)


def tie_rate(logits: np.ndarray, eps: float = 0.0) -> float:
    """top1 と top2 が同点（差 ≤ eps）なサンプル率。

    同点では argmax は *規約* で決まる（np は先頭 index）。2 ベンダーが異なる規約
    （first/last）を使うと、数値が完全一致でも判断がフリップする —— ハード発散でなく
    tie-break 規約の差。量子化(int8/fp8)・マスク(-inf)で多発。tie_rate が高い領域では
    flip_rate の「発散」への帰属は信頼できない（規約依存）ことを示す診断。
    """
    m = margin(logits)
    return float(np.mean(m <= eps)) if m.size else 0.0


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


def predicted_flip_bound(ref_logits: np.ndarray, delta: float,
                         confidence: float = 0.95) -> float:
    """発散 δ が与える判断フリップ率の保守的上界 = P(margin < 2δ)。

    数値の床（calibration）・ノイズの床（nondeterminism）を *タスク影響* に翻訳する橋。
    フリップには margin<2δ が必要ゆえ上界。実フリップ率はこれ以下に収まる。

    fail-safe: P(margin<2δ) は代表 logit（ref_logits）n 件からの *点推定* に過ぎない。
    n が小さい代表集合では、たまたま 0 件（または少数件）しか margin<2δ に該当せず
    真の確率を過小評価しうる（rollout.flip_rate_upper_bound と同じ「0 観測でも
    p=0 と過信しない」問題）。ここでは観測比率でなく Wilson 信頼区間の片側上限を返す。
    n が大きければ上限は点推定にほぼ収束し挙動は変わらない（回帰なし）。
    """
    from .rollout import flip_rate_upper_bound
    m = margin(ref_logits)
    if m.size == 0:
        return 0.0
    k = int(np.count_nonzero(m < 2.0 * delta))
    return flip_rate_upper_bound(k, int(m.size), confidence=confidence)


def flip_bound_from_divergence(ref_logits: np.ndarray, rel_divergence: float,
                               confidence: float = 0.95) -> float:
    """*相対* 発散（propagation のモデル発散）を *タスク* フリップ率上界へ翻訳する。

    propagation は相対発散 δ_rel を返す。logit に効く絶対発散は δ_abs = δ_rel·scale。
    これを predicted_flip_bound に通すことで、第2ベンダーを走らせる前に、静的な
    op グラフ＋代表的な logit 分布だけからタスク影響（判断フリップ率の上界）を予測できる。
    視点4（propagation）→ 視点8（decision）をつなぐ橋。
    """
    x = np.asarray(ref_logits, dtype=np.float64)
    scale = float(np.sqrt(np.mean(x ** 2)) + 1e-30)
    return predicted_flip_bound(ref_logits, rel_divergence * scale, confidence=confidence)


# ── 新視点11: タスク多様性 — argmax ⇏ 全タスク ─────────────────────────────────
# argmax フリップ率は多クラス分類専用。回帰・バイナリ・ランキングへ拡張する。

def regression_flip_rate(a: np.ndarray, b: np.ndarray, *,
                         atol: float = 0.0, rtol: float = 1e-3) -> float:
    """回帰タスクの判断フリップ率: 出力値が atol+rtol·|a| を超えて乖離するサンプル率。

    argmax は回帰に意味がない——値そのものが判断。許容は入力規模に相対的な rtol と
    絶対的な atol の組み合わせ（numpy allclose と整合）。
    スケール不変でないため rtol の設定はタスク依存（例: 価格予測では 0.1%, 物理シミュは 1e-5）。
    """
    a_ = np.asarray(a, dtype=np.float64).ravel()
    b_ = np.asarray(b, dtype=np.float64).ravel()
    if a_.size == 0:
        return 0.0
    tol = atol + rtol * np.abs(a_)
    return float(np.mean(np.abs(a_ - b_) > tol))


def binary_flip_rate(a: np.ndarray, b: np.ndarray, *,
                     threshold: float = 0.5) -> float:
    """バイナリ分類の判断フリップ率: sigmoid 出力が threshold を跨ぐサンプル率。

    argmax よりもマージン（= |出力 − threshold|）が支配する:
    大きなマージンで同じ判断・0 付近でフリップしやすい（argmax の margin と類似の役割）。
    量子化（int8）や dtype 変換で threshold 付近の出力が揺れやすい（tie_rate と対応）。
    """
    a_ = np.asarray(a, dtype=np.float64).ravel()
    b_ = np.asarray(b, dtype=np.float64).ravel()
    if a_.size == 0:
        return 0.0
    return float(np.mean((a_ >= threshold) != (b_ >= threshold)))


def binary_margin(a: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """バイナリ出力の決定境界からのマージン（= |a − threshold|）。

    argmax タスクの margin(logit) に相当。小さいほど near-tie（フリップしやすい）。
    """
    return np.abs(np.asarray(a, dtype=np.float64).ravel() - threshold)


def ranking_flip_rate(scores_a: np.ndarray, scores_b: np.ndarray, *, k: int = 10) -> float:
    """ランキングタスクの判断フリップ率: top-k アイテム集合がベンダー間で変わる率。

    検索/推薦システムでは「上位 k 件が同じか」がユーザーに見える差。スコア値自体の
    乖離より集合一致が重要（argmax の topk_flip_rate と同じ思想・ndim=1 の listwise 版）。
    """
    a_ = np.asarray(scores_a, dtype=np.float64)
    b_ = np.asarray(scores_b, dtype=np.float64)
    if a_.ndim == 1:
        kk = min(k, a_.size)
        ta = set(np.argpartition(a_, -kk)[-kk:])
        tb = set(np.argpartition(b_, -kk)[-kk:])
        return 0.0 if ta == tb else 1.0
    n = a_.shape[0]
    kk = min(k, a_.shape[-1])
    ta = np.argpartition(a_, -kk, axis=-1)[:, -kk:]
    tb = np.argpartition(b_, -kk, axis=-1)[:, -kk:]
    ta.sort(axis=-1)
    tb.sort(axis=-1)
    return float(np.mean(np.any(ta != tb, axis=-1))) if n else 0.0


@dataclass
class TaskReport(FindingReport):
    """非分類タスク（回帰/バイナリ/ランキング）の判断フリップ所見。

    DecisionReport は argmax 分類専用。TaskReport はタスク種別に応じた flip_rate を持つ。
    """

    task: str = "regression"     # "regression" / "binary" / "ranking"
    flip_rate: float = 0.0
    flip_rate_ub: float = 0.0
    n: int = 0
    threshold: float = 0.5       # binary のみ
    k: int = 10                  # ranking のみ
    atol: float = 0.0            # regression のみ
    rtol: float = 1e-3           # regression のみ

    def to_text(self) -> str:  # type: ignore[override]
        detail = ""
        if self.task == "binary":
            detail = f", threshold={self.threshold}"
        elif self.task == "ranking":
            detail = f", k={self.k}"
        elif self.task == "regression":
            detail = f", atol={self.atol:.1e}, rtol={self.rtol:.1e}"
        return super().to_text(
            header=(f"task={self.task} flip_rate={self.flip_rate * 100:.2f}%"
                    f"(≤{self.flip_rate_ub * 100:.2f}% Wilson) "
                    f"(n={self.n}{detail})"),
            empty=f"(no {self.task} decision flips — task-equivalent)")


def compare_task(a: np.ndarray, b: np.ndarray, *, task: str,
                 flip_budget: float = 0.0, threshold: float = 0.5, k: int = 10,
                 atol: float = 0.0, rtol: float = 1e-3,
                 confidence: float = 0.95) -> TaskReport:
    """非分類タスク（回帰/バイナリ/ランキング）のタスクレベル等価判定。

    task: "regression"（値の許容乖離）/ "binary"（sigmoid+threshold）/
          "ranking"（top-k 集合一致）。
    分類は compare_decisions へ（argmax は多クラス専用）。

    これにより decision 層が非分類タスクの出荷判断を持てる:
      - regression モデル（価格/物理量/埋め込み距離）
      - バイナリ分類（医療診断/スパム/異常検知）
      - 検索・推薦（上位 k 件が変わるか）

    fail-safe: 予算判定には観測 flip_rate（点推定）でなく flip_rate_ub（Wilson 上側限界・
    rollout.flip_rate_upper_bound を再利用）を使う。compare_decisions と同型の盲点
    （小標本での 0 件観測を「フリップ率 0%」と過信する偽OK）を同様に埋める。

    ranking の 1D 入力（単一クエリ）は集合一致/不一致の *決定的* な 0.0/1.0 を返し、
    複数試行から推定した比率ではないため Wilson widening を適用しない
    （flip_rate_ub = flip_rate のまま）。ranking の 2D 入力（クエリのバッチ）は
    クエリ数（a_.shape[0]）を試行数として widening する。
    """
    from .rollout import flip_rate_upper_bound
    a_ = np.asarray(a, dtype=np.float64)
    b_ = np.asarray(b, dtype=np.float64)
    n = int(a_.ravel().size)
    if task == "regression":
        fr = regression_flip_rate(a_, b_, atol=atol, rtol=rtol)
    elif task == "binary":
        fr = binary_flip_rate(a_, b_, threshold=threshold)
    elif task == "ranking":
        fr = ranking_flip_rate(a_, b_, k=k)
    else:
        raise ValueError(f"unknown task: {task!r} (regression/binary/ranking)")
    if task == "ranking" and a_.ndim == 1:
        fr_ub = fr    # 決定的な単一比較結果（推定値でない）ゆえ信頼区間は無意味
    else:
        n_trials = int(a_.shape[0]) if task == "ranking" else n
        fr_ub = (flip_rate_upper_bound(int(round(fr * n_trials)), n_trials,
                                       confidence=confidence)
                 if n_trials else fr)
    rep = TaskReport(task=task, flip_rate=fr, flip_rate_ub=fr_ub, n=n,
                     threshold=threshold, k=k, atol=atol, rtol=rtol)
    if fr_ub > flip_budget:
        risk = Risk.BLOCK if fr_ub > max(10 * flip_budget, 0.01) else Risk.WARN
        rep.add(risk, "task",
                f"{task} フリップ率 {fr * 100:.2f}%（上側限界 {fr_ub * 100:.2f}%）"
                f"> 予算 {flip_budget * 100:.2f}% → ベンダー間でユーザーに見える判断が変わる")
    elif fr > 0.0:
        rep.add(Risk.INFO, "task", f"{task} フリップ {fr * 100:.2f}%（予算内）")
    return rep


@dataclass
class DecisionReport(FindingReport):
    flip_rate: float = 0.0
    flip_rate_ub: float = 0.0
    n: int = 0
    flipped_margin_median: float = 0.0
    overall_margin_median: float = 0.0
    predicted_bound: float = 0.0
    systematic_frac: float = 0.0
    topk: int = 1
    topk_flip_rate: float = 0.0
    top_p: float = 0.0
    nucleus_flip_rate: float = 0.0
    tie_rate: float = 0.0

    def to_text(self) -> str:  # type: ignore[override]
        tk = (f", top-{self.topk} set flip={self.topk_flip_rate * 100:.2f}%"
              if self.topk > 1 else "")
        tp = (f", nucleus(p={self.top_p}) flip={self.nucleus_flip_rate * 100:.2f}%"
              if self.top_p > 0 else "")
        ti = f", tie={self.tie_rate * 100:.1f}%" if self.tie_rate > 0 else ""
        return super().to_text(
            header=(f"decision equivalence: flip_rate={self.flip_rate * 100:.2f}%"
                    f"(≤{self.flip_rate_ub * 100:.2f}% Wilson) "
                    f"(n={self.n}, bound≤{self.predicted_bound * 100:.2f}%, "
                    f"systematic={self.systematic_frac * 100:.0f}%{tk}{tp}{ti}, "
                    f"flipped-margin {self.flipped_margin_median:.3g} "
                    f"vs overall {self.overall_margin_median:.3g})"),
            empty="(no decision flips — task-equivalent)")


def compare_decisions(a: np.ndarray, b: np.ndarray, *, flip_budget: float = 0.0,
                      ref: np.ndarray | None = None, topk: int = 1,
                      top_p: float = 0.0, temperature: float = 1.0,
                      confidence: float = 0.95) -> DecisionReport:
    """タスクレベルの等価判定（数値でなく判断のフリップで測る）。

    flip_budget: 許容する判断フリップ率（タスク予算・例 0.001 = 0.1%）。
    ref: マージン基準の logit（既定 a）。
    topk: >1 なら生成タスク向けに top-k 候補集合フリップ率も併記する。
    bound は *残差*（argmax 保存的な系統成分を除いた成分）で評価し系統発散の過大評価を排す。

    fail-safe: 予算判定には観測 flip_rate（点推定）でなく flip_rate_ub（Wilson 上側限界）を
    使う。n が小さい評価バッチではたまたま観測フリップが少なくても母集団の真の率は
    高いことがある（第21回の predicted_flip_bound と同型の盲点・rule-of-three）。
    n が大きければ上限は点推定にほぼ収束し挙動は変わらない。
    """
    from .rollout import flip_rate_upper_bound
    flips = decision_flips(a, b)
    ref_logits = a if ref is None else ref
    m = margin(ref_logits)
    fm = m[flips]
    decomp = decompose_divergence(a, b)
    n = int(np.argmax(a, axis=-1).size)
    fr = flip_rate(a, b)
    fr_ub = flip_rate_upper_bound(int(round(fr * n)), n, confidence=confidence) if n else 0.0
    rep = DecisionReport(
        flip_rate=fr,
        flip_rate_ub=fr_ub,
        n=n,
        flipped_margin_median=float(np.median(fm)) if fm.size else 0.0,
        overall_margin_median=float(np.median(m)) if m.size else 0.0,
        predicted_bound=predicted_flip_bound(ref_logits, decomp["residual"], confidence=confidence),
        systematic_frac=decomp["systematic_frac"],
        topk=topk,
        topk_flip_rate=topk_flip_rate(a, b, topk) if topk > 1 else flip_rate(a, b),
        top_p=top_p,
        nucleus_flip_rate=nucleus_flip_rate(a, b, top_p, temperature) if top_p > 0 else 0.0,
        tie_rate=tie_rate(ref_logits),
    )
    if rep.tie_rate > 0.01:
        rep.add(Risk.WARN, "task",
                f"同点率 {rep.tie_rate * 100:.1f}%: argmax が規約依存（量子化/マスク）。"
                "ベンダー間の tie-break 規約差で数値発散ゼロでもフリップしうる（誤帰属に注意）")
    if rep.flip_rate_ub > flip_budget:
        risk = Risk.BLOCK if rep.flip_rate_ub > max(_FLIP_BLOCK_RATIO * flip_budget, _FLIP_BLOCK_MIN) else Risk.WARN
        rep.add(risk, "task",
                f"判断フリップ率 {rep.flip_rate * 100:.2f}%（上側限界 {rep.flip_rate_ub * 100:.2f}%）"
                f"> 予算 {flip_budget * 100:.2f}% → ベンダー間でユーザーに見える予測が変わる")
    elif rep.flip_rate > 0.0:
        rep.add(Risk.INFO, "task",
                f"判断フリップ {rep.flip_rate * 100:.2f}%（予算内）・near-tie に集中")
    # 健全性: フリップは低マージン(near-tie)の裾に集中するはず。確信領域で起きるなら異常。
    if fm.size and rep.overall_margin_median > 0 and \
            rep.flipped_margin_median > _NEAR_TIE_MARGIN_FRAC * rep.overall_margin_median:
        rep.add(Risk.WARN, "task",
                "フリップが near-tie 裾に集中していない → 確信予測まで変化・系統的発散を疑う")
    # 数値的に大きくても argmax 保存的な系統発散ならタスクは等価（calibration の系統検出と対）
    if rep.systematic_frac > 0.9 and rep.flip_rate == 0.0:
        rep.add(Risk.INFO, "task",
                f"発散の {rep.systematic_frac * 100:.0f}% は argmax 保存的な系統成分（スケール/シフト）"
                "→ 数値的に大きくても判断は不変（タスク等価）")
    return rep
