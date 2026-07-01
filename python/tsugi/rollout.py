"""Tsugi rollout — 自己回帰的発散（per-token 等価 ⇏ per-sequence 等価・新視点9）。

盲点: decision（新視点8）は *1 トークンの* 判断フリップ率 p を測る。だが出荷される LLM は
*自己回帰* で生成する —— トークン t+1 は t に条件づく。ゆえに 2 ベンダーが各ステップで
確率 (1−p) で一致しても、*生成シーケンス* が一致するとは限らない:

  - **一度ズレたら戻らない**: あるトークンで判断が分かれると、以降の文脈が分岐し、
    後続トークンは別軌道（無相関）になる。回復はほぼ不可能。
  - **ゆえに意味ある量は per-token でなく per-sequence**: 長さ L の生成が一致する確率は
    survival = (1−p)^L、初回発散ステップの期待値は 1/p（幾何分布）。

これは propagation（新視点4: per-kernel ⇏ per-model）の自己回帰版である。propagation は
発散を op グラフの *深さ* に沿って合成した。rollout は判断フリップ risk を生成 *長* に沿って
合成する。どちらも「局所の等価は大域の等価に合成されない」。

実証（numpy）: p=1% は 1 トークンでは「許容」に見えるが、L=100 で survival=0.37、
L=1000 で 0.004 —— 自己回帰生成では実質確実に分岐する。per-token の許容判断は
シーケンスでは破綻する。本当に問うべきは「何 % のトークンが変わるか」でなく
「ベンダー間で *同じ文章* を何トークン保てるか」。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .report import FindingReport, Risk

_INF_LEN = 2 ** 62   # safe_generation_length の実質 ∞（int で扱える上限・p=0 用）


def sequence_survival(flip_rate: float, length: int) -> float:
    """長さ `length` の生成が 2 ベンダーで完全一致する確率 = (1−p)^length。

    全ステップで一致して初めてシーケンス一致（1 度でも分岐すれば以降無相関）。
    """
    p = min(max(flip_rate, 0.0), 1.0)
    return (1.0 - p) ** max(0, int(length))


def expected_divergence_step(flip_rate: float) -> float:
    """初回発散トークンの期待位置 = 1/p（幾何分布の平均）。p=0 なら ∞（決して分岐しない）。"""
    p = min(max(flip_rate, 0.0), 1.0)
    return math.inf if p <= 0.0 else 1.0 / p


def safe_generation_length(flip_rate: float, confidence: float = 0.99) -> int:
    """survival ≥ confidence を保てる最大生成長 L。

    L = floor(log(confidence)/log(1−p))。p=0 なら ∞（任意長で安全）、p=1 なら 0。
    """
    p = min(max(flip_rate, 0.0), 1.0)
    if p <= 0.0:
        return _INF_LEN                      # 実質 ∞
    if p >= 1.0:
        return 0
    c = min(max(confidence, 0.0), 1.0 - 1e-12)
    return int(math.floor(math.log(c) / math.log(1.0 - p)))


def flip_rate_upper_bound(flips: int, n_samples: int, confidence: float = 0.95) -> float:
    """観測フリップ数からの p の片側上側信頼限界（Wilson）。

    盲点の修正: 点推定 p̂=flips/n は小標本で過小評価する。特に 0 フリップ観測でも
    p=0 を意味しない（rule of three: p ≲ 3/n）。移植可を *過信* するのは calibration
    （新視点6）の偽OK と同じ致命傷ゆえ、rollout は既定で上限を使い fail-safe に倒す。

    n=0（データ無し）は 1.0（最大不確実＝最も保守的）を返す。
    """
    if n_samples <= 0:
        return 1.0
    k = min(max(int(flips), 0), int(n_samples))
    phat = k / n_samples
    z = NormalDist().inv_cdf(min(max(confidence, 0.5), 1.0 - 1e-12))   # 片側
    z2 = z * z
    denom = 1.0 + z2 / n_samples
    center = phat + z2 / (2.0 * n_samples)
    margin = z * math.sqrt(phat * (1.0 - phat) / n_samples + z2 / (4.0 * n_samples ** 2))
    return min(1.0, (center + margin) / denom)


def divergence_step_quantile(flip_rate: float, q: float = 0.5) -> float:
    """初回発散が起きるトークン位置の q 分位（例 q=0.5 で中央値）。

    幾何分布の CDF が q に達する最小ステップ = ceil(log(1−q)/log(1−p))。
    """
    p = min(max(flip_rate, 0.0), 1.0)
    if p <= 0.0:
        return math.inf
    if p >= 1.0:
        return 1.0
    qq = min(max(q, 0.0), 1.0 - 1e-12)
    return math.ceil(math.log(1.0 - qq) / math.log(1.0 - p))


@dataclass
class RolloutReport(FindingReport):
    """自己回帰生成のシーケンス等価所見（per-token を生成長へ合成）。"""

    flip_rate: float = 0.0
    length: int = 0
    survival: float = 1.0
    expected_step: float = math.inf
    median_step: float = math.inf
    safe_length: int = 0

    def to_text(self) -> str:  # type: ignore[override]
        exp = "∞" if math.isinf(self.expected_step) else f"{self.expected_step:.0f}"
        med = "∞" if math.isinf(self.median_step) else f"{self.median_step:.0f}"
        return super().to_text(
            header=(f"rollout (p={self.flip_rate * 100:.3f}%/tok, L={self.length}: "
                    f"survival={self.survival * 100:.2f}%, "
                    f"E[初回発散]=tok {exp} / 中央値=tok {med}, safe_len={self.safe_length})"),
            empty="(sequence-equivalent within confidence)")


def analyze_rollout(flip_rate: float, target_length: int, *,
                    confidence: float = 0.99) -> RolloutReport:
    """per-token フリップ率を目標生成長 L に合成し、シーケンス等価の verdict を返す。

    判定:
      - target_length ≤ safe_len（survival ≥ confidence 圏内）→ OK
      - survival ≥ 0.5（一致が分岐より優勢だが confidence 未満）→ WARN
      - それ未満（L 内で分岐が優勢）→ BLOCK

    初回発散ステップは幾何分布に従い右に裾を引くため、平均（expected_step=1/p）は
    中央値（median_step≈ln2/p）より系統的に大きい —— 平均だけ見ると「典型的には
    もっと長く保つ」と楽観視しやすい（右裾の少数の長生存run に平均が引っ張られる）。
    fail-safe のため両方を報告する（divergence_step_quantile は実装済みだが従来
    どのレポートにも接続されていなかった）。
    """
    p = min(max(flip_rate, 0.0), 1.0)
    surv = sequence_survival(p, target_length)
    safe = safe_generation_length(p, confidence)
    exp = expected_divergence_step(p)
    med = divergence_step_quantile(p, 0.5)
    rep = RolloutReport(flip_rate=p, length=int(target_length), survival=surv,
                        expected_step=exp, median_step=med, safe_length=safe)
    exps = "∞" if math.isinf(exp) else f"{exp:.0f}"
    meds = "∞" if math.isinf(med) else f"{med:.0f}"
    if p <= 0.0 or target_length <= safe:
        rep.add(Risk.OK, "rollout",
                f"L={target_length} は safe_len={safe} 以内 "
                f"(survival={surv * 100:.2f}% ≥ {confidence * 100:.0f}%)")
    elif surv >= 0.5:
        rep.add(Risk.WARN, "rollout",
                f"per-token {p * 100:.3f}% は許容でも L={target_length} で survival="
                f"{surv * 100:.2f}% (<{confidence * 100:.0f}%)・初回発散 平均tok {exps}"
                f"/中央値tok {meds}")
    else:
        rep.add(Risk.BLOCK, "rollout",
                f"per-token {p * 100:.3f}% が L={target_length} で複利的に増幅: survival="
                f"{surv * 100:.2f}%・初回発散 平均tok {exps}/中央値tok {meds}"
                f"・safe_len={safe} のみ (per-token 許容 ⇏ per-sequence 許容)")
    return rep


def _per_token_flips(logits_a: np.ndarray, logits_b: np.ndarray, decode: str,
                     topk: int, top_p: float, temperature: float) -> tuple[int, int]:
    """デコード方式に整合した per-token フリップ (n_samples, n_flips) を返す。

    生成は argmax だけでない: top-k / nucleus サンプリングでは候補 *集合* が分岐すれば
    生成分布が分かれる。decision 層の集合フリップ率を再利用し、生成長へ合成する素材にする。
    """
    from .decision import decision_flips, nucleus_flip_rate, topk_flip_rate

    a = np.asarray(logits_a)
    b = np.asarray(logits_b)
    if decode == "greedy":
        flips = decision_flips(a, b)
        return int(flips.size), int(flips.sum())
    n = int(a.shape[0]) if a.ndim >= 1 else 0
    if decode == "topk":
        rate = topk_flip_rate(a, b, topk)
    elif decode == "nucleus":
        rate = nucleus_flip_rate(a, b, top_p, temperature)
    else:
        raise ValueError(f"unknown decode mode: {decode!r} (greedy/topk/nucleus)")
    return n, int(round(rate * n))


def rollout_from_logits(logits_a: np.ndarray, logits_b: np.ndarray,
                        target_length: int, *, confidence: float = 0.99,
                        conservative: bool = True,
                        sample_confidence: float = 0.95,
                        decode: str = "greedy", topk: int = 5,
                        top_p: float = 0.9, temperature: float = 1.0) -> RolloutReport:
    """代表 logit 群から per-token フリップ率を測り、生成長 L へ合成する。

    `conservative=True`（既定）: 観測フリップ率の点推定でなく **上側信頼限界**
    （`flip_rate_upper_bound`）を p に使う。小標本・0 フリップ観測で survival を 100% と
    誤報し移植可を過信する事故を防ぐ（fail-safe）。`conservative=False` で点推定。

    `decode`: 運用のデコード方式に per-token フリップ率を整合させる（改善: 既定の
    greedy argmax フリップ率は *サンプリング* 生成の per-token 発散を過小評価しうる
    —— 候補集合が分岐すれば argmax が同じでも生成分布は分かれる）。
      - "greedy" : argmax フリップ率（貪欲デコード）
      - "topk"   : top-k 候補集合フリップ率（k=`topk`）
      - "nucleus": top-p 集合フリップ率（`top_p`/`temperature`・確率依存）

    仮定: 渡した logit は生成中の代表的ステップ群で、フリップ率は定常（位置に依らず一定）。
    分布シフト時は再評価が要る（provenance/decision と同じ前提）。
    `survival` は *完全トークン一致* の確率（厳格な下界）であり、意味的に等価な別文は
    「発散」に数える —— task 等価より厳しい側に倒す指標である点に注意。
    """
    n, k = _per_token_flips(logits_a, logits_b, decode, topk, top_p, temperature)
    p = flip_rate_upper_bound(k, n, sample_confidence) if conservative else (k / n if n else 0.0)
    return analyze_rollout(p, target_length, confidence=confidence)


def simulate_rollout(flip_rate: float, length: int, trials: int = 2000,
                     seed: int = 0) -> float:
    """Monte Carlo でシーケンス survival 率を実測（解析式 (1−p)^L の確認用）。

    各ステップで確率 p のフリップを引き、length 内で 1 度もフリップしなければ survival。
    """
    p = min(max(flip_rate, 0.0), 1.0)
    rng = np.random.default_rng(seed)
    flips = rng.random((trials, max(0, int(length)))) < p
    survived = ~np.any(flips, axis=-1)
    return float(np.mean(survived)) if trials else 1.0
