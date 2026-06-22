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

import numpy as np

from .report import FindingReport, Risk


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
        return 2 ** 62                       # 実質 ∞（int で扱える上限）
    if p >= 1.0:
        return 0
    c = min(max(confidence, 0.0), 1.0 - 1e-12)
    return int(math.floor(math.log(c) / math.log(1.0 - p)))


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
    safe_length: int = 0

    def to_text(self) -> str:  # type: ignore[override]
        exp = "∞" if math.isinf(self.expected_step) else f"{self.expected_step:.0f}"
        return super().to_text(
            header=(f"rollout (p={self.flip_rate * 100:.3f}%/tok, L={self.length}: "
                    f"survival={self.survival * 100:.2f}%, "
                    f"E[初回発散]=tok {exp}, safe_len={self.safe_length})"),
            empty="(sequence-equivalent within confidence)")


def analyze_rollout(flip_rate: float, target_length: int, *,
                    confidence: float = 0.99) -> RolloutReport:
    """per-token フリップ率を目標生成長 L に合成し、シーケンス等価の verdict を返す。

    判定:
      - target_length ≤ safe_len（survival ≥ confidence 圏内）→ OK
      - survival ≥ 0.5（一致が分岐より優勢だが confidence 未満）→ WARN
      - それ未満（L 内で分岐が優勢）→ BLOCK
    """
    p = min(max(flip_rate, 0.0), 1.0)
    surv = sequence_survival(p, target_length)
    safe = safe_generation_length(p, confidence)
    exp = expected_divergence_step(p)
    rep = RolloutReport(flip_rate=p, length=int(target_length), survival=surv,
                        expected_step=exp, safe_length=safe)
    exps = "∞" if math.isinf(exp) else f"{exp:.0f}"
    if p <= 0.0 or target_length <= safe:
        rep.add(Risk.OK, "rollout",
                f"L={target_length} は safe_len={safe} 以内 "
                f"(survival={surv * 100:.2f}% ≥ {confidence * 100:.0f}%)")
    elif surv >= 0.5:
        rep.add(Risk.WARN, "rollout",
                f"per-token {p * 100:.3f}% は許容でも L={target_length} で survival="
                f"{surv * 100:.2f}% (<{confidence * 100:.0f}%)・初回発散 ~tok {exps}")
    else:
        rep.add(Risk.BLOCK, "rollout",
                f"per-token {p * 100:.3f}% が L={target_length} で複利的に増幅: survival="
                f"{surv * 100:.2f}%・初回発散 ~tok {exps}・safe_len={safe} のみ "
                f"(per-token 許容 ⇏ per-sequence 許容)")
    return rep


def rollout_from_logits(logits_a: np.ndarray, logits_b: np.ndarray,
                        target_length: int, *, confidence: float = 0.99) -> RolloutReport:
    """代表 logit 群から per-token フリップ率を測り、生成長 L へ合成する。

    仮定: 渡した logit は生成中の代表的ステップ群で、フリップ率は定常（位置に依らず一定）。
    分布シフト時は再評価が要る（provenance/decision と同じ前提）。
    """
    from .decision import flip_rate as _flip_rate
    p = _flip_rate(logits_a, logits_b)
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
