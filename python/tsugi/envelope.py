"""Tsugi envelope — 数値エンベロープの実行時検査。

盲点: portability/tolerance/feasibility/propagation はすべて *デプロイ前の静的検証*。
等価性は「scale・条件数・K がこうである」という前提の下で *認証* される
（tolerance は scale を、propagation は cond を仮定する）。だが本番では NVIDIA も
oracle も第2ベンダーも存在せず、AMD 単体で *未知の入力* が流れる。
発散はデータ依存（propagation で見た通り条件数次第）なので、本番入力が認証時の
*エンベロープ（認証済み動作範囲）を逸脱* すると、静的保証は静かに無効化される。

新視点: 静的「証明」を一度きりで信じるのでなく、**認証済みエンベロープ + その前提が
成り立つかを実行時に検査する契約**（design-by-contract を数値に適用）にする。
oracle も第2ベンダーも不要・単一ベンダーで計算できる安価な数値ヘルスチェック:
  - dtype 上限超え（fp16 overflow / inf）・denormal 域（FTZ がベンダー差を生む）
  - 出力スケールが認証時の scale_max を超過 → 認証 atol が無効
  - softmax/exp の logit が dtype の exp-overflow 閾値を超過（fp16 で特に致命的）

dtype 数値限界は IEEE 754 の実値。深刻度モデル（Risk/Finding）は report 層と共有。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .report import FindingReport, Risk  # 検証層共通の深刻度モデル

# --- 閾値定数 (Q4: magic number 排除) ---
# dtype 上限の何割を超えたら overflow 近接 WARN にするか（0.1 = 10%）。
_OVERFLOW_WARN_FRAC: float = 0.1
# exp-overflow 閾値の何割を超えたら softmax WARN にするか（0.7 = 70%）。
# fp16 の exp-overflow ≈ 11.09（ln 65504）なので 7.76 超で警告。
_EXP_WARN_FRAC: float = 0.7
# 認証 scale_max の何倍を超えたら BLOCK にするか（1.5 = 50% 超過）。
# 1.0× は「近接（WARN）」、1.5× 超は「認証無効（BLOCK）」。
_SCALE_BLOCK_RATIO: float = 1.5


@dataclass(frozen=True)
class DtypeLimits:
    """IEEE 754 の数値限界（overflow / denormal / exp-overflow 閾値）。"""

    max_normal: float    # 表現可能な最大正規数
    min_normal: float    # 最小正規数（これ未満は denormal → FTZ でベンダー差）
    exp_overflow: float  # exp(x) が max_normal を超える x = ln(max_normal)


# fp16 は範囲が狭く overflow が主リスク・bf16 は範囲広いが precision/denormal が主リスク。
# TF32 は指数部が fp32 と同じため overflow リスクは fp32 と同等（bf16/fp32 と同じ DtypeLimits）。
DTYPE_LIMITS: dict[str, DtypeLimits] = {
    "float16":  DtypeLimits(65504.0, 6.103515625e-05, math.log(65504.0)),          # exp_of ≈ 11.09
    "bfloat16": DtypeLimits(3.3895314e38, 1.1754944e-38, math.log(3.3895314e38)),  # exp_of ≈ 88.7
    "float32":  DtypeLimits(3.4028235e38, 1.1754944e-38, math.log(3.4028235e38)),  # exp_of ≈ 88.7
    "float64":  DtypeLimits(1.7976931348623157e308, 2.2250738585072014e-308,
                            math.log(1.7976931348623157e308)),                      # exp_of ≈ 709.8
    # TF32: 指数部は fp32 と同じ → overflow/denormal リスクは fp32 と同等
    "tf32":     DtypeLimits(3.4028235e38, 1.1754944e-38, math.log(3.4028235e38)),  # exp_of ≈ 88.7
}


def dtype_limits(dtype: str) -> DtypeLimits:
    return DTYPE_LIMITS.get(dtype, DTYPE_LIMITS["float32"])


@dataclass
class Envelope:
    """等価性を認証したときの前提（= 保証が有効な動作範囲）。"""

    dtype: str
    scale_max: float        # 認証時に仮定した出力スケール（RMS）
    cond_max: float = 1.0   # 認証時に仮定した条件数
    K: int = 1              # 認証時に仮定した累積深さ
    certified_atol: float = 0.0  # この前提下で保証した絶対許容（記録用）

    def to_text(self) -> str:
        lim = dtype_limits(self.dtype)
        return (f"certified envelope [{self.dtype}]: "
                f"scale ≤ {self.scale_max:.3g}, cond ≤ {self.cond_max:.3g}, K={self.K} "
                f"→ atol={self.certified_atol:.2e} "
                f"(dtype max={lim.max_normal:.3g}, exp-overflow at |x|>{lim.exp_overflow:.2f})")


def certify_gemm(K: int, dtype: str = "float16", scale: float = 1.0,
                 cond: float = 1.0, noise_floor: float = 0.0) -> Envelope:
    """tolerance 導出と同じ前提でエンベロープを発行する（静的層との接続）。"""
    from .tolerance import derive_tolerance

    tol = derive_tolerance(K, dtype, scale, noise_floor)
    return Envelope(dtype=dtype, scale_max=scale, cond_max=cond, K=K,
                    certified_atol=tol["atol"])


@dataclass
class EnvelopeReport(FindingReport):
    @property
    def in_envelope(self) -> bool:
        return self.ok

    def to_text(self) -> str:  # type: ignore[override]
        status = "IN-ENVELOPE" if self.in_envelope else "OUT-OF-ENVELOPE"
        return super().to_text(header=f"envelope check ({status})",
                               empty="(within certified envelope)")


def check_tensor(x: np.ndarray, env: Envelope) -> EnvelopeReport:
    """本番テンソルが認証済みエンベロープ内かを単一ベンダーで検査する（oracle 不要）。"""
    rep = EnvelopeReport()
    lim = dtype_limits(env.dtype)
    xf = np.asarray(x, dtype=np.float64)

    # NaN/Inf は即破綻（ベンダー間で伝播挙動も異なる）
    if not np.all(np.isfinite(xf)):
        rep.add(Risk.BLOCK, "tensor", "NaN/Inf 検出 → 数値破綻・ベンダー間挙動も発散")
        return rep

    max_abs = float(np.abs(xf).max()) if xf.size else 0.0
    # dtype overflow（特に fp16 は max=65504 と狭い）
    if max_abs >= lim.max_normal:
        rep.add(Risk.BLOCK, "tensor",
            f"max|x|={max_abs:.3g} ≥ {env.dtype} 上限 {lim.max_normal:.3g} → overflow/inf")
    elif max_abs >= _OVERFLOW_WARN_FRAC * lim.max_normal:
        rep.add(Risk.WARN, "tensor",
            f"max|x|={max_abs:.3g} が {env.dtype} 上限の {_OVERFLOW_WARN_FRAC*100:.0f}% 超 → overflow 近接")

    # denormal 域: FTZ（flush-to-zero）の有無がベンダーで異なり発散源になる
    nz = np.abs(xf[xf != 0.0])
    if nz.size:
        min_abs = float(nz.min())
        if min_abs < lim.min_normal:
            rep.add(Risk.WARN, "tensor",
                f"min nonzero |x|={min_abs:.3g} < {env.dtype} 最小正規数 {lim.min_normal:.3g} "
                "→ denormal・FTZ 挙動がベンダー差を生む")

    # 出力スケールが認証時の前提を超過 → 認証 atol はもはや無効
    scale = float(np.sqrt(np.mean(xf ** 2))) if xf.size else 0.0
    if scale > env.scale_max * _SCALE_BLOCK_RATIO:
        implied = env.certified_atol * (scale / max(env.scale_max, 1e-30))
        rep.add(Risk.BLOCK, "scale",
            f"実スケール {scale:.3g} が認証 scale_max {env.scale_max:.3g} を {_SCALE_BLOCK_RATIO}× 超 "
            f"→ 認証 atol={env.certified_atol:.2e} 無効（実許容 ~{implied:.2e}）・要再認証")
    elif scale > env.scale_max:
        rep.add(Risk.WARN, "scale",
            f"実スケール {scale:.3g} が認証 scale_max {env.scale_max:.3g} 近接 → 余裕が縮小")
    return rep


def check_softmax_input(logits: np.ndarray, env: Envelope) -> EnvelopeReport:
    """softmax/exp 入力が dtype の exp-overflow 閾値内かを検査する。

    fp16 は ln(65504)≈11.09 で exp が inf。片方のベンダーが fp16 で exp を計算すると
    ここで破綻し、もう片方（f32 経路）と壊滅的に発散する。max-subtract 前の生 logit を渡す。
    """
    rep = EnvelopeReport()
    lim = dtype_limits(env.dtype)
    xf = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(xf)):
        rep.add(Risk.BLOCK, "softmax", "logit に NaN/Inf")
        return rep
    max_logit = float(np.abs(xf).max()) if xf.size else 0.0
    if max_logit > lim.exp_overflow:
        rep.add(Risk.BLOCK, "softmax",
            f"max|logit|={max_logit:.2f} > {env.dtype} exp-overflow {lim.exp_overflow:.2f} "
            "→ exp が inf（max-subtract 未適用なら片ベンダーで softmax 破綻）")
    elif max_logit > _EXP_WARN_FRAC * lim.exp_overflow:
        rep.add(Risk.WARN, "softmax",
            f"max|logit|={max_logit:.2f} が exp-overflow {lim.exp_overflow:.2f} の "
            f"{_EXP_WARN_FRAC*100:.0f}% に近接 → max-subtract 必須・ベンダー差が出やすい")
    return rep


def channel_scale_spread(x: np.ndarray, axis: int = -1) -> float:
    """チャネル別 RMS の広がり = max(channel RMS) / median(channel RMS)。

    outlier feature（数チャネルだけ巨大＝LLM の massive activations）で桁違いに大きくなる。
    near-uniform-scale なら ~1。
    """
    xf = np.asarray(x, dtype=np.float64)
    if xf.ndim < 2:
        return 1.0
    chan = np.sqrt(np.mean(xf ** 2, axis=tuple(i for i in range(xf.ndim) if i != axis % xf.ndim)))
    med = float(np.median(chan))
    return float(chan.max() / (med + 1e-30)) if med > 0 else 1.0


def check_outlier_features(x: np.ndarray, axis: int = -1,
                           spread_warn: float = 10.0) -> EnvelopeReport:
    """outlier feature を検出し、単一スケール（global RMS）仮定の破綻を警告する。

    tolerance/envelope/detectability floor は単一の scale を仮定する。だが実 LLM 活性は
    一部チャネルが 100–1000x 大きい（massive activations・LLM.int8/SmoothQuant が示す通り）。
    その場合 global scale は outlier チャネルを過小評価し、そこでの発散が誤った許容で
    判定される。チャネル scale 広がりが大きければ per-channel 検証が要ると警告する。
    """
    rep = EnvelopeReport()
    spread = channel_scale_spread(x, axis)
    if spread > spread_warn:
        rep.add(Risk.WARN, "outlier",
            f"チャネル scale 広がり {spread:.0f}× → outlier feature 検出。"
            "global scale 単一仮定が破綻（per-channel 許容/検証が必要・massive activations）")
    return rep
