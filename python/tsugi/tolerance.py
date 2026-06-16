"""Tsugi tolerance — 許容誤差を演算の数値条件から *導出* する（ソクラテス問答の新視点）。

盲点: equivalence は固定 1e-2 を使っていた。だが両ベンダーは累積順序が違うだけで
両方 IEEE 正当 — どちらも"真値"でない。許容すべき発散幅は「数学が許す範囲」であり、
累積深さ K・dtype の機械イプシロン・値スケールから導出できる。

モデル（誠実な近似・厳密な最悪ケース境界ではない）:
  GEMM の f16 入力・f32 累積では、支配的な発散は入力量子化（u_fp16）が累積で
  ランダムウォーク的に広がる項。絶対誤差 ~ safety * sqrt(K) * u_input * scale。
  → K が大きいほど許容も大きい（大K GEMM は正当に大きくズレる）。
"""
from __future__ import annotations

import math

from .constants import SAFETY

# 単位丸め誤差（unit roundoff, u = 2^-(mantissa_bits+1)）
UNIT_ROUNDOFF = {
    "float16": 2.0 ** -11,   # 10 仮数ビット → u ≈ 4.88e-4
    "bfloat16": 2.0 ** -8,   # 7 仮数ビット → u ≈ 3.91e-3（fp16 より粗い）
    "float32": 2.0 ** -24,   # 23 仮数ビット → u ≈ 5.96e-8
}


def unit_roundoff(dtype: str) -> float:
    return UNIT_ROUNDOFF.get(dtype, UNIT_ROUNDOFF["float32"])


def expected_gemm_abs_error(K: int, dtype: str = "float16",
                            scale: float = 1.0, safety: float = SAFETY) -> float:
    """K 次元の累積を持つ GEMM の、ベンダー間で正当に生じうる絶対誤差の目安。

    safety: 安全係数（モデルの粗さを吸収）。
    scale: 出力要素の典型的な大きさ（標準正規入力なら ~sqrt(K)）。
    """
    u = unit_roundoff(dtype)
    return safety * math.sqrt(max(1, K)) * u * scale


def derive_tolerance(K: int, dtype: str = "float16", scale: float = 1.0,
                     noise_floor: float = 0.0, safety: float = SAFETY) -> dict[str, float]:
    """導出された許容誤差。数値条件とノイズフロアの大きい方を採用。

    noise_floor: ハードウェアの run-to-run 非決定性の実測幅（あれば）。
    返り値: equivalence.compare に渡せる {atol, rtol} 形式。
    """
    derived = expected_gemm_abs_error(K, dtype, scale, safety)
    atol = max(derived, noise_floor)
    # 相対許容は dtype 由来の最小桁＋累積項
    rtol = max(unit_roundoff(dtype) * math.sqrt(max(1, K)) * safety, 1e-3)
    return {"atol": atol, "rtol": rtol, "derived": derived, "noise_floor": noise_floor}


def explain(K: int, dtype: str = "float16", scale: float = 1.0) -> str:
    """導出の内訳を人間可読に（なぜこの閾値かを説明）。"""
    u = unit_roundoff(dtype)
    tol = derive_tolerance(K, dtype, scale)
    return (f"K={K} dtype={dtype} scale={scale}: "
            f"u={u:.2e} sqrt(K)={math.sqrt(K):.1f} "
            f"→ atol={tol['atol']:.2e} rtol={tol['rtol']:.2e} "
            f"（固定 1e-2 と異なり K に応じて変化）")
