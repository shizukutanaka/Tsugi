"""Tsugi equivalence — クロスベンダー数値等価性の判定（新視点の柱その2）。

リファレンス oracle を真値に、2 つの実装結果（例: NVIDIA と AMD の GPU 出力）が
等価かを判定する。Triton は両ベンダーのカーネルを生成するが、両者が同じ数値を
出す保証はしない。ここがその保証を与える。

GPU が無い本環境では「擬似的に異なるベンダー」（累積順序を変えた matmul）を
生成し、検出器がズレを捕まえることを実証する（シミュレーション・明示）。
実 GPU 比較は同じ compare() を実機出力に適用するだけ。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .report import Risk  # 検証層共通の深刻度モデル

# dtype 別の許容誤差（BENCHMARK.md §4 と整合）
TOLERANCE = {
    "float16": dict(atol=1e-2, rtol=1e-2),
    "bfloat16": dict(atol=2e-2, rtol=2e-2),  # bf16 はベンダー差が大きい
    "float32": dict(atol=1e-4, rtol=1e-4),
}


@dataclass
class EquivalenceReport:
    equivalent: bool
    max_abs_err: float
    max_rel_err: float
    n_mismatch: int
    n_total: int
    atol: float
    rtol: float

    # report.FindingReport は所見リスト型。等価判定はスカラ計量なので継承せず、
    # 共通の判定インターフェース（risk/max_risk/ok）だけ揃えて第一級レポートにする。
    @property
    def risk(self) -> Risk:
        return Risk.OK if self.equivalent else Risk.BLOCK

    @property
    def max_risk(self) -> Risk:
        return self.risk

    @property
    def ok(self) -> bool:
        return self.equivalent

    def to_text(self) -> str:
        status = "EQUIVALENT" if self.equivalent else "DIVERGENT"
        return (f"[{status}] max_abs={self.max_abs_err:.3e} max_rel={self.max_rel_err:.3e} "
                f"mismatch={self.n_mismatch}/{self.n_total} (atol={self.atol}, rtol={self.rtol})")


def compare(a: np.ndarray, b: np.ndarray, dtype: str = "float16") -> EquivalenceReport:
    """2 実装の出力 a, b が等価かを判定する（固定許容・dtype 別）。

    a: 基準（例 リファレンス oracle / NVIDIA）
    b: 候補（例 AMD）
    """
    tol = TOLERANCE.get(dtype, TOLERANCE["float32"])
    return _compare_with(a, b, tol["atol"], tol["rtol"])


def compare_gemm(a: np.ndarray, b: np.ndarray, K: int, dtype: str = "float16",
                 scale: float | None = None, noise_floor: float = 0.0) -> EquivalenceReport:
    """GEMM 専用: 許容誤差を K・dtype から *導出* して判定（新視点）。

    固定 1e-2 でなく「数学が許す発散幅」で等価性を問う。両ベンダーは正当に異なる。
    scale 未指定時は出力の典型スケールを a から推定。
    """
    from .tolerance import derive_tolerance

    if scale is None:
        scale = float(np.sqrt(np.mean(np.abs(a.astype(np.float64)) ** 2)) + 1e-12)
    tol = derive_tolerance(K, dtype, scale, noise_floor)
    return _compare_with(a, b, tol["atol"], tol["rtol"])


def _compare_with(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> EquivalenceReport:
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    abs_err = np.abs(af - bf)
    rel_err = abs_err / (np.abs(af) + 1e-12)
    close = abs_err <= (atol + rtol * np.abs(af))
    n_mismatch = int(np.size(close) - np.count_nonzero(close))
    return EquivalenceReport(
        equivalent=bool(np.all(close)),
        max_abs_err=float(abs_err.max()),
        max_rel_err=float(rel_err.max()),
        n_mismatch=n_mismatch,
        n_total=int(np.size(close)),
        atol=atol,
        rtol=rtol,
    )


# --- 擬似ベンダー生成（検出器テスト用・シミュレーション） ------------------

def simulate_vendor_matmul(a: np.ndarray, b: np.ndarray, *,
                           accum: str = "f32", split_k: int = 1) -> np.ndarray:
    """異なるベンダーの数値挙動を擬似再現する（CPU・テスト専用）。

    accum="f16": fp16 で累積（一部 GPU 経路を模す・誤差大）
    split_k>1: K を分割して部分和を別精度で合算（累積順序差を模す）
    これは *シミュレーション*。実 GPU の値ではない（主張と実装の一致）。
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    acc_dtype = np.float16 if accum == "f16" else np.float32
    out = np.zeros((M, N), dtype=np.float32)
    step = max(1, K // split_k)
    for k0 in range(0, K, step):
        k1 = min(k0 + step, K)
        partial = (a[:, k0:k1].astype(np.float32) @ b[k0:k1, :].astype(np.float32))
        out += partial.astype(acc_dtype).astype(np.float32)
    return out
