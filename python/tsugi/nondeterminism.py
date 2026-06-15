"""Tsugi nondeterminism — 非決定実行とノイズフロア（出力は点でなく分布）。

盲点: 既存 6 視点＋較正メタ層はすべて「同じカーネルを同じベンダーで走らせれば
同じ結果が出る」を暗黙に仮定していた。だが GPU の atomic 加算（split-K の atomicAdd 等）は
スレッド到着順で和の順序が変わり、run-to-run で結果が揺れる。**ベンダーの出力は
固定点でなく分布**である。

帰結:
- 単一 run A と単一 run B の比較は、2 つの差の源 ——（a）ベンダー *内* の run-to-run
  ノイズ と（b）ベンダー *間* の発散 —— を混同する。両者を分離できなければ食い違いを
  attribute できず、そもそも「ベンダー A の正しい答え」が一意に定義できない。
- クロスベンダー差が **ノイズフロア未満** なら、「A vs B」は「A vs A（別 run）」と区別不能
  → 等価でも発散でもなく **INDISTINGUISHABLE**（判定原理的に未定義）。
- これは calibration の検出限界（safety·√K·u）とは独立した第二の床。検証器の実効分解能 =
  max(数値検出限界, ノイズフロア)。ノイズが数値許容を超えれば検証器は *ノイズ律速* になり、
  ノイズを発散と誤判定（偽BLOCK）するか、許容を緩めて偽OK を増やす。

tolerance.derive_tolerance は noise_floor 引数を持つが既定 0（＝決定論を仮定）だった。
本モジュールはその noise_floor を **複数 run の実測** で埋める（CPU では atomic 非決定を
擬似再現・明示）。実 GPU では run_fn を実機カーネルにすればそのまま使える。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .report import FindingReport, Risk

EQUIVALENT = "EQUIVALENT"
DIVERGENT = "DIVERGENT"
INDISTINGUISHABLE = "INDISTINGUISHABLE"  # クロス差がノイズ未満＝判定未定義


def simulate_nondeterministic_reduction(parts: np.ndarray, seed: int) -> np.ndarray:
    """atomicAdd の非決定スケジュールを擬似再現する（CPU・シミュレーション・明示）。

    parts: 部分和（split-K の各タイル和など）。到着順をランダム置換し fp32 で逐次加算。
    seed ごとに順序が変わり run-to-run の揺れを生む。実 GPU の値ではない。
    """
    rng = np.random.default_rng(seed)
    flat = parts.reshape(-1).astype(np.float32)
    order = rng.permutation(flat.size)
    acc = np.float32(0.0)
    for i in order:
        acc = np.float32(acc + flat[i])
    return np.asarray(acc, dtype=np.float32)


def measure_noise_floor(run_fn: Callable[[int], np.ndarray], n_runs: int = 16,
                        seed0: int = 0) -> dict[str, float]:
    """同一ベンダー・同一入力を n_runs 回走らせ run-to-run ノイズを実測する。

    run_fn(seed) -> 出力テンソル。返り値の spread（要素ごと max-min の最大）を
    保守的なノイズフロア（= tolerance.derive_tolerance の noise_floor）として使う。
    """
    runs = [np.asarray(run_fn(seed0 + i), dtype=np.float64) for i in range(n_runs)]
    stack = np.stack(runs)
    spread = float((stack.max(axis=0) - stack.min(axis=0)).max())
    mean_mag = float(np.sqrt(np.mean(stack ** 2)) + 1e-30)
    return {"spread": spread, "std": float(stack.std(axis=0).max()),
            "rel": spread / mean_mag, "n_runs": n_runs}


def attribute(cross_diff: float, noise_floor: float, tol: float) -> str:
    """クロス差を 3 状態に帰属する（noise_floor <= tol を前提）。

    [0, noise]      → INDISTINGUISHABLE（ベンダー内ノイズと区別不能）
    (noise, tol]    → EQUIVALENT（ノイズと区別できる正当な数値差）
    (tol, ∞)        → DIVERGENT
    """
    if cross_diff <= noise_floor:
        return INDISTINGUISHABLE
    if cross_diff <= tol:
        return EQUIVALENT
    return DIVERGENT


@dataclass
class StabilityReport(FindingReport):
    verdict: str = INDISTINGUISHABLE
    noise_floor: float = 0.0
    cross_diff: float = 0.0
    tol: float = 0.0
    numerical_floor: float = 0.0

    @property
    def noise_limited(self) -> bool:
        """ノイズが数値許容を支配＝検証器の分解能が HW ノイズで決まる。"""
        return self.noise_floor > self.numerical_floor

    def to_text(self) -> str:  # type: ignore[override]
        return super().to_text(
            header=(f"stability [{self.verdict}] cross={self.cross_diff:.2e} "
                    f"noise={self.noise_floor:.2e} tol={self.tol:.2e}"),
            empty="(comparison is well-resolved)")


def compare_stable(run_a: Callable[[int], np.ndarray],
                   run_b: Callable[[int], np.ndarray], K: int,
                   dtype: str = "float16", n_runs: int = 16) -> StabilityReport:
    """方法論的に健全なクロスベンダー比較（出力を分布として扱う）。

    1) 各ベンダーの run-to-run ノイズを実測 → noise_floor
    2) noise を織り込んだ許容を導出（決定論仮定 noise=0 を排す）
    3) クロス差を noise/tol に対し 3 状態へ帰属（INDISTINGUISHABLE を正直に出す）
    """
    from .tolerance import derive_tolerance, expected_gemm_abs_error

    nf_a = measure_noise_floor(run_a, n_runs)
    nf_b = measure_noise_floor(run_b, n_runs)
    noise = max(nf_a["spread"], nf_b["spread"])

    a = np.asarray(run_a(0), dtype=np.float64)
    b = np.asarray(run_b(0), dtype=np.float64)
    scale = float(np.sqrt(np.mean(a ** 2)) + 1e-30)
    numerical = expected_gemm_abs_error(K, dtype, scale)
    tol = derive_tolerance(K, dtype, scale, noise_floor=noise)["atol"]
    cross = float(np.abs(a - b).max())

    rep = StabilityReport(verdict=attribute(cross, noise, tol), noise_floor=noise,
                          cross_diff=cross, tol=tol, numerical_floor=numerical)
    if rep.verdict == INDISTINGUISHABLE:
        rep.add(Risk.WARN, "noise",
                f"クロス差 {cross:.2e} ≤ run-to-run ノイズ {noise:.2e} "
                "→ ベンダー内ノイズと区別不能・等価判定は未定義")
    elif rep.verdict == DIVERGENT:
        rep.add(Risk.BLOCK, "cross",
                f"クロス差 {cross:.2e} > 許容 {tol:.2e}（ノイズ {noise:.2e} 超）→ 真の発散")
    if rep.noise_limited:
        rep.add(Risk.WARN, "noise",
                f"ノイズ律速: run-to-run ノイズ {noise:.2e} が数値検出限界 {numerical:.2e} "
                "を支配 → 検証器の分解能は HW 非決定性で決まる（noise_floor 実測が必須）")
    return rep
