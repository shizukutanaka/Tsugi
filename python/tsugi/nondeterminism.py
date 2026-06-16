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

最新研究の取り込み（2025）:
- **バッチ不変性（batch invariance）が LLM 推論の *支配的* 非決定源**であり、atomic 並行性の
  浮動小数非結合性ではない（Thinking Machines Lab 2025・SC'24 arXiv:2408.05148）。あるサンプルの
  出力は forward の *バッチサイズ* に依存する —— バッチ依存のタイル/縮約順序が丸めを変えるため。
  これは run-to-run の atomic ノイズとは別の **決定論的だがバッチ変動で生じる第三の床**。
  GPU 固有でなく CPU/TPU でも生じる。クロスベンダーでは「タイルが違う＝実効バッチが違う」ため、
  各ベンダーが個別に決定論的でも発散しうる。→ measure_batch_variance で実測する。
- **浮動小数ノイズは独立ガウスでなく構造的（相関）**（arXiv:2511.00025）。これは calibration の
  系統（RMS 比）検出が必要十分でなく *必要* である根拠を外部から裏づける（max_abs 単独では
  相関誤差を見逃す）。
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


def _spread_stats(stack: np.ndarray) -> dict[str, float]:
    """run スタックから spread（max-min）と robust spread（10-90 パーセンタイル幅）を出す。

    max-min は保守的だが外れ値 1 個（測定グリッチ等）で過大評価される。robust 版は
    上下 10% を切り捨て、単発の外れ値に頑健（SOCRATIC Q49）。要素ごとの最大を取る。
    """
    spread = float((stack.max(axis=0) - stack.min(axis=0)).max())
    robust = float((np.percentile(stack, 90, axis=0)
                    - np.percentile(stack, 10, axis=0)).max())
    mean_mag = float(np.sqrt(np.mean(stack ** 2)) + 1e-30)
    return {"spread": spread, "spread_robust": robust,
            "std": float(stack.std(axis=0).max()), "rel": spread / mean_mag,
            "rel_robust": robust / mean_mag}


def measure_noise_floor(run_fn: Callable[[int], np.ndarray], n_runs: int = 16,
                        seed0: int = 0) -> dict[str, float]:
    """同一ベンダー・同一入力を n_runs 回走らせ run-to-run ノイズを実測する。

    run_fn(seed) -> 出力テンソル。`spread`（max-min・保守的）と `spread_robust`
    （10-90 パーセンタイル幅・外れ値に頑健）の両方を返す。後者は測定グリッチで床が
    過大評価され偽BLOCK 化するのを防ぐ（compare_stable(robust=True) で選択）。
    """
    runs = [np.asarray(run_fn(seed0 + i), dtype=np.float64) for i in range(n_runs)]
    stats = _spread_stats(np.stack(runs))
    stats["n_runs"] = n_runs
    return stats


def simulate_batch_variant_reduction(parts: np.ndarray, tile: int) -> np.ndarray:
    """バッチ依存タイルでの縮約を擬似再現する（CPU・シミュレーション・明示）。

    バッチ不変性の機構: バッチサイズが変わると縮約のタイル/分割が変わり、部分和の
    丸め順序が変わる → 同じ論理入力でも *決定論的に* 異なる結果になる（Thinking
    Machines 2025）。tile（= バッチ依存の分割幅）ごとに部分和を fp32 で作り合算する。
    seed は不要（run-to-run の atomic ノイズと違い、バッチが同じなら毎回同じ値）。
    """
    flat = parts.reshape(-1).astype(np.float32)
    acc = np.float32(0.0)
    for i in range(0, flat.size, max(1, tile)):
        chunk = np.float32(0.0)
        for v in flat[i:i + tile]:
            chunk = np.float32(chunk + v)
        acc = np.float32(acc + chunk)
    return np.asarray(acc, dtype=np.float32)


def measure_batch_variance(run_of_batch: Callable[[int], np.ndarray],
                           batch_tiles=(32, 64, 128, 256, 512)) -> dict[str, float]:
    """同一論理入力をバッチ依存タイル群で走らせ batch-invariance 床を実測する。

    run_of_batch(tile) -> 出力テンソル。バッチ変動で生じる *決定論的* な差の spread を返す。
    run-to-run の atomic ノイズとは独立した床。実機ではバッチサイズ違いの forward を渡す。
    本番でバッチが変動するなら、この床も等価判定に織り込むべき（実効床 = max(run-to-run,
    batch-variance, 数値検出限界)）。
    """
    runs = [np.asarray(run_of_batch(t), dtype=np.float64) for t in batch_tiles]
    stats = _spread_stats(np.stack(runs))
    stats["n_batches"] = len(batch_tiles)
    return stats


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
                   dtype: str = "float16", n_runs: int = 16,
                   batch_floor: float = 0.0, robust: bool = False) -> StabilityReport:
    """方法論的に健全なクロスベンダー比較（出力を分布として扱う）。

    1) 各ベンダーの run-to-run ノイズを実測 → noise_floor
    2) 実効床 = max(run-to-run, batch-invariance 床) ——後者は measure_batch_variance で
       実測して batch_floor に渡す（2025 研究: バッチ変動が支配的非決定源）
    3) noise を織り込んだ許容を導出（決定論仮定 noise=0 を排す）
    4) クロス差を noise/tol に対し 3 状態へ帰属（INDISTINGUISHABLE を正直に出す）

    robust=True で run-to-run 床に外れ値頑健な robust spread（10-90 パーセンタイル幅）を
    使う。測定グリッチ 1 個で床が過大評価され偽BLOCK 化するのを防ぐ（SOCRATIC Q49）。
    """
    from .tolerance import derive_tolerance, expected_gemm_abs_error

    nf_a = measure_noise_floor(run_a, n_runs)
    nf_b = measure_noise_floor(run_b, n_runs)
    key = "spread_robust" if robust else "spread"
    run_to_run = max(nf_a[key], nf_b[key])
    noise = max(run_to_run, batch_floor)

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
    if batch_floor > run_to_run:
        rep.add(Risk.WARN, "batch",
                f"バッチ不変性律速: batch-invariance 床 {batch_floor:.2e} が run-to-run "
                f"{run_to_run:.2e} を支配 → 本番バッチ変動が主因（バッチ不変カーネルが要る）")
    return rep
