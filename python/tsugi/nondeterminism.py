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
- **vLLM batch-invariant モード（`VLLM_BATCH_INVARIANT=1`）は NVIDIA 専用**（Compute
  Capability ≥ 8.0・RMSNorm/matmul/attention をカバー）。**ROCm/AMD は未対応**——
  クロスベンダー比較では「NVIDIA 側だけ batch-invariant・AMD 側は従来通りバッチ変動の
  影響を受ける」という新たな非対称が生じうる。measure_batch_variance は両ベンダーに
  同じ手法で床を実測するため、この非対称自体（片側の床がもう片側よりゼロに近い）も
  検出可能（docs/SOURCES.md 参照）。
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .arrays import asarray

from .report import FindingReport, Risk

EQUIVALENT = "EQUIVALENT"
DIVERGENT = "DIVERGENT"
INDISTINGUISHABLE = "INDISTINGUISHABLE"  # クロス差がノイズ未満＝判定未定義


# --- run-to-run 非決定な演算の静的カタログ（PyTorch 公式由来） ---
# https://pytorch.org/docs/stable/notes/randomness.html
# 大半は GPU で atomicAdd を使い、スレッド到着順で和の順序が変わるため seed 固定でも
# run-to-run で揺れる。一部（cumsum/kthvalue/median）は atomicAdd ではなく並列 scan の
# 縮約順序や重複値の選択順が CUDA 実行スケジュールに依存するために揺れる——機構は違うが
# 「seed 固定でも静的許容だけでは不十分・noise floor 実測が必須」という結論は同じなので
# 同一カタログに含める。reason 文字列は機構を正確に区別する（catalog の判定は
# requires_noise_floor に直結するため、機構を誤記すると診断メッセージが偽情報になる）。
# これは measure_noise_floor の *静的な事前警告* 版: グラフにこれらの op があれば、
# 実行前に「この比較は noise floor 実測なしには信頼できない」と宣言できる。
ATOMIC_NONDET_OPS: dict[str, str] = {
    # forward が atomicAdd を使う（出力そのものが run-to-run で揺れる）
    "scatter_add": "forward atomicAdd（散布加算の到着順）",
    "index_add": "forward atomicAdd（添字加算の到着順）",
    "index_put": "forward atomicAdd（accumulate=True の重複添字）",
    "bincount": "forward atomicAdd（ヒストグラム集計）",
    "scatter_reduce": "forward atomicAdd（reduce='sum' 経路）",
    "index_select_backward": "atomicAdd（gather の逆伝播）",
    "histc": "forward atomicAdd（CUDA ヒストグラム集計・bincount と同機構）",
    "put_": "forward atomicAdd（accumulate=True の重複添字・index_put と同機構）",
    # backward が atomicAdd を使う（勾配が run-to-run で揺れる → 学習で顕著）
    "embedding_bag": "backward atomicAdd（埋め込み勾配の集約）",
    "embedding": "backward atomicAdd（重複添字の勾配集約）",
    "ctc_loss": "backward atomicAdd（系列勾配の集約）",
    "max_pool": "backward atomicAdd（重複入力位置の勾配集約）",
    "adaptive_avg_pool": "backward atomicAdd（プーリング勾配の集約）",
    "grid_sample": "backward atomicAdd（サンプリング勾配の集約）",
    "interpolate": "backward atomicAdd（アップサンプリング勾配の集約）",
    # atomicAdd ではなく CUDA の並列縮約スケジュール依存で run-to-run 揺れる（PyTorch 公式 doc 掲載）
    "cumsum": "非atomic・CUDA 並列 scan の縮約順序（浮動小数の結合律非成立で揺れる）",
    "kthvalue": "非atomic・CUDA 上の重複値の選択順序（同値要素間の tie-break が run ごとに変わる）",
    "median": "非atomic・CUDA 上の重複値の選択順序（indices 付き median の tie-break が揺れる）",
}


def op_is_nondeterministic(op_name: str) -> bool:
    """この op が atomicAdd 由来で本質的に run-to-run 非決定か（PyTorch 公式カタログ照合）。

    op_name は完全一致または前方一致で判定する（"scatter_add_"・"aten.scatter_add.default"・
    "max_pool2d" 等の表記揺れを吸収）。
    """
    name = op_name.lower()
    return any(key in name for key in ATOMIC_NONDET_OPS)


def nondeterminism_reason(op_name: str) -> str | None:
    """非決定 op ならその理由を返す（決定論的なら None）。"""
    name = op_name.lower()
    for key, reason in ATOMIC_NONDET_OPS.items():
        if key in name:
            return reason
    return None


@dataclass
class NondetCatalogReport(FindingReport):
    """グラフ中の非決定 op を静的に列挙する（noise floor 実測の必要性を事前宣言）。"""

    nondet_ops: tuple[str, ...] = ()

    @property
    def requires_noise_floor(self) -> bool:
        """1 つでも非決定 op があれば noise floor 実測が必須（静的許容では不十分）。"""
        return len(self.nondet_ops) > 0

    def to_text(self) -> str:  # type: ignore[override]
        return super().to_text(
            header=(f"nondeterminism catalog "
                    f"({'NOISE-FLOOR REQUIRED' if self.requires_noise_floor else 'deterministic'})"),
            empty="(no atomicAdd-based ops — static tolerance is sufficient)")


def classify_nondeterminism(op_names) -> NondetCatalogReport:
    """グラフの op 名リストを走査し、atomicAdd 由来の非決定 op を静的に列挙する。

    これらの op があれば、derive_tolerance の静的許容だけでは不十分で、
    measure_noise_floor で run-to-run ノイズを実測してから等価判定すべき
    （compare_stable に渡す）。実行前にこの要件を宣言できるのが利点。
    """
    rep = NondetCatalogReport()
    hits = []
    for name in op_names:
        reason = nondeterminism_reason(name)
        if reason is not None:
            hits.append(str(name))
            rep.add(Risk.WARN, str(name),
                    f"{reason} → seed 固定でも run-to-run で揺れる。"
                    "静的許容でなく noise floor 実測（measure_noise_floor）が必須")
    rep.nondet_ops = tuple(hits)
    return rep



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
    return asarray(acc, dtype=np.float32)


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


def collect_runs(run_fn: Callable[[int], np.ndarray], n_runs: int = 16,
                 seed0: int = 0) -> np.ndarray:
    """同一ベンダー・同一入力を n_runs 回走らせ、出力を 1 つのスタックに集める。

    実機では 1 run が高価（GPU 実行）なので、run を消費する検証（ノイズ床の算出・
    SAFETY 校正・比較対象の取得）は **この 1 セットから全部導く**。関数を分けずに
    スタックを共有するのはそのため（同じ run を二度走らせない）。
    """
    return np.stack([asarray(run_fn(seed0 + i), dtype=np.float64)
                     for i in range(n_runs)])


def noise_floor_from_runs(stack: np.ndarray) -> dict[str, float]:
    """既に集めた run スタックからノイズ床統計を出す（再実行しない版）。"""
    s = asarray(stack, dtype=np.float64)
    stats = _spread_stats(s)
    stats["n_runs"] = int(s.shape[0])
    return stats


def pair_deviations(stack: np.ndarray) -> np.ndarray:
    """run スタックから *独立な run 対の差* を標本化する（SAFETY 校正の入力）。

    等価判定が実際に比較するのは「2 つの単発 run の差」`max|a-b|` である。ゆえに
    校正の標本も **対の差** でなければ単位が合わない —— 中心（平均/中央値）からの
    偏差を使うと、比較される量の約半分を測ることになり要求 SAFETY を系統的に
    過小評価する（＝許容を緩く見積もる偽OK 方向の誤り）。

    対は重ならない (0,1),(2,3),… を使う。共通の参照 run（全部 run_0 との差）を
    使うと標本どうしが run_0 を通じて相関し、許容限界の統計（独立標本を前提とする）
    が正当化できないため。返り値は floor(n/2) 個の `max|run_2i - run_2i+1|`。
    """
    s = asarray(stack, dtype=np.float64)
    m = s.shape[0] // 2
    if m == 0:
        return np.zeros(0)
    return np.array([float(np.abs(s[2 * i] - s[2 * i + 1]).max()) for i in range(m)])


def measure_noise_floor(run_fn: Callable[[int], np.ndarray], n_runs: int = 16,
                        seed0: int = 0) -> dict[str, float]:
    """同一ベンダー・同一入力を n_runs 回走らせ run-to-run ノイズを実測する。

    run_fn(seed) -> 出力テンソル。`spread`（max-min・保守的）と `spread_robust`
    （10-90 パーセンタイル幅・外れ値に頑健）の両方を返す。後者は測定グリッチで床が
    過大評価され偽BLOCK 化するのを防ぐ（compare_stable(robust=True) で選択）。
    """
    return noise_floor_from_runs(collect_runs(run_fn, n_runs, seed0))


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
    return asarray(acc, dtype=np.float32)


def measure_batch_variance(run_of_batch: Callable[[int], np.ndarray],
                           batch_tiles=(32, 64, 128, 256, 512)) -> dict[str, float]:
    """同一論理入力をバッチ依存タイル群で走らせ batch-invariance 床を実測する。

    run_of_batch(tile) -> 出力テンソル。バッチ変動で生じる *決定論的* な差の spread を返す。
    run-to-run の atomic ノイズとは独立した床。実機ではバッチサイズ違いの forward を渡す。
    本番でバッチが変動するなら、この床も等価判定に織り込むべき（実効床 = max(run-to-run,
    batch-variance, 数値検出限界)）。
    """
    runs = [asarray(run_of_batch(t), dtype=np.float64) for t in batch_tiles]
    stats = _spread_stats(np.stack([asarray(r) for r in runs]))
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


def runs_to_resolve(cross_diff: float, noise_floor: float,
                    confidence: float = 0.95, max_runs: int = 10 ** 6) -> int:
    """INDISTINGUISHABLE を解消するのに要る 1 ベンダーあたりの run 数を返す（0=解消不要）。

    単発比較でクロス差がノイズに埋もれても、**独立な run を平均すれば平均のノイズは
    σ/√N に縮む**——系統差 d は平均しても縮まないので、N を増やせば SNR = d·√N/σ は
    伸び、いずれ分離できる。必要条件 d > z·σ/√N より **N > (z·σ/d)²**。

    これは DiFR（"Inference Verification Despite Nondeterminism"）が採る枠組みと同型:
    単発では良性の浮動小数ノイズと真の差を区別できないが、**多数トークン/試行に証拠を
    累積すれば SNR が伸び検出できる**（同論文は数千トークンで設定誤りを検出できるとする）。
    `docs/SOURCES.md`「非決定下での検証（証拠の累積）」節を参照。

    本ライブラリでの意義: 従来 INDISTINGUISHABLE は *終端* 状態（「判定未定義」で行き止まり）
    だった。本関数はそれを「あと N run 集めれば決着する」という **実行可能な次手** に変える。
    d=0（完全一致）なら分離すべき差が無いので 0。d>σ なら既に分離済みで 0。

    confidence: 片側信頼水準（既定 0.95 → z≈1.645）。max_runs は非現実的な巨大値の上限。
    """
    d = abs(float(cross_diff))
    sigma = abs(float(noise_floor))
    if d <= 0.0 or sigma <= 0.0 or d > sigma:
        return 0                       # 差が無い／ノイズが無い／既に分離済み
    z = normal_quantile(confidence)   # 片側正規分位点（scipy 非依存）
    n = math.ceil((z * sigma / d) ** 2)
    return int(min(max(n, 1), max_runs))


def _erfinv(y: float) -> float:
    """逆誤差関数の近似（Winitzki）。scipy 非依存（このプロジェクトは numpy のみに依存）。"""
    if y <= -1.0 or y >= 1.0:
        raise ValueError(f"erfinv domain error: {y}")
    if y == 0.0:
        return 0.0
    a = 0.147
    ln1my2 = math.log(1.0 - y * y)
    t1 = 2.0 / (math.pi * a) + ln1my2 / 2.0
    return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln1my2 / a) - t1), y)


def normal_quantile(p: float) -> float:
    """標準正規分布の p 分位点 z_p（scipy 非依存・`_erfinv` の Winitzki 近似に基づく）。

    片側信頼限界・許容限界の係数計算で共通に使う（`runs_to_resolve` と
    `calibration.tolerance_factor_normal` の両方がこれを参照し、分位点の
    実装が二重化しないようにする）。
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"normal_quantile domain error: {p}")
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


@dataclass
class StabilityReport(FindingReport):
    verdict: str = INDISTINGUISHABLE
    noise_floor: float = 0.0
    cross_diff: float = 0.0
    tol: float = 0.0
    numerical_floor: float = 0.0
    runs_needed: int = 0   # INDISTINGUISHABLE を解消するのに要る 1 ベンダーあたり run 数

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

    a = asarray(run_a(0), dtype=np.float64)
    b = asarray(run_b(0), dtype=np.float64)
    scale = float(np.sqrt(np.mean(a ** 2)) + 1e-30)
    numerical = expected_gemm_abs_error(K, dtype, scale)
    tol = derive_tolerance(K, dtype, scale, noise_floor=noise)["atol"]
    cross = float(np.abs(a - b).max())

    rep = StabilityReport(verdict=attribute(cross, noise, tol), noise_floor=noise,
                          cross_diff=cross, tol=tol, numerical_floor=numerical)
    if rep.verdict == INDISTINGUISHABLE:
        # 終端で終わらせず「あと何 run で決着するか」を出す（証拠の累積・DiFR 同型）。
        rep.runs_needed = runs_to_resolve(cross, noise)
        nxt = (f"→ 1 ベンダーあたり約 {rep.runs_needed} run を平均すれば分離可能"
               "（平均のノイズは σ/√N で縮み系統差は縮まない）"
               if rep.runs_needed else "→ クロス差が 0（分離すべき差が無い）")
        rep.add(Risk.WARN, "noise",
                f"クロス差 {cross:.2e} ≤ run-to-run ノイズ {noise:.2e} "
                f"→ ベンダー内ノイズと区別不能・等価判定は未定義 {nxt}")
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
