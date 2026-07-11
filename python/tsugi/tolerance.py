"""Tsugi tolerance — 許容誤差を演算の数値条件から *導出* する（ソクラテス問答の新視点）。

盲点: equivalence は固定 1e-2 を使っていた。だが両ベンダーは累積順序が違うだけで
両方 IEEE 正当 — どちらも"真値"でない。許容すべき発散幅は「数学が許す範囲」であり、
累積深さ K・dtype の機械イプシロン・値スケールから導出できる。

モデル（誠実な近似・厳密な最悪ケース境界ではない）:
  GEMM の f16 入力・f32 累積では、支配的な発散は入力量子化（u_fp16）が累積で
  ランダムウォーク的に広がる項。絶対誤差 ~ safety * sqrt(K) * u_input * scale。
  → K が大きいほど許容も大きい（大K GEMM は正当に大きくズレる）。

TF32 (TensorFloat32) について:
  NVIDIA Ampere+ GPU は float32 GEMM/conv を TF32 Tensor Core で実行する（デフォルト）。
  TF32 は fp32 の指数部（8 bit）と fp16 の仮数部（10 bit）を組み合わせたハイブリッド形式
  → 精度は fp16 と同等（u_TF32 = 2^-11）だが動的範囲は fp32 と同等。
  torch.backends.cuda.matmul.allow_tf32 は PyTorch 1.12 以降デフォルト False だが、
  torch.backends.cudnn.allow_tf32（conv 側）はデフォルト True。
  AMD ROCm は TF32 をサポートしないため、クロスベンダー比較では float32 入出力でも
  TF32 由来の 1e-3 級誤差が生じうる → dtype="tf32" で明示的に緩い許容を使う。

  PyTorch 2.9 での API 変更: allow_tf32（bool）は非推奨となり
  torch.backends.cuda.matmul.fp32_precision = 'ieee' | 'tf32'（文字列）に移行した
  （set_float32_matmul_precision('highest')→ieee／'high'・'medium'→tf32 に対応）。
  **FlexAttention のデフォルト精度がリリース間で ieee→tf32 に回帰した実例がある**
  （pytorch#161022 とされる）——同じコードでも PyTorch のバージョンが変わるだけで
  数値が変わりうる実在の事故。provenance（どの PyTorch バージョンで測定したか）の
  stale 検出が必要な根拠になる（docs/SOURCES.md「torch.compile / Triton の数値精度 API」節）。

FP8 (OCP OFP8: E4M3 / E5M2) について:
  H100/MI300/B200 世代の推論で主流（vLLM ネイティブ・キャリブレーション不要）。
  E4M3=4 指数 3 仮数（max 448・無限大なし・NaN のみ）、E5M2=5 指数 2 仮数（max 57344・inf/nan あり）。
  仮数が 2〜3 bit しかなく丸め誤差が巨大（u=0.0625〜0.125）。だがクロスベンダーの真のリスクは
  **per-tensor の amax スケーリング係数**: FP8 は値域が狭いため amax で正規化してから量子化する。
  全 GPU で共通スケールを使うと量子化誤差が増えるため各々が個別スケールを持つが、amax の縮約順序が
  ベンダー間で違うとスケール自体がずれ、テンソル全体が系統的にシフトする（calibration の系統検査が効く）。

Microscaling (OCP MX v1.0: MXFP4/MXFP6/MXFP8) について（2025-26 外部調査ベース）:
  block=32 要素ごとに 1 個の共有スケール（E8M0・2^-127〜2^127・power-of-two のみ）を持つ
  低精度形式群。NVIDIA Blackwell と AMD CDNA4（MI350/MI355）の**両方が HW ネイティブ対応する
  唯一の共通低精度フォーマット**（NVFP4 は NVIDIA 専用・後述）。
    MXFP4 = 要素 E2M1（仮数 1 bit・max=6.0・min_normal=1.0）→ u=2^-1（全 dtype 中最粗）。
    MXFP6 = E2M3（仮数 3 bit・max=7.5）または E3M2（仮数 2 bit・max=28）。
  ここでの UNIT_ROUNDOFF は**要素型**の相対丸め誤差であり、block スケール（E8M0）は
  block 単位の動的レンジを決めるだけで要素内の相対精度は変えない（scale は乗法的に
  キャンセルされるため compare() の相対誤差には現れない）。ただし 1 block=32 要素に
  スケール 1 個しかないため、block 内に outlier が 1 つあると他の 31 要素が丸め潰される
  リスクは envelope 層（channel_scale_spread 等）で別途捉えるべき——ここの UNIT_ROUNDOFF は
  その効果を含まない「量子化グリッドの粗さ」のみのモデル。
  丸めモードは MX spec 上は実装定義（RNE/確率的丸め）——勾配側は確率的丸め、重み/活性側は
  RNE が慣行（NVFP4/MXFP4 学習論文, 2025）。RNE か SR かの実装差もクロスベンダー発散源になりうる。

  NVFP4（NVIDIA Blackwell 専用・AMD 非対応）: 要素は MXFP4 と同じ E2M1 だが block=16・
  スケールは E4M3（power-of-two に限らずより細かい block 内配置が可能）。**AMD に対応する
  HW が存在しないため、NVFP4 で量子化したモデルはそもそもクロスベンダー移植の対象外**——
  これは数値許容の問題でなく dtype 選定自体の移植性判断であり、Tsugi の
  「クロスベンダー共通フォーマットのみ」という UNIT_ROUNDOFF/TOLERANCE/DTYPE_LIMITS の
  対象範囲から意図的に除外する。
"""
from __future__ import annotations

import math

from .constants import SAFETY

# 単位丸め誤差（unit roundoff, u = 2^-(mantissa_bits+1)）
UNIT_ROUNDOFF = {
    "float16": 2.0 ** -11,   # 10 仮数ビット → u ≈ 4.88e-4
    "bfloat16": 2.0 ** -8,   # 7 仮数ビット → u ≈ 3.91e-3（fp16 より粗い）
    "float32": 2.0 ** -24,   # 23 仮数ビット → u ≈ 5.96e-8
    "float64": 2.0 ** -53,   # 52 仮数ビット → u ≈ 1.11e-16（倍精度・oracle と同精度）
    # TF32: NVIDIA Ampere+ の fp32 GEMM/conv で使われるハイブリッド形式。
    # 仮数部 10 bit（fp16 と同じ）→ u は fp16 と同値。
    # AMD ROCm は TF32 非対応 → NVIDIA vs AMD で float32 計算に最大 1e-3 誤差。
    "tf32": 2.0 ** -11,      # 10 仮数ビット（fp16 と同等）→ u ≈ 4.88e-4
    # FP8 (OCP OFP8 仕様・H100/MI300/B200 の推論で主流): 仮数が極端に少なく u が大きい。
    # クロスベンダーの主リスクは丸めでなく **per-tensor amax スケーリング係数の差**:
    # 各ベンダーが amax を別の縮約順序で計算するとスケールがずれ、テンソル全体が系統シフトする。
    "float8_e4m3": 2.0 ** -4,   # 3 仮数ビット → u = 0.0625（重み/活性・前向き）
    "float8_e5m2": 2.0 ** -3,   # 2 仮数ビット → u = 0.125（勾配・後ろ向き・さらに粗い）
    # Microscaling (OCP MX v1.0)。NVIDIA Blackwell / AMD CDNA4 の両方が HW ネイティブ対応する
    # 唯一の共通低精度フォーマット群（NVFP4 は NVIDIA 専用のためここに含めない）。
    # u は要素型の相対丸め誤差（block スケール E8M0 は動的レンジのみを決める・上記 docstring 参照）。
    "mxfp4_e2m1": 2.0 ** -1,   # 1 仮数ビット → u = 0.5（全 dtype 中最粗）
    "mxfp6_e2m3": 2.0 ** -3,   # 3 仮数ビット → u = 0.125
    "mxfp6_e3m2": 2.0 ** -2,   # 2 仮数ビット → u = 0.25
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
