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

# dtype 別の許容誤差（BENCHMARK.md §4 と整合）。
# 参考: PyTorch `torch.testing.assert_close` の dtype 別デフォルト（同一ベンダー想定）は
#   float16=(rtol=1e-3, atol=1e-3) / float32=(1e-4, 1e-5) / float64=(1e-5, 1e-8)。
# Tsugi は *クロスベンダー*（NVIDIA↔AMD で正当に異なる）を扱うため概ね 1 桁緩めるが、
# fail-safe 哲学（偽OK は致命的）に従い float64 を float32 にフォールバックさせない
# （float64 を 1e-4 で見ると 5 桁緩く偽OK 源になる）。未知 dtype のみ float32 を既定とする。
# TF32: NVIDIA Ampere+ が float32 GEMM/conv に使うハイブリッド形式（10-bit 仮数、fp32 指数）。
# AMD ROCm は TF32 非対応 → NVIDIA(TF32) vs AMD(full fp32) で最大 ~1e-3 誤差が生じうる。
# dtype="tf32" は float32 演算だが TF32 精度で比較したい場合に使う。
TOLERANCE = {
    "float16": dict(atol=1e-2, rtol=1e-2),
    "bfloat16": dict(atol=2e-2, rtol=2e-2),  # bf16 はベンダー差が大きい
    "float32": dict(atol=1e-4, rtol=1e-4),
    "float64": dict(atol=1e-7, rtol=1e-7),   # 倍精度: PyTorch 1e-8 比で 1 桁緩め（cross-vendor）
    "tf32":    dict(atol=1e-2, rtol=1e-2),   # 仮数 10 bit（fp16 と同等）→ fp16 と同じ許容
    # FP8 (OCP OFP8・H100/MI300/B200 推論の主流): 仮数 2〜3 bit で丸めが巨大。
    # クロスベンダーでは per-tensor amax スケールの差も乗る → 大幅に緩い許容が必要。
    "float8_e4m3": dict(atol=1e-1, rtol=1e-1),  # 3 仮数 (u=0.0625)・重み/活性
    "float8_e5m2": dict(atol=2e-1, rtol=2e-1),  # 2 仮数 (u=0.125)・勾配・最も粗い
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
    # NaN/Inf が検出されたかを明示的に追跡する（数値発散とデータ破壊を区別）。
    # NaN は element-wise 比較で必ず mismatch になるが、その原因が「精度差」か
    # 「overflow/NaN 伝播」かを区別しないと根本原因診断ができない。
    has_non_finite: bool = False
    # 形状不一致を明示的に追跡する（NumPy の暗黙 broadcast による偽比較を防ぐ）。
    # a.shape != b.shape は素朴な element-wise 引き算では broadcast され、
    # 偶然 shape が合う（例 (64,1) vs (64,64)）と誤って比較が成立してしまう。
    # 形状不一致自体が構造的なバグの証拠（誤ったテンソルを渡した・カーネルが
    # 間違った形状を返した等）であり、broadcast で握りつぶさず即 DIVERGENT にする。
    shape_mismatch: bool = False
    shape_a: tuple = ()
    shape_b: tuple = ()

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
        if self.shape_mismatch:
            return (f"[SHAPE MISMATCH] a.shape={self.shape_a} vs b.shape={self.shape_b} "
                    "→ 比較不能な構造的発散（broadcast による偽比較を拒否）")
        status = "EQUIVALENT" if self.equivalent else "DIVERGENT"
        nan_tag = " [NaN/Inf検出]" if self.has_non_finite else ""
        return (f"[{status}]{nan_tag} max_abs={self.max_abs_err:.3e} max_rel={self.max_rel_err:.3e} "
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


# element-wise 不一致の原因分類（レイアウト不一致 vs 真の数値発散） ----------
DV_EQUIVALENT = "EQUIVALENT"
DV_LAYOUT = "LAYOUT"
DV_DIVERGENT = "DIVERGENT"


def classify_divergence(a: np.ndarray, b: np.ndarray, K: int,
                        dtype: str = "float16") -> str:
    """element-wise 不一致が *レイアウト不一致*（値は正しいが位置が違う）か
    *真の数値発散* かを区別する。

    cross-vendor カーネルは同じ論理テンソルを異なるレイアウト（row/col-major・タイル順・
    転置）で書きうる。素朴な element-wise 比較は転置-but-equal を巨大発散=BLOCK と誤判定する。
    だがレイアウト不一致は値の *多重集合*（ソート済み全要素）を保存する:
      EQUIVALENT : element-wise 一致
      LAYOUT     : element-wise 不一致 だが multiset 一致 → レイアウトバグ（数値は正しい）
      DIVERGENT  : multiset も不一致 → 真の数値発散
    LAYOUT は transpose/再タイルで修正可能 —— 数値検証の対象でなく codegen の整列問題。
    """
    af = np.asarray(a)
    bf = np.asarray(b)
    if af.shape == bf.shape and compare_gemm(af, bf, K, dtype).equivalent:
        return DV_EQUIVALENT
    fa = af.reshape(-1)
    fb = bf.reshape(-1)
    if fa.shape != fb.shape:
        return DV_DIVERGENT     # 要素数が違えば multiset 不能＝発散扱い
    sa = np.sort(fa.astype(np.float64))
    sb = np.sort(fb.astype(np.float64))
    return DV_LAYOUT if compare_gemm(sa, sb, K, dtype).equivalent else DV_DIVERGENT


def _compare_with(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> EquivalenceReport:
    a_raw = np.asarray(a)
    b_raw = np.asarray(b)
    # 形状不一致を最初に検査する。NumPy の暗黙 broadcast（例 (64,1) と (64,64)・
    # スカラと行列）に頼って比較すると、方向次第で偽 DIVERGENT にも偽 OK にもなりうる
    # （broadcast された片方の値が他方に「たまたま」一致すれば equivalent=True になる）。
    # 形状不一致は誤ったテンソルを渡した／カーネルが間違った形状を返したという
    # 構造的な証拠なので、broadcast で握りつぶさず比較不能な発散として即座に返す。
    if a_raw.shape != b_raw.shape:
        return EquivalenceReport(
            equivalent=False, max_abs_err=float("inf"), max_rel_err=float("inf"),
            n_mismatch=0, n_total=0, atol=atol, rtol=rtol,
            shape_mismatch=True, shape_a=tuple(a_raw.shape), shape_b=tuple(b_raw.shape),
        )
    af = a_raw.astype(np.float64)
    bf = b_raw.astype(np.float64)
    # NaN/Inf を先に検出する（精度発散とデータ破壊を区別するため）。
    # NaN は close[] を常に False にするため mismatch にカウントされるが、
    # has_non_finite=True により診断層が「overflow/NaN 伝播」と識別できる。
    has_non_finite = bool(not (np.all(np.isfinite(af)) and np.all(np.isfinite(bf))))
    abs_err = np.abs(af - bf)
    rel_err = abs_err / (np.abs(af) + 1e-12)
    close = abs_err <= (atol + rtol * np.abs(af))
    n_mismatch = int(np.size(close) - np.count_nonzero(close))
    return EquivalenceReport(
        equivalent=bool(np.all(close)),
        max_abs_err=float(np.nanmax(abs_err)) if abs_err.size else 0.0,
        max_rel_err=float(np.nanmax(rel_err)) if rel_err.size else 0.0,
        n_mismatch=n_mismatch,
        n_total=int(np.size(close)),
        atol=atol,
        rtol=rtol,
        has_non_finite=has_non_finite,
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
