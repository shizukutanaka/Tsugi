"""equivalence 検出器のテスト。擬似ベンダーで数値発散を*捕まえる*ことを実証。

これは GPU シミュレーション（CPU）。実 GPU では同じ compare() を実機出力に適用する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.equivalence import (  # noqa: E402
    DV_DIVERGENT,
    DV_EQUIVALENT,
    DV_LAYOUT,
    classify_divergence,
    compare,
    simulate_vendor_matmul,
)


def _inputs(M=128, N=128, K=512):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    return a, b


def test_identical_is_equivalent():
    a, b = _inputs()
    ref = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    rep = compare(ref, ref.copy(), dtype="float16")
    assert rep.equivalent, rep.to_text()


def test_f32_vs_f32_within_tolerance():
    # 同じ f32 累積・分割違いは微差 → 等価のはず
    a, b = _inputs()
    v1 = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    v2 = simulate_vendor_matmul(a, b, accum="f32", split_k=4)
    rep = compare(v1, v2, dtype="float16")
    assert rep.equivalent, f"f32 split should be equivalent: {rep.to_text()}"


def test_f16_accum_detected_as_divergent():
    # fp16 累積は大 K で誤差が膨らむ → DIVERGENT を検出できるべき
    a, b = _inputs(K=2048)
    good = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    bad = simulate_vendor_matmul(a, b, accum="f16", split_k=64)
    rep = compare(good, bad, dtype="float16")
    # 検出器が機能 = ズレを equivalent と誤判定しない
    assert not rep.equivalent, f"divergence not caught: {rep.to_text()}"
    assert rep.max_abs_err > 1e-2


def test_report_fields():
    a, b = _inputs()
    ref = simulate_vendor_matmul(a, b)
    rep = compare(ref, ref.copy(), "float16")
    assert rep.n_mismatch == 0
    assert rep.n_total == ref.size


def test_uniform_risk_interface():
    # EquivalenceReport も他レポートと同じ risk/max_risk/ok を持つ（検証層の統一）
    from tsugi.report import Risk
    a, b = _inputs()
    ref = simulate_vendor_matmul(a, b)
    eq = compare(ref, ref.copy(), "float16")
    assert eq.ok and eq.risk is Risk.OK and eq.max_risk is Risk.OK
    dv = compare(ref, ref + 10.0, "float16")
    assert not dv.ok and dv.risk is Risk.BLOCK


def test_classify_layout_vs_numerical_divergence():
    # element-wise 不一致を レイアウト不一致(値正しい) vs 真の数値発散 に区別
    rng = np.random.default_rng(0)
    K = 256
    a = rng.standard_normal((64, 64)).astype(np.float32)
    assert classify_divergence(a, a.copy(), K) == DV_EQUIVALENT
    # 転置・置換は値の多重集合を保存 → LAYOUT（数値は正しい・整列バグ）
    assert classify_divergence(a, a.T.copy(), K) == DV_LAYOUT
    shuffled = rng.permutation(a.reshape(-1)).reshape(64, 64).astype(np.float32)
    assert classify_divergence(a, shuffled, K) == DV_LAYOUT
    # 真のスケール発散は multiset も崩す → DIVERGENT
    assert classify_divergence(a, (a * 1.5).astype(np.float32), K) == DV_DIVERGENT
    # 要素数が違えば layout ではありえない → DIVERGENT
    assert classify_divergence(a, a[:, :32].copy(), K) == DV_DIVERGENT


def test_float64_does_not_fall_back_to_float32_tolerance():
    """float64 比較が float32 の緩い許容(1e-4)にフォールバックしないこと（偽OK 防止）。

    研究知見（PyTorch assert_close は float64=atol 1e-8）に基づく修正の回帰テスト。
    倍精度で 1e-6 ずれた 2 実装は、float32 許容(1e-4)では「等価」と誤判定されるが、
    float64 専用許容(1e-7)では正しく DIVERGENT を検出すべき。
    """
    base = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    perturbed = base + 1e-6   # float32 tol(1e-4)では隠れる / float64 tol(1e-7)では見える
    rep64 = compare(base, perturbed, dtype="float64")
    assert not rep64.equivalent, (
        f"float64 で 1e-6 のズレを見逃した（偽OK）: {rep64.to_text()}")
    # 対照: 同じズレを float32 許容で見ると（緩いので）等価扱い = フォールバックの危険性
    rep32 = compare(base, perturbed, dtype="float32")
    assert rep32.equivalent, "対照: float32 許容(1e-4)では 1e-6 は等価扱い（だから fallback は危険）"


def test_float64_accepts_genuine_double_precision_noise():
    """float64 で真の倍精度丸め(~1e-15)は等価と判定する（偽BLOCK を出さない）。"""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(100).astype(np.float64)
    noise = base + rng.standard_normal(100) * 1e-14  # 倍精度丸め級
    rep = compare(base, noise, dtype="float64")
    assert rep.equivalent, f"倍精度丸めを過剰検出（偽BLOCK）: {rep.to_text()}"


def test_nan_in_output_flagged_as_non_finite():
    """NaN を含む出力は has_non_finite=True かつ DIVERGENT と判定（データ破壊の識別）。

    精度発散（finite な差）と NaN 伝播（データ破壊）は根本原因が異なる。
    前者はアルゴリズム精度問題、後者は overflow/除零/入力破損。
    has_non_finite フラグでスタックトレースなしに根本原因を絞り込める。
    """
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_nan = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    rep = compare(a, b_nan, dtype="float32")
    assert not rep.equivalent, "NaN を含む出力が等価と誤判定された"
    assert rep.has_non_finite, "NaN が has_non_finite=False のまま（識別できていない）"
    assert "NaN/Inf" in rep.to_text(), f"to_text() に NaN/Inf が表示されない: {rep.to_text()}"

    # 双方に Inf を含む場合も非有限を識別できる
    b_inf = np.array([1.0, np.inf, 3.0], dtype=np.float32)
    rep_inf = compare(a, b_inf, dtype="float32")
    assert rep_inf.has_non_finite
    assert not rep_inf.equivalent

    # 有限同士の差は has_non_finite=False
    rep_finite = compare(a, a + 0.5, dtype="float32")
    assert not rep_finite.has_non_finite, "有限の差で has_non_finite=True（偽陽性）"


def test_tf32_tolerance_matches_float16():
    """TF32 dtype の許容誤差が float16 と同等（10-bit 仮数 → fp16 精度）。

    NVIDIA Ampere+ の float32 GEMM/conv は TF32 Tensor Core を使い仮数が 10 bit（fp16 と同等）。
    AMD ROCm は TF32 非対応 → NVIDIA vs AMD 比較では dtype="tf32" で fp16 級許容が必要。
    """
    from tsugi.equivalence import TOLERANCE
    tol_tf32 = TOLERANCE["tf32"]
    tol_f16 = TOLERANCE["float16"]
    assert tol_tf32["atol"] == tol_f16["atol"], (
        f"TF32 atol={tol_tf32['atol']} ≠ float16 atol={tol_f16['atol']}")
    assert tol_tf32["rtol"] == tol_f16["rtol"]

    # dtype="tf32" で compare が動作するか（KeyError が起きない）
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = a + 5e-3   # float32 許容(1e-4) では DIVERGENT / tf32 許容(1e-2) では EQUIVALENT
    rep_tf32 = compare(a, b, dtype="tf32")
    rep_f32 = compare(a, b, dtype="float32")
    assert rep_tf32.equivalent, f"TF32 許容内(5e-3)を DIVERGENT と誤検出: {rep_tf32.to_text()}"
    assert not rep_f32.equivalent, "float32 許容(1e-4)で 5e-3 差が等価（想定外）"


def test_tf32_unit_roundoff_matches_float16():
    """TF32 の unit_roundoff が float16 と同等（仮数 10 bit, u = 2^-11）。"""
    from tsugi.tolerance import UNIT_ROUNDOFF
    assert UNIT_ROUNDOFF["tf32"] == UNIT_ROUNDOFF["float16"], (
        f"TF32 unit_roundoff={UNIT_ROUNDOFF['tf32']} ≠ float16={UNIT_ROUNDOFF['float16']}")


def main() -> int:
    ok = True
    tests = [
        test_identical_is_equivalent,
        test_f32_vs_f32_within_tolerance,
        test_f16_accum_detected_as_divergent,
        test_report_fields,
        test_uniform_risk_interface,
        test_classify_layout_vs_numerical_divergence,
        test_float64_does_not_fall_back_to_float32_tolerance,
        test_float64_accepts_genuine_double_precision_noise,
        test_nan_in_output_flagged_as_non_finite,
        test_tf32_tolerance_matches_float16,
        test_tf32_unit_roundoff_matches_float16,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: 擬似 NVIDIA(f32) vs 擬似 AMD(f16累積) の照合
    a, b = _inputs(K=2048)
    nv = simulate_vendor_matmul(a, b, accum="f32")
    amd = simulate_vendor_matmul(a, b, accum="f16", split_k=64)
    print("\n--- 擬似ベンダー等価性照合（K=2048）---")
    print("nvidia(f32) vs amd(f16累積):", compare(nv, amd, "float16").to_text())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
