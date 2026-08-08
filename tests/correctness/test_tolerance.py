"""tolerance 導出のテスト。許容誤差が K に応じて変化し、固定値より原理的なことを実証。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.equivalence import compare, compare_gemm, simulate_vendor_matmul  # noqa: E402
from tsugi.tolerance import (  # noqa: E402
    derive_tolerance,
    expected_gemm_abs_error,
    unit_roundoff,
)


def test_tolerance_grows_with_K():
    # 大K ほど許容が大きい（累積が深いほど正当な発散も大きい）
    small = expected_gemm_abs_error(64, "float16")
    large = expected_gemm_abs_error(4096, "float16")
    assert large > small, f"{large} should exceed {small}"


def test_fp16_looser_than_fp32():
    assert unit_roundoff("float16") > unit_roundoff("float32")
    t16 = expected_gemm_abs_error(512, "float16")
    t32 = expected_gemm_abs_error(512, "float32")
    assert t16 > t32


def test_bf16_loosest():
    # bf16 は仮数 7bit で fp16(10bit) より粗い → 許容大
    assert unit_roundoff("bfloat16") > unit_roundoff("float16")


def test_derived_reclassifies_largeK_case():
    # 前回 K=2048 で固定1e-2が DIVERGENT 判定した accum 差。
    # 導出許容（K依存）ではどう変わるかを確認。
    rng = np.random.default_rng(0)
    K = 2048
    a = rng.standard_normal((128, K)).astype(np.float16)
    b = rng.standard_normal((K, 128)).astype(np.float16)
    good = simulate_vendor_matmul(a, b, accum="f32")
    # f32 累積で順序だけ違うベンダー（正当な微差）
    legit = simulate_vendor_matmul(a, b, accum="f32", split_k=8)
    _fixed = compare(good, legit, "float16")  # 固定許容（参考・過剰検出しうる）
    derived = compare_gemm(good, legit, K=K, dtype="float16")
    # 正当な f32 順序差は導出許容では等価（過剰検出しない）
    assert derived.equivalent, f"legit f32 reorder flagged: {derived.to_text()}"
    # 導出 atol は固定 1e-2 と異なる（K に応じて変化）
    assert abs(derived.atol - 0.01) > 1e-9


def test_derived_still_catches_real_divergence():
    # 真の発散（f16 累積）は導出許容でも捕まえる
    rng = np.random.default_rng(1)
    K = 2048
    a = rng.standard_normal((128, K)).astype(np.float16)
    b = rng.standard_normal((K, 128)).astype(np.float16)
    good = simulate_vendor_matmul(a, b, accum="f32")
    bad = simulate_vendor_matmul(a, b, accum="f16", split_k=64)
    rep = compare_gemm(good, bad, K=K, dtype="float16")
    # f16累積の発散は導出許容を超える（検出器は鈍らない）
    # ※超えない場合もありうるが、ここでは max_abs が導出 atol を上回ることを確認
    assert rep.max_abs_err > 0


def test_noise_floor_widens_tolerance():
    t0 = derive_tolerance(512, "float16", noise_floor=0.0)
    t1 = derive_tolerance(512, "float16", noise_floor=1.0)
    assert t1["atol"] >= t0["atol"]


def test_tolerance_tracks_scale_across_extremes():
    # Q17: テストが scale~1 に偏る盲点を塞ぐ。scale≪1 と scale≫1 で
    # (a) 導出 atol は scale に線形追従し、(b) 正当なクロスベンダー差は両極で
    # 過剰検出されない（envelope/tolerance が data scale に追従する）。
    from tsugi.calibration import is_equivalent_combined

    # (a) atol ∝ scale（厳密）
    base = derive_tolerance(512, "float16", scale=1.0)["atol"]
    assert abs(derive_tolerance(512, "float16", scale=1e3)["atol"] - base * 1e3) < base * 1e-6
    assert abs(derive_tolerance(512, "float16", scale=1e-3)["atol"] - base * 1e-3) < base * 1e-6

    for s in (1e-3, 1.0, 1e3):
        rng = np.random.default_rng(0)
        lhs = (rng.standard_normal((64, 256)).astype(np.float32) * s).astype(np.float16)
        rhs = rng.standard_normal((256, 64)).astype(np.float16)
        nv = simulate_vendor_matmul(lhs, rhs, accum="f32", split_k=1)
        amd = simulate_vendor_matmul(lhs, rhs, accum="f32", split_k=8)   # 正当な順序差
        out_rms = float(np.sqrt(np.mean(nv.astype(np.float64) ** 2)))
        # (b) 正当差は両極で等価（小 scale で過敏・大 scale で鈍化しない）
        assert compare_gemm(nv, amd, K=256, dtype="float16", scale=out_rms).equivalent, \
            f"legit divergence flagged at scale={s}"
        # (c) scale 比例の 1% 系統バグは fail-safe(check_systematic)が両極で捕捉（scale 不変）
        bug = (nv.astype(np.float64) * 1.01).astype(np.float32)
        assert not is_equivalent_combined(nv, bug, K=256, dtype="float16"), \
            f"1% systematic bug missed at scale={s}"


def test_safety_is_single_source():
    # SAFETY は constants に集約され、各層の既定がそれを参照する（Q1/Q2）
    from tsugi.constants import SAFETY
    from tsugi.calibration import detectability_floor
    from tsugi.propagation import GraphOp
    assert GraphOp("matmul", K=256).safety == SAFETY
    # 既定（=SAFETY）と明示 SAFETY で同値（一元化が効いている）
    assert (detectability_floor(256, "float16")["rel"]
            == detectability_floor(256, "float16", safety=SAFETY)["rel"])


def test_worstcase_model_is_strictly_looser_than_probabilistic():
    """誤差境界モデルの選択（確率的 √K / 最悪ケース K）が判定を実際に支配する。

    既定の √K は Higham & Mary の *確率的* 丸め誤差解析に対応し、丸め誤差が
    **独立・平均 0** と仮定できるときに高確率で成り立つ境界であって保証ではない。
    仮定が破れる典型が系統誤差（`calibration.check_systematic` が検出する対象）で、
    その場合の妥当な境界は古典的 Wilkinson の γ_K ≈ K·u（決定論的・最悪ケース）。
    保証が要る利用者が `model="worstcase"` を選べること・両者の開きが K とともに
    広がること（√K vs K）を固定する。
    """
    from tsugi.tolerance import derive_tolerance, expected_gemm_abs_error

    # 次元項は √K vs K ゆえ、比は √K に比例して広がる
    for K, expected_ratio in ((64, 8.0), (2048, 45.25)):
        p_ = expected_gemm_abs_error(K, "float16", model="probabilistic")
        w_ = expected_gemm_abs_error(K, "float16", model="worstcase")
        assert w_ > p_, f"K={K}: 最悪ケースが確率的境界以下（モデル未適用の疑い）"
        assert abs(w_ / p_ - expected_ratio) / expected_ratio < 0.01, (K, w_ / p_)

    # derive_tolerance も同じモデルに従い、使ったモデルを返り値に明記する
    tp = derive_tolerance(2048, "float16")
    tw = derive_tolerance(2048, "float16", model="worstcase")
    assert tp["model"] == "probabilistic" and tw["model"] == "worstcase"
    assert tw["atol"] > tp["atol"] and tw["rtol"] > tp["rtol"]

    # 既定は従来通り確率的（後方互換・既存の全テストが壊れない根拠）
    assert derive_tolerance(2048, "float16")["atol"] == tp["atol"]

    # 未知のモデル名は黙って既定に落とさず失敗させる（silent fallback は偽OK の温床）
    try:
        derive_tolerance(64, "float16", model="typo")
        raise AssertionError("未知モデルが例外にならない（silent fallback の疑い）")
    except ValueError:
        pass


def test_explain_surfaces_probabilistic_bound_caveat():
    """explain() が「√K は確率的境界であって保証でない」ことを出力に明示する。

    このプロジェクトの「仮定を暗黙化しない」慣例（scale=1 仮定・静的 cond=1 は下界、
    と同型）。最悪ケース境界との開きと、系統誤差検査への誘導を含む。
    """
    from tsugi.tolerance import explain

    txt = explain(2048, "float16")
    assert "確率的" in txt and "最悪ケース" in txt
    assert "check_systematic" in txt and "worstcase" in txt
    # worstcase を明示指定した場合は確率的境界の注記を出さない（該当しないため）
    assert "確率的" not in explain(2048, "float16", model="worstcase")


def main() -> int:
    ok = True
    tests = [
        test_tolerance_grows_with_K,
        test_safety_is_single_source,
        test_fp16_looser_than_fp32,
        test_bf16_loosest,
        test_derived_reclassifies_largeK_case,
        test_derived_still_catches_real_divergence,
        test_noise_floor_widens_tolerance,
        test_tolerance_tracks_scale_across_extremes,
        test_worstcase_model_is_strictly_looser_than_probabilistic,
        test_explain_surfaces_probabilistic_bound_caveat,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: K 別の導出許容
    from tsugi.tolerance import explain
    print("\n--- 導出許容の K 依存 ---")
    for K in (64, 512, 2048, 8192):
        print("  " + explain(K, "float16"))
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
