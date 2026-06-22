"""検証器そのものの検証テスト（偽OK の非対称コストと検出限界）。

許容ベース等価判定の検出限界の下に隠れる系統バグ（偽OK）を実証し、
scale/K 不変な相補計量（RMS 比）がそれを捕まえることを確かめる。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tsugi.calibration import (  # noqa: E402
    SM_DIVERGENT,
    SM_OK,
    SM_SHARED,
    check_systematic,
    detect_shared_mode,
    detectability_floor,
    evaluate,
    is_equivalent_combined,
    make_corpus,
    roc_sweep,
    systematic_divergence,
)
from tsugi.equivalence import compare_gemm, simulate_vendor_matmul  # noqa: E402


def test_floor_grows_with_K():
    # 検出限界 = safety·√K·u は K とともに拡大する（偽OK 盲点が広がる）
    f256 = detectability_floor(256, "float16")["rel"]
    f8192 = detectability_floor(8192, "float16")["rel"]
    assert f8192 > f256 * 4  # √(8192/256)=√32≈5.7 倍


def test_max_abs_misses_subfloor_scale_bug():
    # 0.5% 系統スケール誤差は max_abs（導出許容）では偽OK になる
    rng = np.random.default_rng(0)
    K = 2048
    a = rng.standard_normal((64, K)).astype(np.float16)
    b = rng.standard_normal((K, 64)).astype(np.float16)
    base = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    bug = base * 1.005
    assert compare_gemm(base, bug, K, "float16").equivalent  # 見逃す（偽OK）


def test_systematic_check_catches_what_max_abs_misses():
    rng = np.random.default_rng(0)
    K = 2048
    a = rng.standard_normal((64, K)).astype(np.float16)
    b = rng.standard_normal((K, 64)).astype(np.float16)
    base = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    bug = base * 1.005
    rep = check_systematic(base, bug, K, "float16")
    assert not rep.ok                     # 系統検査は捕まえる
    assert abs(rep.bias - 0.005) < 1e-4   # RMS 比が 0.5% を正しく測る


def test_systematic_check_passes_legitimate_order_divergence():
    # 累積順序だけ違う等価ベンダーは系統バイアス ~0 → 通過（偽BLOCK にしない）
    rng = np.random.default_rng(1)
    K = 2048
    a = rng.standard_normal((64, K)).astype(np.float16)
    b = rng.standard_normal((K, 64)).astype(np.float16)
    va = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    vb = simulate_vendor_matmul(a, b, accum="f32", split_k=8)
    assert check_systematic(va, vb, K, "float16").ok
    assert abs(systematic_divergence(va, vb)) < 1e-3


def test_systematic_threshold_is_sensitive_to_safety_constant():
    # Q6: 定数 SAFETY が判定境界を *実際に* 支配することを境界±で固定する。
    # 閾値 thresh=SAFETY·u 直上で BLOCK、直下で WAR(>0.5·thresh)、0.5·thresh 直下で OK。
    # SAFETY を変えれば閾値も動く（thresh を定数から算出して bias を構成）ので、
    # この境界反転が壊れたら定数の意味が変わったと分かる（silent drift の番人）。
    from tsugi.constants import SAFETY
    from tsugi.report import Risk
    from tsugi.tolerance import unit_roundoff

    a = np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32)
    thresh = SAFETY * unit_roundoff("float16")

    def risk_at(bias: float) -> Risk:
        b = (a.astype(np.float64) * (1.0 + bias)).astype(np.float32)
        return check_systematic(a, b, K=2048, dtype="float16").max_risk

    # 検出限界（K=2048 で ~8.8%）の遥か下でも、境界は定数どおりに反転する
    assert risk_at(thresh * 1.01) == Risk.BLOCK     # 直上 → fail-safe で BLOCK
    assert risk_at(thresh * 0.99) == Risk.WARN      # 直下 → WARN 帯
    assert risk_at(thresh * 0.60) == Risk.WARN      # 0.5·thresh より上 → WARN
    assert risk_at(thresh * 0.40) == Risk.OK        # 0.5·thresh 未満 → OK


def test_combined_verifier_is_trustworthy_corpus():
    # 合成判定（max_abs + 系統）はコーパスで偽OK ゼロ＝信頼に足る
    corpus = make_corpus(seed=0)
    conf = evaluate(corpus, lambda a, b, K: is_equivalent_combined(a, b, K, "float16"))
    assert conf.trustworthy            # false_ok == 0
    assert conf.false_ok == 0


def test_max_abs_alone_is_untrustworthy_corpus():
    # max_abs 単独はコーパスで偽OK を出す＝信頼できない（検証器の検証）
    corpus = make_corpus(seed=0)
    conf = evaluate(corpus, lambda a, b, K: compare_gemm(a, b, K, "float16").equivalent)
    assert not conf.trustworthy
    assert "scale" in conf.missed      # 系統スケールバグを見逃す


def test_shared_mode_failure_is_cross_vendor_blind_spot():
    # 共有モード障害: 両ベンダーが同じバグ→A≈B(cross-vendorは緑)だが両方 oracle と不一致。
    # oracle 照合でのみ検出可能（cross-vendor 一致は必要十分でない）。
    rng = np.random.default_rng(0)
    K = 256
    oracle = rng.standard_normal((64, 64)).astype(np.float32)
    a = (oracle * 1.05).astype(np.float32)
    b = (oracle * 1.05 + 1e-9).astype(np.float32)
    # cross-vendor 単独は等価と誤判定（盲点）
    assert is_equivalent_combined(a, b, K, "float16")
    # oracle 照合は共有モードを暴く
    assert detect_shared_mode(a, b, oracle, K) == SM_SHARED
    # 通常の発散と全一致は正しく分類
    assert detect_shared_mode((oracle * 1.5).astype(np.float32), oracle, oracle, K) == SM_DIVERGENT
    assert detect_shared_mode(oracle, oracle.copy(), oracle, K) == SM_OK


def test_roc_sweep_combined_catches_above_threshold():
    # 強度掃引: 合成判定は系統閾値超で偽OK=0、max_abs 単独は広範囲で偽OK（ROC 化）
    rows = roc_sweep(strengths=(0.005, 0.05), K=2048, seeds=10)
    for row in rows:
        assert row["false_ok_combined"] == 0.0     # 0.5%/5% は合成判定が全捕捉
        assert row["false_ok_max_abs"] > 0.0       # max_abs は一様スケールを吸収し見逃す


def test_roc_sweep_honest_subthreshold_blindspot():
    # 系統閾値(~safety·u≈0.2%)未満は合成判定でも見逃す（正直な残存盲点）
    rows = roc_sweep(strengths=(0.001,), K=2048, seeds=10)
    assert rows[0]["false_ok_combined"] > 0.0


def main() -> int:
    ok = True
    tests = [
        test_floor_grows_with_K,
        test_roc_sweep_combined_catches_above_threshold,
        test_roc_sweep_honest_subthreshold_blindspot,
        test_max_abs_misses_subfloor_scale_bug,
        test_systematic_check_catches_what_max_abs_misses,
        test_systematic_check_passes_legitimate_order_divergence,
        test_systematic_threshold_is_sensitive_to_safety_constant,
        test_combined_verifier_is_trustworthy_corpus,
        test_max_abs_alone_is_untrustworthy_corpus,
        test_shared_mode_failure_is_cross_vendor_blind_spot,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: 検出限界が K で拡大する様子と相補計量の効果
    print("\n--- max_abs 検出限界 vs 系統検査（0.5% スケールバグ）---")
    corpus = make_corpus(seed=0)
    c_alone = evaluate(corpus, lambda a, b, K: compare_gemm(a, b, K, "float16").equivalent)
    c_comb = evaluate(corpus, lambda a, b, K: is_equivalent_combined(a, b, K, "float16"))
    print("  max_abs 単独 : " + c_alone.to_text().replace("\n", "\n  "))
    print("  合成(相補)   : " + c_comb.to_text().replace("\n", "\n  "))
    for K in (256, 2048, 8192):
        print(f"  floor@K={K:<5} = {detectability_floor(K, 'float16')['rel'] * 100:.1f}%")
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
