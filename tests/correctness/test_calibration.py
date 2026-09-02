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
    SRC_CROSS_VENDOR,
    calibrate_safety,
    check_systematic,
    detect_shared_mode,
    detectability_floor,
    evaluate,
    is_equivalent_combined,
    make_corpus,
    roc_sweep,
    systematic_divergence,
    systematic_divergence_stderr,
    tolerance_factor_normal,
    wilks_confidence,
    wilks_min_runs,
)
from tsugi.equivalence import compare_gemm, simulate_vendor_matmul  # noqa: E402
from tsugi.report import Risk  # noqa: E402


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


def test_systematic_check_uses_upper_bound_not_point_estimate_for_small_n():
    """check_systematic は小 N で bias 点推定でなく上側限界(bias+stderr)で判定する（第20回）。

    Q55（過不足の続き）: bias は N 要素からの点推定に過ぎない。小テンソルでは、たまたま
    小さい bias が出て真の系統誤差を見逃す（偽OK）可能性がある。rollout.flip_rate_upper_bound
    と同じ fail-safe パターン（点推定でなく推定の不確実性込みの上側限界で判定）を
    systematic_divergence にも適用したことを実証する。

    N=4 の小テンソルで 1 要素だけ 5% 摂動させると、bias 点推定はたまたま極小
    （旧コードなら OK 相当）になるが、ブートストラップ標準誤差が大きく
    上側限界は閾値を大きく超える → 新コードは正しく BLOCK にする。
    """
    rng = np.random.default_rng(2)
    n = 4
    a = rng.standard_normal(n) * 1.0
    b = a.copy()
    b[0] *= 1.05   # 1/4 要素だけ 5% 系統摂動（小 N ゆえ RMS 比への寄与は運次第）

    bias = systematic_divergence(a, b)
    stderr = systematic_divergence_stderr(a, b, n_boot=300, seed=2)
    from tsugi.tolerance import unit_roundoff
    from tsugi.constants import SAFETY
    thresh = SAFETY * unit_roundoff("float16")
    half = 0.5 * thresh

    # 点推定だけなら OK 判定になるはずの状況を固定（旧ロジックの弱点を再現）
    assert abs(bias) < half, f"bias 点推定が既に大きい（この検証ケースが機能しない）: {bias}"
    # だが推定不確実性（stderr）を足すと閾値を大きく超える
    assert abs(bias) + stderr > thresh, "stderr を足しても閾値を超えない（テストケース不成立）"

    rep = check_systematic(a, b, K=1, dtype="float16")
    assert rep.max_risk == Risk.BLOCK, (
        f"小 N の真の系統誤差を上側限界で捕まえられていない（偽OK 復活）: {rep.to_text()}")
    assert rep.bias_upper_bound > thresh


def test_systematic_check_large_n_unaffected_by_upper_bound():
    """大 N（典型的な GEMM 出力）では stderr が無視できるほど小さく、挙動は点推定判定と同じ。"""
    rng = np.random.default_rng(0)
    K = 2048
    a = rng.standard_normal((64, K)).astype(np.float16)
    b = rng.standard_normal((K, 64)).astype(np.float16)
    base = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    bug = base * 1.005
    rep = check_systematic(base, bug, K, "float16")
    # stderr は bias 本体よりずっと小さい（大 N で推定は安定）
    assert rep.bias_stderr < abs(rep.bias) * 0.1
    assert rep.max_risk == Risk.BLOCK  # 従来通り捕まる（回帰なし）


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


def test_detection_verdict_is_seed_independent_at_safety_times_u():
    """検出境界が seed に依らず SAFETY·u に一致する（SOCRATIC-50 Q43・乱数境界の点検）。

    Q43 の懸念は「乱数依存テストは seed 固定でも境界付近では脆い（別 seed で反転しうる）」。
    本テストはそれを *仮定でなく実測* で潰す: 系統バグ強度を理論境界 SAFETY·u の
    ±1% に置き、多数の seed で判定が **全会一致** になることを固定する。
    全会一致であれば、他の固定 seed テストがたまたま通っているのではなく、判定が
    seed 非依存（バグ強度のみに支配される）であることの根拠になる。

    併せて Q6（定数 SAFETY が境界を支配する）を seed 横断に一般化した形でもある。
    """
    from tsugi.constants import SAFETY
    from tsugi.tolerance import unit_roundoff

    K, dtype, n_seeds = 256, "float16", 40
    thresh = SAFETY * unit_roundoff(dtype)     # 理論上の検出境界（fp16 で ~0.195%）

    def equivalent_count(strength):
        c = 0
        for s in range(n_seeds):
            a = np.random.default_rng(s).standard_normal((64, 64)).astype(np.float32)
            c += bool(is_equivalent_combined(a, a * (1 + strength), K, dtype))
        return c

    # 境界のわずか下: 全 seed で「等価」（過剰検出＝偽BLOCK が無い）
    below = equivalent_count(0.99 * thresh)
    assert below == n_seeds, f"境界直下で判定が seed 依存（{below}/{n_seeds} のみ等価）"

    # 境界のわずか上: 全 seed で「非等価」（見逃し＝偽OK が無い）
    above = equivalent_count(1.01 * thresh)
    assert above == 0, f"境界直上で判定が seed 依存（{above}/{n_seeds} が等価と誤判定）"


# --- SAFETY 定数の実機校正（FEATURE-AUDIT A-2） ---

def test_tolerance_factor_matches_published_one_sided_table():
    """Natrella 近似が公表された片側許容係数表を再現する（係数の外部検証）。

    このプロジェクトの設計ガードレールは「未検証の数値係数を導入しない」。
    tolerance_factor_normal は SAFETY の要求値を直接スケールするため、値が
    間違っていれば校正結果ごと間違う。標準的な片側許容限界表（coverage=0.99・
    confidence=0.95）と照合して、実装が既知の値を再現することを固定する。
    """
    table = {10: 3.981, 15: 3.520, 20: 3.295, 25: 3.158, 30: 3.064,
             50: 2.862, 100: 2.684}
    for n, expected in table.items():
        got = tolerance_factor_normal(n, 0.99, 0.95)
        rel = abs(got - expected) / expected
        assert rel < 0.015, f"n={n}: k={got:.4f} vs 表 {expected}（差 {rel:.1%}）"
        # 近似は表より *小さい* 側に外れる（要求 SAFETY を過小 = 許容を締める向き
        # = 偽BLOCK 側）。偽OK 側に外れていないことを明示的に固定する。
        assert got <= expected, f"n={n}: 近似が表より大きい（偽OK 方向の外れ）"

    # n→∞ で coverage の正規分位点に単調収束する（σ が既知なら k=z_p——「4σ」という
    # 素朴な言い方が成立するのはこの極限だけで、有限標本では必ず k>z_p になる）
    from tsugi.nondeterminism import normal_quantile
    ks = [tolerance_factor_normal(n, 0.99, 0.95) for n in (100, 1_000, 10_000, 100_000)]
    assert ks == sorted(ks, reverse=True), f"n について単調減少でない: {ks}"
    assert abs(ks[-1] - normal_quantile(0.99)) < 0.02, f"z_0.99 に収束しない: {ks[-1]}"
    # 標本が少なすぎれば「主張できない」を inf で返す（黙って小さい値で埋めない）
    assert tolerance_factor_normal(2, 0.99, 0.95) == float("inf")


def test_wilks_sample_size_and_confidence_are_consistent():
    """Wilks の必要標本数と達成信頼度が互いの逆算になっている（分布仮定なしの側）。"""
    n = wilks_min_runs(0.99, 0.95)
    assert n == 299, f"ln(0.05)/ln(0.99) = 299 のはずが {n}"
    assert wilks_confidence(n, 0.99) >= 0.95      # ちょうど満たす
    assert wilks_confidence(n - 1, 0.99) < 0.95   # 1 つ足りなければ満たさない
    # 実機で現実的な 16 対では信頼度は 15% 程度しかない（「16 run 回して終わり」の弱さ）
    assert 0.10 < wilks_confidence(16, 0.99) < 0.20


def test_calibrate_safety_flags_safety_that_cannot_cover_measured_noise():
    """実測の良性発散が SAFETY のヘッドルームを超えたら WARN（校正の本体・再現ケース）。

    SAFETY=4.0 は一度も実機ノイズで校正されていない経験値。実機ノイズが理論 1σ の
    4 倍を超えていれば、良性ノイズが発散と誤判定される（偽BLOCK）。ここでは
    「1σ の約 6 倍の良性発散」を人工的に与え、校正が要求値 6σ 超を検出して
    現行 SAFETY では覆えないと言うことを固定する。
    """
    K, dtype, scale = 256, "float16", 1.0
    from tsugi.tolerance import expected_gemm_abs_error
    sigma = expected_gemm_abs_error(K, dtype, scale, safety=1.0)

    # 良性発散が 1σ の ~6 倍（ばらつき小）。SAFETY=4.0 では覆えない。
    big = np.full(32, 6.0 * sigma)
    rep = calibrate_safety(big, K, dtype=dtype, scale=scale)
    assert rep.required > 4.0, f"要求値が 4.0 以下: {rep.required}"
    assert not rep.covers_measured_noise
    assert any(f.risk is Risk.WARN and "覆えていない" in f.message for f in rep.findings), \
        rep.to_text()

    # 回帰なし: 1σ の 0.5 倍しか揺れなければ WARN は出ない（過剰警告しない）
    small = np.full(32, 0.5 * sigma)
    ok = calibrate_safety(small, K, dtype=dtype, scale=scale)
    assert ok.covers_measured_noise, ok.to_text()
    assert not any(f.risk >= Risk.WARN for f in ok.findings), ok.to_text()


def test_calibrate_safety_takes_max_of_normal_theory_and_sample_max():
    """正規理論と標本最大の大きい方を採る（GPU ノイズは i.i.d. ガウスでないため）。

    arXiv:2511.00025 は GPU の浮動小数誤差が独立ガウスでなく構造的・高相関である
    ことを実測で示した。ゆえに正規理論の許容限界 mean+k*sd 単独は信用できない。
    重い外れ値を 1 つ混ぜ、要求値が標本最大を下回らないことを固定する
    （下回れば「実際に観測された良性発散」を許容外と宣言することになり偽BLOCK）。
    """
    from tsugi.tolerance import expected_gemm_abs_error
    sigma = expected_gemm_abs_error(256, "float16", 1.0, safety=1.0)
    d = np.concatenate([np.full(31, 0.1 * sigma), [20.0 * sigma]])  # 裾の重い 1 発
    rep = calibrate_safety(d, 256, scale=1.0)
    assert rep.required >= rep.ratio_max >= 19.9, rep.to_text()
    assert rep.required >= rep.required_normal


def test_run_to_run_calibration_refuses_to_justify_lowering_safety():
    """run-to-run 由来の校正は「SAFETY を下げてよい」と言わない（偽OK 方向の封じ）。

    同一ベンダー内の揺れは縮約順序差だけを含み、クロスベンダー発散（タイル形状・
    行列コア・ライブラリ実装差を含む）の下界にすぎない。これで SAFETY を下げると
    未測定成分を許容から外すことになり偽OK に倒れる。既定 source では常にこの
    但し書きを出し、余裕（下げ代）の提示は cross_vendor 標本のときだけに限る。
    """
    from tsugi.tolerance import expected_gemm_abs_error
    sigma = expected_gemm_abs_error(256, "float16", 1.0, safety=1.0)
    tiny = np.full(32, 0.01 * sigma)   # 極端に静か = 一見「SAFETY を下げられる」

    r2r = calibrate_safety(tiny, 256, scale=1.0)
    assert any("下げる根拠にはならない" in f.message for f in r2r.findings), r2r.to_text()
    assert not any("余裕" in f.message for f in r2r.findings), \
        "run-to-run 標本で下げ代を提示してはならない（偽OK 方向）"

    cross = calibrate_safety(tiny, 256, scale=1.0, source=SRC_CROSS_VENDOR)
    assert any("余裕" in f.message for f in cross.findings), cross.to_text()


def test_calibrate_safety_reports_no_evidence_when_samples_are_empty():
    """標本ゼロを「校正済み」と誤らせない（沈黙でなく WARN + required=inf）。"""
    rep = calibrate_safety(np.zeros(0), 256, scale=1.0)
    assert rep.required == float("inf")
    assert not rep.covers_measured_noise
    assert any(f.risk is Risk.WARN and "標本がゼロ" in f.message for f in rep.findings)


def main() -> int:
    ok = True
    tests = [
        test_floor_grows_with_K,
        test_roc_sweep_combined_catches_above_threshold,
        test_roc_sweep_honest_subthreshold_blindspot,
        test_max_abs_misses_subfloor_scale_bug,
        test_systematic_check_catches_what_max_abs_misses,
        test_systematic_check_uses_upper_bound_not_point_estimate_for_small_n,
        test_systematic_check_large_n_unaffected_by_upper_bound,
        test_systematic_check_passes_legitimate_order_divergence,
        test_systematic_threshold_is_sensitive_to_safety_constant,
        test_combined_verifier_is_trustworthy_corpus,
        test_max_abs_alone_is_untrustworthy_corpus,
        test_shared_mode_failure_is_cross_vendor_blind_spot,
        test_detection_verdict_is_seed_independent_at_safety_times_u,
        test_tolerance_factor_matches_published_one_sided_table,
        test_wilks_sample_size_and_confidence_are_consistent,
        test_calibrate_safety_flags_safety_that_cannot_cover_measured_noise,
        test_calibrate_safety_takes_max_of_normal_theory_and_sample_max,
        test_run_to_run_calibration_refuses_to_justify_lowering_safety,
        test_calibrate_safety_reports_no_evidence_when_samples_are_empty,
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
