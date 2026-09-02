"""タスクレベル等価性のテスト（判断は数値でなく決定で測る）。

数値発散と判断フリップが decouple すること（スケール不変）、フリップが低マージン
裾に集中すること、フリップ率が P(margin<2δ) の上界に収まることを実証。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tsugi.decision import (  # noqa: E402
    binary_flip_rate,
    binary_margin,
    compare_decisions,
    compare_task,
    decision_flips,
    decompose_divergence,
    divergence_rms,
    flip_bound_support,
    flip_bound_support_from_divergence,
    flip_rate,
    margin,
    nucleus_flip_rate,
    predicted_flip_bound,
    ranking_flip_rate,
    regression_flip_rate,
    residual_divergence_rms,
    sampling_divergence,
    sampling_epsilon,
    tie_rate,
    topk_flip_rate,
    tv_bound,
    tv_bound_from_divergence,
)


def _logits(seed=0, n=4000, c=200):
    return np.random.default_rng(seed).standard_normal((n, c)).astype(np.float32)


def _vendor(z, eps, seed):
    return z + eps * np.random.default_rng(seed).standard_normal(z.shape).astype(np.float32)


def test_margin_is_top1_minus_top2():
    x = np.array([[1.0, 5.0, 3.0]])
    assert abs(float(margin(x)[0]) - 2.0) < 1e-9   # 5 - 3


def test_tie_rate_flags_convention_dependent_decisions():
    # 量子化 logit は同点多発 → argmax は規約依存（tie-break）。tie_rate がそれを露出。
    rng = np.random.default_rng(0)
    quant = np.round(rng.standard_normal((4000, 50)) * 3).astype(np.float32)  # 整数=同点多発
    assert tie_rate(quant) > 0.1                      # 同点が顕著
    cont = rng.standard_normal((4000, 50)).astype(np.float32)                 # 連続値=同点ほぼ無
    assert tie_rate(cont) < 0.01
    # compare_decisions は同点率が高いと警告を立てる（誤帰属の注意喚起）
    rep = compare_decisions(quant, quant.copy())
    assert rep.tie_rate > 0.1
    assert any("tie-break" in f.message or "同点" in f.message for f in rep.findings)


def test_topk_flip_rate_generalizes_argmax():
    # 生成タスク向け: top-k 候補集合の一致。k=1 は argmax フリップ率に一致、k で単調増加、
    # スケール不変、同一入力で 0。
    z = _logits()
    a, b = _vendor(z, 3e-2, 1), _vendor(z, 3e-2, 2)
    assert abs(topk_flip_rate(a, b, 1) - flip_rate(a, b)) < 1e-12
    assert topk_flip_rate(a, b, 1) <= topk_flip_rate(a, b, 5) <= topk_flip_rate(a, b, 10)
    assert topk_flip_rate(a, b, 5) == topk_flip_rate(a * 7.0, b * 7.0, 5)  # スケール不変
    assert topk_flip_rate(a, a, 5) == 0.0
    # k がクラス数を超えても安全（全集合一致 → flip 0 ではなく valid な率）
    assert 0.0 <= topk_flip_rate(a, b, 9999) <= 1.0


def test_decision_flips_are_scale_invariant():
    # logit を 10 倍すれば abs 誤差も 10 倍だが判断(argmax)は不変→フリップ率同一
    z = _logits()
    a, b = _vendor(z, 1e-2, 1), _vendor(z, 1e-2, 2)
    r1 = flip_rate(a, b)
    r10 = flip_rate(a * 10.0, b * 10.0)
    assert abs(r1 - r10) < 1e-12          # 数値 abs 誤差と完全に decouple
    assert np.abs(a * 10 - b * 10).max() > np.abs(a - b).max()  # abs 誤差は10倍


def test_flips_concentrate_in_low_margin_tail():
    z = _logits()
    a, b = _vendor(z, 1e-2, 1), _vendor(z, 1e-2, 2)
    f = decision_flips(a, b)
    m = margin(a)
    assert f.any()
    # フリップしたサンプルのマージンは全体中央値よりずっと小さい(near-tie)
    assert np.median(m[f]) < 0.3 * np.median(m)


def test_near_tie_threshold_is_sensitive_to_its_constant():
    """SOCRATIC-50 Q5: 定数 `_NEAR_TIE_MARGIN_FRAC` が判定境界を *実際に* 支配することを
    境界±で固定する（Q6 の `SAFETY` 感度テストと同型・silent drift の番人）。

    多数派（999件・margin=M_large・フリップ無し）と少数派（1件・margin=m_flip・
    フリップ有り）を混ぜると、overall_margin_median は多数派に支配されて M_large に
    固定され、flipped_margin_median は少数派 1 件の値 m_flip そのものになる。
    m_flip を `_NEAR_TIE_MARGIN_FRAC・M_large` の直上/直下に置くことで、
    「フリップが near-tie 裾に集中していない」警告の境界を狙って再現できる。
    """
    from tsugi.decision import _NEAR_TIE_MARGIN_FRAC

    def _build(n_large, m_large, m_flip):
        a = np.zeros((n_large + 1, 2), dtype=np.float64)
        a[:n_large] = [m_large, 0.0]
        a[n_large] = [m_flip, 0.0]
        b = a.copy()
        b[n_large] = [0.0, m_flip]   # このサンプルだけ argmax が反転する
        return a.astype(np.float32), b.astype(np.float32)

    def not_concentrated_warned(mult: float) -> bool:
        a, b = _build(999, 10.0, _NEAR_TIE_MARGIN_FRAC * 10.0 * mult)
        rep = compare_decisions(a, b, flip_budget=1.0)   # 巨大予算で near-tie 警告だけを見る
        return any("集中していない" in f.message for f in rep.findings)

    assert not_concentrated_warned(1.01), "閾値直上で「near-tie に集中していない」警告が出ない"
    assert not not_concentrated_warned(0.99), "閾値直下で警告が誤って出ている"


def test_predicted_bound_is_upper_bound():
    z = _logits()
    for eps in (1e-3, 1e-2, 1e-1):
        a, b = _vendor(z, eps, 1), _vendor(z, eps, 2)
        from tsugi.decision import divergence_rms
        actual = flip_rate(a, b)
        bound = predicted_flip_bound(a, divergence_rms(a, b))
        assert actual <= bound + 1e-9     # P(margin<2δ) は実フリップ率の上界


def test_predicted_bound_uses_wilson_upper_bound_for_small_representative_set():
    """predicted_flip_bound は点推定 k/n でなく Wilson 上側限界で判定する（第21回）。

    P(margin<2δ) は代表 logit（ref_logits）n 件からの *点推定* に過ぎない。第20回の
    calibration.check_systematic と同型の盲点: n が小さい代表集合ではたまたま
    margin<2δ に該当するサンプルが 0 件でも、母集団の真の確率は 0 ではない
    （rollout.flip_rate_upper_bound の rule-of-three と同じ問題）。0 件観測を
    「フリップ率 0%」と過信するのは fail-safe に反する。
    """
    rng = np.random.default_rng(0)
    n_small = 20
    z_small = rng.standard_normal((n_small, 50)).astype(np.float32) * 5.0  # 大マージン中心
    delta = 0.01
    m = margin(z_small)
    k = int(np.count_nonzero(m < 2.0 * delta))
    assert k == 0, "この検証ケースは 0 件観測が前提（テストケース不成立）"

    point_estimate = k / n_small   # 旧ロジック相当（0.0 になるはず）
    bound = predicted_flip_bound(z_small, delta)
    assert point_estimate == 0.0
    assert bound > point_estimate, (
        f"0 件観測を過信して bound=0 のまま（偽OK 復活）: bound={bound}")
    assert bound > 0.05, f"小標本の不確実性を反映した上側限界になっていない: bound={bound}"

    # n が大きければ Wilson 上限は点推定にほぼ収束する（回帰なし）
    n_large = 20000
    z_large = np.random.default_rng(1).standard_normal((n_large, 50)).astype(np.float32) * 5.0
    m_large = margin(z_large)
    k_large = int(np.count_nonzero(m_large < 2.0 * delta))
    bound_large = predicted_flip_bound(z_large, delta)
    if k_large > 0:
        assert abs(bound_large - k_large / n_large) < 0.01, (
            "大標本で Wilson 上限が点推定から大きく乖離（回帰の疑い）")


def test_numerical_divergence_not_sufficient_for_task_divergence():
    # マージンが大きい(確信)モデルは、相当な数値発散(max_abs~0.37)でもタスク影響は無視可能
    z = _logits() * 100.0   # logit を増幅 → マージン大
    a, b = _vendor(z, 5e-2, 1), _vendor(z, 5e-2, 2)
    assert np.abs(a - b).max() > 0.1      # 数値 abs 発散は相当大きい
    assert flip_rate(a, b) < 0.005        # だがタスク(判断)影響は無視可能
    assert compare_decisions(a, b, flip_budget=0.01).ok  # タスク予算内で等価


def test_nucleus_flip_rate_is_probability_dependent():
    # top-p(nucleus)集合フリップ。生成向け。同一→0、valid [0,1]、そして argmax/top-k と違い
    # *スケール不変でない*（温度＝logit スケールで nucleus が伸縮する）。
    z = _logits(n=2000, c=200)
    b = _vendor(z, 5e-2, 3)
    assert nucleus_flip_rate(z, z, 0.9) == 0.0
    assert 0.0 <= nucleus_flip_rate(z, b, 0.9) <= 1.0
    # スケール（温度）で結果が変わる＝確率依存（argmax/top-k 集合は不変だった）
    assert nucleus_flip_rate(z, b, 0.9) != nucleus_flip_rate(z * 5.0, b * 5.0, 0.9)


def test_compare_decisions_reports_topk():
    # compare_decisions(topk=k) は top-k 集合フリップ率も併記（生成タスク向け統合）
    z = _logits()
    b = _vendor(z, 5e-2, 3)
    r1 = compare_decisions(z, b)
    assert r1.topk == 1 and abs(r1.topk_flip_rate - r1.flip_rate) < 1e-12
    r5 = compare_decisions(z, b, topk=5)
    assert r5.topk == 5
    assert r5.topk_flip_rate >= r5.flip_rate          # 集合は argmax より緩く変化
    assert "top-5 set flip" in r5.to_text()
    # top_p 統合: nucleus フリップ率も併記
    rp = compare_decisions(z, b, top_p=0.9)
    assert rp.top_p == 0.9 and rp.nucleus_flip_rate > 0.0
    assert "nucleus(p=0.9)" in rp.to_text()


def test_compare_decisions_blocks_high_flip_rate():
    z = _logits()
    a, b = _vendor(z, 1e-1, 1), _vendor(z, 1e-1, 2)  # 多数フリップ
    rep = compare_decisions(a, b, flip_budget=0.001)
    assert not rep.ok                     # 予算超で BLOCK


def test_compare_decisions_uses_flip_rate_ub_for_small_batch():
    """compare_decisions は観測 flip_rate（点推定）でなく flip_rate_ub（Wilson 上側限界）で
    予算判定する（第22回・第21回 predicted_flip_bound と同型の修正を主判定にも適用）。

    小さい評価バッチ（n=30）でたまたま観測フリップが 0 件でも、母集団の真のフリップ率が
    予算を超えている可能性は排除できない。旧ロジック（点推定のみ）なら「予算内」と
    誤判定するケースで、flip_rate_ub を使う新ロジックが正しく WARN/BLOCK に倒すことを実証。
    """
    from tsugi.report import Risk as _Risk
    rng = np.random.default_rng(5)
    n = 30
    z = rng.standard_normal((n, 20)).astype(np.float32)
    a = _vendor(z, 3e-2, 12)   # seed=6 の組が「観測フリップ 0 件だが真の率は高い」を再現する
    b = _vendor(z, 3e-2, 13)
    budget = 0.03

    assert flip_rate(a, b) == 0.0, "観測フリップ 0 件が前提（テストケース不成立）"
    rep = compare_decisions(a, b, flip_budget=budget)
    assert rep.flip_rate_ub > budget, (
        f"小標本の不確実性が上側限界に反映されていない: flip_rate_ub={rep.flip_rate_ub}")
    assert rep.max_risk >= _Risk.WARN, (
        f"観測フリップ 0 件を過信して予算内(OK)のまま（偽OK 復活）: {rep.to_text()}")

    # n が大きければ flip_rate_ub は点推定に近づき（相対差 <20%）、既存挙動（多数フリップで
    # BLOCK）は不変。フリップ率自体は 0 でないので絶対誤差でなく相対誤差で比較する。
    z_large = _logits(n=4000, c=200)
    a_large, b_large = _vendor(z_large, 1e-1, 1), _vendor(z_large, 1e-1, 2)
    rep_large = compare_decisions(a_large, b_large, flip_budget=0.001)
    assert rep_large.flip_rate_ub / rep_large.flip_rate < 1.2
    assert not rep_large.ok


def test_systematic_affine_divergence_does_not_flip():
    # argmax 保存的な系統発散(スケール×1.5・一様シフト)は数値大でもフリップ 0・残差 ~0
    z = _logits(n=3000, c=300)
    for b in (z * 1.5, z + 0.5 * np.random.default_rng(9).standard_normal((z.shape[0], 1)).astype(np.float32)):
        b = b.astype(np.float32)
        assert flip_rate(z, b) == 0.0
        d = decompose_divergence(z, b)
        assert d["total"] > 0.1                 # 数値的には大きい
        assert d["residual"] < 1e-3             # だが argmax を動かす残差は ~0
        assert d["systematic_frac"] > 0.99
        rep = compare_decisions(z, b, flip_budget=0.001)
        assert rep.ok                           # タスク等価


def test_residual_bound_tighter_than_total_for_systematic():
    # 系統成分を含む混合発散: 残差ベース bound は total ベースよりずっと小さく、かつ上界
    z = _logits(n=4000, c=400)
    shift = 0.4 * np.random.default_rng(3).standard_normal((z.shape[0], 1)).astype(np.float32)
    rand = 0.05 * np.random.default_rng(4).standard_normal(z.shape).astype(np.float32)
    b = (z * 1.3 + shift + rand).astype(np.float32)
    from tsugi.decision import divergence_rms
    total_bound = predicted_flip_bound(z, divergence_rms(z, b))
    resid_bound = predicted_flip_bound(z, residual_divergence_rms(z, b))
    actual = flip_rate(z, b)
    assert resid_bound < total_bound * 0.5      # 系統発散の過大評価を排す
    assert actual <= resid_bound + 1e-9          # 残差 bound も上界として成立


def test_regression_flip_rate_matches_value_closeness():
    # 新視点11: 回帰タスクの判断は「値が近いか」。rtol 以内は flip しない。
    rng = np.random.default_rng(0)
    a = rng.standard_normal(2000).astype(np.float32)
    # 全サンプルで |a-b| = 0.1 * |a|（ちょうど rtol=0.1 の境界）
    b_on = a * (1.0 + 1e-9)           # < rtol=0.1 → no flip
    b_over = a * 1.2                   # 20% 差 > rtol=0.1 → 全 flip
    assert regression_flip_rate(a, b_on, rtol=0.1) == 0.0
    assert regression_flip_rate(a, b_over, rtol=0.1) == 1.0
    # atol 絶対値で clamp: 小値は atol に守られる
    tiny = np.full(100, 1e-6, dtype=np.float64)
    assert regression_flip_rate(tiny, tiny + 1e-7, atol=1e-5, rtol=0.0) == 0.0


def test_binary_flip_rate_detects_threshold_crossing():
    # バイナリ分類: sigmoid 出力が threshold 0.5 を跨ぐ率。
    # 大きなマージン（>0.3）では発散があっても threshold 跨がない。near-tie で跨ぐ。
    a = np.array([0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1])
    b = a + np.array([-0.05, -0.1, -0.2, 0.2, 0.2, 0.1, 0.05])
    fr = binary_flip_rate(a, b)
    # threshold 跨ぎ: 0.4 + 0.2 = 0.6 (positive), 0.3+0.1=0.4→0.3 (ここだけ-側 to -側=same)
    # index 4: 0.4 → 0.6 (跨ぎ: neg→pos = flip)  index 5: 0.3→0.4 (neg→neg, no flip)
    assert 0.0 < fr < 1.0
    # 全部同値なら flip 0
    assert binary_flip_rate(a, a.copy()) == 0.0
    # 全部 threshold 反対側なら flip 1
    high = np.full(5, 0.9)
    low = np.full(5, 0.1)
    assert binary_flip_rate(high, low) == 1.0
    # margin: threshold から遠いほど大きい
    m = binary_margin(np.array([0.9, 0.5, 0.1]))
    assert m[0] > m[1] and m[2] > m[1]


def test_ranking_flip_rate_measures_top_k_set_change():
    # 上位 k 集合が変わるサンプル率。argmax フリップと topk_flip_rate の listwise 版。
    rng = np.random.default_rng(1)
    docs = 100
    scores_a = rng.standard_normal(docs)
    # 極めて小さな摂動: top-10 集合は変わらない（境界付近の値が無ければ順位不変）
    scores_b_close = scores_a + 1e-9 * rng.standard_normal(docs)
    # 大きな摂動（独立乱数）: top-10 は壊れやすい
    scores_b_far = rng.standard_normal(docs)
    assert ranking_flip_rate(scores_a, scores_b_close, k=10) == 0.0  # ランキング不変
    assert ranking_flip_rate(scores_a, scores_b_far, k=10) == 1.0    # ほぼ確実に壊れる
    # 完全同一は 0
    assert ranking_flip_rate(scores_a, scores_a.copy(), k=10) == 0.0
    # バッチ形式
    A = np.vstack([scores_a] * 10)
    B = np.vstack([scores_b_far] * 10)
    assert 0.0 < ranking_flip_rate(A, B, k=10) <= 1.0


def test_compare_task_regression_blocks_large_value_drift():
    # compare_task は回帰フリップ率が予算を超えれば BLOCK する
    rng = np.random.default_rng(2)
    a = rng.standard_normal(1000).astype(np.float32)
    b = a + 5.0 * rng.standard_normal(1000).astype(np.float32)
    rep = compare_task(a, b, task="regression", flip_budget=0.05, rtol=0.01)
    assert rep.flip_rate > 0.0
    assert rep.max_risk.value >= 2   # WARN or BLOCK
    # flip_budget=0.01（非ゼロ・他のテストと同じ規模）: n=1000 で観測フリップ 0 件なら
    # Wilson 上限（≈0.27%）も budget を十分下回り OK。flip_budget=0.0（「未来永劫ゼロ
    # フリップ」の意）は fail-safe の下では有限標本から証明不能なので使わない
    # （compare_decisions と同型の判断・第25回相当の修正）。
    assert compare_task(a, a.copy(), task="regression", flip_budget=0.01, rtol=0.01).max_risk.value == 0


def test_compare_task_uses_flip_rate_ub_for_small_batch():
    """compare_task(task="regression") は観測 flip_rate（点推定）でなく flip_rate_ub
    （Wilson 上側限界）で予算判定する（A-1 の修正・compare_decisions と同型）。

    小さい評価バッチ（n=30）で全要素が許容内（観測フリップ 0 件）でも、母集団の真の
    フリップ率が予算を超えている可能性は排除できない。0 件観測を「フリップ率 0%」と
    過信するのは偽OK の温床（rule of three）。
    """
    rng = np.random.default_rng(0)
    n = 30
    a = rng.standard_normal(n)
    b = a.copy()   # 全要素が許容内 → 観測フリップは厳密に 0 件
    budget = 0.03  # 3%（n=30 では rule-of-three 的に上限が budget を遥かに超える）

    rep = compare_task(a, b, task="regression", flip_budget=budget, rtol=0.01)
    assert rep.flip_rate == 0.0, "観測フリップ 0 件が前提（テストケース不成立）"
    assert rep.flip_rate_ub > budget, (
        f"小標本の不確実性が上側限界に反映されていない: flip_rate_ub={rep.flip_rate_ub}")
    assert rep.max_risk.value >= 2, (
        f"観測フリップ 0 件を過信して予算内(OK)のまま（偽OK 復活）: {rep.to_text()}")

    # n が大きければ flip_rate_ub は点推定に近づき、既存挙動（多数フリップで BLOCK）は不変
    rng2 = np.random.default_rng(2)
    a_large = rng2.standard_normal(1000).astype(np.float32)
    b_large = a_large + 5.0 * rng2.standard_normal(1000).astype(np.float32)
    rep_large = compare_task(a_large, b_large, task="regression", flip_budget=0.05, rtol=0.01)
    assert rep_large.flip_rate > 0.0
    assert rep_large.flip_rate_ub / rep_large.flip_rate < 1.2
    assert rep_large.max_risk.value >= 2


def test_compare_task_ranking_single_query_no_wilson_widening():
    """ranking タスクの 1D 入力（単一クエリ）は決定的な結果ゆえ Wilson widening を
    適用しない（flip_rate_ub == flip_rate）。複数試行から推定した比率ではなく、
    1 回の厳密な集合比較結果だから信頼区間を付けるのは統計的に無意味。

    2D バッチ入力（複数クエリ）はクエリ数を試行数として widening が適用されることも
    併せて確認する（決定的でなく複数試行の平均比率だから）。
    """
    rng = np.random.default_rng(1)
    docs = 100
    scores_a = rng.standard_normal(docs)
    scores_b_close = scores_a + 1e-9 * rng.standard_normal(docs)   # top-k 不変
    scores_b_far = rng.standard_normal(docs)                        # top-k ほぼ確実に変わる

    rep_close = compare_task(scores_a, scores_b_close, task="ranking", k=10)
    assert rep_close.flip_rate == 0.0
    assert rep_close.flip_rate_ub == rep_close.flip_rate, "1D ranking で widening が適用されている"

    rep_far = compare_task(scores_a, scores_b_far, task="ranking", k=10, flip_budget=0.0)
    assert rep_far.flip_rate == 1.0
    assert rep_far.flip_rate_ub == 1.0
    assert rep_far.max_risk.value >= 2

    # 2D バッチ入力（クエリ 5 件）はクエリ数を試行数として widening が働く
    A = np.vstack([scores_a] * 5)
    B = np.vstack([scores_b_close] * 5)
    rep_batch = compare_task(A, B, task="ranking", k=10)
    assert rep_batch.flip_rate == 0.0
    assert rep_batch.flip_rate_ub > 0.0, "2D ranking で widening が働いていない（n_trials 誤り）"


def test_compare_task_binary_ok_for_large_margin():
    # binary タスクの大マージン出力はベンダー差があっても flip しない
    a = np.array([0.95, 0.05, 0.9, 0.1] * 100)
    b = a + 0.02 * np.random.default_rng(3).standard_normal(len(a))
    rep = compare_task(a, b, task="binary", flip_budget=0.01)
    assert rep.flip_rate == 0.0
    assert rep.max_risk.value == 0   # OK


def test_compare_task_binary_warns_when_flips_not_near_tie():
    """compare_task(task="binary") が compare_decisions と同型の near-tie 健全性
    チェックを持つ（binary_margin は実装・テスト済みだったがこの診断には未接続だった）。

    フリップは決定境界近傍（低マージン）に集中するはず。確信領域（高マージン）まで
    巻き込むフリップは系統的発散の兆候であり、near-tie の裾だけがフリップする
    正常系と区別して WARN すべき。
    """
    # near-tie のみがフリップ（正常系）: WARN が出ない
    a_neartie = np.concatenate([np.full(500, 0.501), np.full(500, 0.95)])
    b_neartie = a_neartie.copy()
    b_neartie[:500] = 0.499   # 閾値付近の半分だけが跨ぐ
    rep_ok = compare_task(a_neartie, b_neartie, task="binary", flip_budget=0.6)
    assert not any("near-tie" in f.message for f in rep_ok.findings)

    # 確信領域までフリップ（異常系）: WARN が出る
    a_confident = a_neartie.copy()
    b_confident = a_confident.copy()
    b_confident[500:] = 0.05   # 確信領域（0.95）が 0.05 まで動く
    rep_warn = compare_task(a_confident, b_confident, task="binary", flip_budget=0.6)
    assert any("near-tie" in f.message for f in rep_warn.findings), (
        f"確信領域までのフリップを見逃した: {rep_warn.to_text()}")
    assert rep_warn.flipped_margin_median > rep_warn.overall_margin_median


def test_compare_task_unknown_raises():
    # 未知タスク種別は ValueError を投げる（静かに誤計算しない）
    try:
        compare_task(np.ones(10), np.ones(10), task="segmentation")
        raise AssertionError("should raise")
    except ValueError:
        pass


def test_flip_bound_from_divergence_bridges_propagation_to_task():
    # propagation の相対発散 → タスクフリップ率上界。第2ベンダー無しで予測でき、
    # 同じ δ で実際に摂動したフリップ率の上界になっている（保守的）。
    from tsugi.decision import flip_bound_from_divergence
    z = _logits(n=5000, c=500)
    rel = 0.03
    bound = flip_bound_from_divergence(z, rel)
    scale = float(np.sqrt(np.mean(z.astype(np.float64) ** 2)))
    b = z + rel * scale * np.random.default_rng(7).standard_normal(z.shape).astype(np.float32)
    actual = flip_rate(z, b)
    assert 0.0 < bound <= 1.0
    assert actual <= bound + 1e-9         # 上界として成立


def test_flip_bound_from_divergence_does_not_underestimate_high_scale_outlier():
    """SOCRATIC-50 Q19: 低スケールサンプルが多数を占めるバッチに紛れた高スケール・
    近接マージンの少数サンプルを、グローバル RMS だけの δ では見逃す（偽OK方向）。
    per-sample scale との max を取る修正版はこれを正しく検出する。
    """
    from tsugi.decision import flip_bound_from_divergence
    n_small = 5000
    # 低スケール・確信度高（自身のスケールに対しては margin が十分大きい）サンプル群
    small = np.stack([np.full(n_small, 0.05), np.full(n_small, 0.03)], axis=-1)
    # 高スケール・near-tie な少数サンプル（margin=0.5 が自身のスケール~49.75 に対しては小さい）
    big = np.array([[50.0, 49.5]])
    z = np.concatenate([small, big], axis=0).astype(np.float32)
    rel = 0.01

    # 旧実装相当（グローバル RMS のみで一律 δ）: outlier が margin<2δ にカウントされない
    # （Wilson 上側限界は k=0 でも rule-of-three で >0 を返すため bound==0 では検査できない。
    #  「見逃し」の機構は k=0、ユーザーに見える効果は bound の過小、の両方を固定する）
    global_scale = float(np.sqrt(np.mean(z.astype(np.float64) ** 2)))
    m = margin(z)
    k_global_only = int(np.count_nonzero(m < 2.0 * rel * global_scale))
    assert k_global_only == 0, (
        f"再現前提が崩れている: グローバル scale 版が既に outlier を捕捉 (k={k_global_only})")

    # 修正後の flip_bound_from_divergence は per-sample scale との max を取り、
    # 高スケール outlier を margin<2δ にカウントする → bound が厳密に大きくなる
    bound_global_only = predicted_flip_bound(z, rel * global_scale)
    bound_fixed = flip_bound_from_divergence(z, rel)
    assert bound_fixed > bound_global_only, (
        f"高スケール near-tie outlier のフリップ risk を過小評価した（偽OK・Q19 未解消）: "
        f"fixed={bound_fixed} ≤ global-only={bound_global_only}")


# --- A-9: 温度サンプリング下の分布一致（TV 距離の橋） ---

def _softmax_np(z, T=1.0):
    e = np.exp((z - z.max(-1, keepdims=True)) / T)
    return e / e.sum(-1, keepdims=True)


def _measured_tv(a, b, T):
    return 0.5 * np.abs(_softmax_np(a, T) - _softmax_np(b, T)).sum(-1)


def test_tv_bound_is_upper_bound_across_temperature_and_eps():
    """tanh(ε/T) が実測 TV を上界する（大域的・一次近似でない）。tanh/2 では破れる。

    ‖b‖∞ ≤ ε のとき ‖softmax(z/T)−softmax((z+b)/T)‖_TV ≤ tanh(ε/T)。証明は確率比が
    [e^{−2ε/T}, e^{2ε/T}] に収まることから閉形式 (e^{2ε/T}−1)/(e^{2ε/T}+1) を得る
    （docs/SOURCES.md）。指示書は出発点として `|p_a−p_b|₁ ≤ (2/T)·δ` 型の係数を挙げていたが、
    **係数は数値実験で確かめてから入れる**（A-5 の先例）——ここで tanh/2 型が実際に
    破れる（＝偽OK になる）ことを固定し、係数 1 の tanh が正しいことを外部検証する。
    """
    half_violated = False
    for V in (2, 5, 50, 500):
        for T in (0.1, 0.5, 1.0, 2.0):
            for eps in (0.01, 0.1, 1.0, 3.0):
                for seed in range(3):
                    r = np.random.default_rng(seed * 97 + V)
                    z = r.standard_normal(V) * r.choice([0.5, 2.0, 8.0])
                    b = r.choice([-eps, eps], size=V)      # ∞ノルムちょうど eps
                    tv = float(_measured_tv(z[None], (z + b)[None], T)[0])
                    bound = float(tv_bound(eps, T))
                    assert tv <= bound + 1e-9, \
                        f"V={V} T={T} eps={eps}: 実測 {tv:.4f} > 上界 {bound:.4f}"
                    if tv > bound / 2 + 1e-9:
                        half_violated = True
    assert half_violated, "tanh/2 型の係数が破れる例が無い（係数の外部検証が効いていない）"
    # 極限の挙動: T→∞ で 0（分布が一様に潰れる）・ε→0 で 0
    assert tv_bound(1.0, 1e6) < 1e-5 and tv_bound(0.0, 1.0) == 0.0


def test_sampling_epsilon_is_shift_invariant_but_scale_sensitive():
    """サンプリングの ε は shift 不変・scale 非不変（既存 2 量がどちらも壊れることの固定）。

    softmax は shift 不変だが scale 非不変（一様スケールは温度変化そのもの）。argmax は
    両方に不変。この非対称のため:
      - residual_divergence_rms（argmax 用・アフィン成分を除去）は純 scale で ≈0 →
        実際は TV≠0 なので **偽OK**
      - divergence_rms（total）は純 shift で大きな値 → 実際は TV=0 なので **偽BLOCK**
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((500, 32)) * 2.0

    shifted = a + 3.0
    scaled = 1.1 * a
    # 純 shift: TV は厳密に 0、ε も 0（total を使うと 3.0 になり偽BLOCK）
    assert float(_measured_tv(a, shifted, 1.0).max()) < 1e-12
    assert float(sampling_epsilon(a, shifted).max()) < 1e-9
    assert divergence_rms(a, shifted) > 2.9                    # total は大きい＝使えない
    # 純 scale: TV は明確に非ゼロ、ε も非ゼロ（residual を使うと ≈0 になり偽OK）
    assert float(_measured_tv(a, scaled, 1.0).mean()) > 0.01
    assert float(sampling_epsilon(a, scaled).min()) > 0.01
    assert residual_divergence_rms(a, scaled) < 1e-9           # residual は ≈0＝偽OK

    # scale+shift の合成では shift 成分だけが落ちる（scale 成分は残る）
    both = 1.1 * a + 3.0
    assert np.allclose(sampling_epsilon(a, both), sampling_epsilon(a, scaled))
    # ε は上界の入力として妥当（per-sample で実測 TV を上回る）
    tv = _measured_tv(a, both, 1.0)
    assert bool((tv <= tv_bound(sampling_epsilon(a, both), 1.0) + 1e-12).all())


def test_sampling_flip_rate_converges_to_argmax_flip_rate_as_temperature_falls():
    """T→0 で TV 平均が argmax フリップ率に一致する（層の連続性・受け入れ基準）。

    TV は最適結合の下で「両ベンダーから引いた 1 サンプルが食い違う確率」。T→0 では
    分布が argmax の点質量に潰れるので、その確率はちょうど argmax フリップ率になる。
    ゆえに **サンプリング層は decision 層の厳密な一般化**（貪欲は T→0 の特例）。
    """
    z = _logits(seed=0, n=4000, c=32) * 2.0
    a, b = _vendor(z, 0.05, 1), _vendor(z, 0.05, 2)
    greedy = flip_rate(a, b)
    assert greedy > 0.005, "フリップが起きない設定では連続性を検証できない"
    low_t = compare_task(a, b, task="sampling", temperature=0.01).flip_rate
    assert abs(low_t - greedy) <= 0.1 * greedy, \
        f"T→0 で argmax フリップ率に収束しない: TV={low_t:.4f} vs argmax={greedy:.4f}"
    # 温度を上げると分布が滑らかになり食い違い確率は単調に下がる
    temps = [compare_task(a, b, task="sampling", temperature=T).flip_rate
             for T in (0.01, 0.5, 1.0, 4.0)]
    assert temps == sorted(temps, reverse=True), f"温度について単調でない: {temps}"


def test_tv_bound_reports_itself_vacuous_at_low_temperature():
    """低温では worst-case 上界が無情報になることを自ら申告する（偽BLOCK 対策）。

    tanh(ε/T) は T→0 で 1 に飽和し「TV ≤ 0.98」のような無意味な上界になる（実測は桁違いに
    小さい）。fail-safe だが額面通り BLOCK 判定に使えば偽BLOCK を量産するため、判定は実測
    TV で行い、上界が使えないときは INFO で明示する。閾値 _TV_BOUND_VACUOUS が実際に
    その境界を支配することも固定する（Q5 の定数感度テストと同型）。
    """
    from tsugi.decision import _TV_BOUND_VACUOUS
    z = _logits(seed=0, n=500, c=32) * 2.0
    a, b = _vendor(z, 0.05, 1), _vendor(z, 0.05, 2)

    cold = compare_task(a, b, task="sampling", temperature=0.02)
    assert cold.tv_predicted > _TV_BOUND_VACUOUS
    assert any("無情報" in f.message for f in cold.findings), cold.to_text()
    # 上界が無情報でも判定自体は実測 TV で行うので過剰 BLOCK にならない
    assert cold.flip_rate < 0.5 and cold.tv_predicted > 0.9

    warm = compare_task(a, b, task="sampling", temperature=8.0)
    assert warm.tv_predicted <= _TV_BOUND_VACUOUS
    assert not any("無情報" in f.message for f in warm.findings), warm.to_text()


def test_compare_task_sampling_blocks_high_distribution_divergence():
    """分布差が予算を超えたら BLOCK し、小標本では Wilson 上側限界で判定する。"""
    from tsugi.report import Risk as _Risk
    z = _logits(seed=0, n=2000, c=32) * 2.0
    big = compare_task(z, _vendor(z, 1.0, 3), task="sampling", temperature=1.0,
                       flip_budget=0.001)
    assert big.max_risk is _Risk.BLOCK and big.flip_rate > 0.01
    # 同一 logit なら TV は厳密に 0（予算は Wilson の分解能より上に取る必要がある——
    # n=2000・0 件観測でも上側限界は 0.14% なので、それ未満の予算は原理的に満たせない）
    same = compare_task(z, z.copy(), task="sampling", temperature=1.0, flip_budget=0.01)
    assert same.flip_rate == 0.0 and same.max_risk < _Risk.WARN
    # fail-safe: 予算 0（完全一致要求）では 0 件観測でも Wilson 上側限界が 0 を上回るため
    # 警告が出る。他タスクと同じ「0 観測を p=0 と過信しない」規約がサンプリングにも効く。
    strict = compare_task(z, z.copy(), task="sampling", temperature=1.0, flip_budget=0.0)
    assert strict.flip_rate == 0.0 and strict.flip_rate_ub > 0.0
    assert strict.max_risk >= _Risk.WARN
    # fail-safe: 小標本では点推定でなく上側限界で判定する（他タスクと同型）
    small = compare_task(z[:20], _vendor(z, 0.05, 4)[:20], task="sampling",
                         temperature=1.0, flip_budget=0.0)
    assert small.flip_rate_ub > small.flip_rate


def test_tv_bound_from_divergence_bridges_propagation_to_sampling():
    """相対発散 → TV 上界の予測橋（実 logit 無しでサンプリング影響を予測）。"""
    z = _logits(seed=0, n=1000, c=32) * 2.0
    warm = tv_bound_from_divergence(z, 0.01, temperature=1.0)
    cold = tv_bound_from_divergence(z, 0.01, temperature=0.1)
    assert 0.0 < warm < cold <= 1.0        # 低温ほど上界は大きい（飽和方向）
    assert tv_bound_from_divergence(z, 0.0, temperature=1.0) == 0.0
    # 一様 shift は分布を変えない（softmax の shift 不変性）。厳密性を見るため float64 で
    # 確かめる——float32 では 0.05 の加算自体が丸めを生み、ε は 0 でなく ~1e-7 になる
    # （それ自体は正しい: 丸めも実在する微小摂動）。
    z64 = z.astype(np.float64)
    sd = sampling_divergence(z64, z64 + 0.05, temperature=1.0)
    assert sd["tv_mean"] < 1e-12 and sd["eps_max"] < 1e-12
    assert sampling_divergence(z, z + 0.05, temperature=1.0)["eps_max"] < 1e-5


# --- A-9/Q21: 代表集合の裾サポート（予測フリップ率上界の信頼性） ---

def test_flip_bound_support_exposes_unrepresentative_calibration_set():
    """代表集合が決定境界を踏んでいないと予測が偽OK になり、裾サポートがそれを暴く（Q21）。

    Wilson（predicted_flip_bound が使う）は「与えられた集合の P(margin<2δ) の比率不確実性」
    は織り込むが、**その集合が本番分布を代表しているか**は問えない。大マージンばかりの
    代表集合は少数の near-tie しか含まず、near-tie が多い本番のフリップ率を過小評価する
    （偽OK）。この gap は Wilson では閉じない —— 裾サポート（超過数 k）が暴く。
    """
    rng = np.random.default_rng(0)
    delta = 0.05
    # 代表集合: 大マージン・決定境界をほとんど踏まない
    ref = rng.standard_normal((500, 10)) * 6.0
    # 本番: 同じモデルだが入力が境界近傍に多く落ちる
    prod_a = rng.standard_normal((500, 10)) * 0.3
    prod_b = prod_a + delta * rng.standard_normal(prod_a.shape)

    bound = predicted_flip_bound(ref, delta)
    true_flip = flip_rate(prod_a, prod_b)
    # 問題の再現: 代表集合からの予測が本番フリップ率を過小評価する（偽OK）
    assert true_flip > bound * 2.0, (
        f"偽OK gap を再現できていない: 予測 {bound:.3f} vs 本番 {true_flip:.3f}")
    # 診断がそれを暴く: 裾サポート不足（超過数が閾値未満・相対不確実性が大きい）
    sup = flip_bound_support(ref, delta)
    assert not sup["well_supported"]
    assert sup["exceedances"] < sup["min_exceedances"]
    assert sup["rel_uncertainty"] > 0.15               # ≈1/√k が無視できない大きさ

    # 境界を十分踏む代表集合なら well_supported（回帰なし）
    rich = rng.standard_normal((3000, 10)) * 0.3
    sup_rich = flip_bound_support(rich, delta)
    assert sup_rich["well_supported"] and sup_rich["exceedances"] >= sup_rich["min_exceedances"]


def test_flip_bound_support_relative_uncertainty_tracks_one_over_sqrt_k():
    """裾推定の相対不確実性は total n でなく超過数 k に支配される（≈1/√k・n 非依存）。"""
    # 同じ k でも n が桁違いに違っても rel_uncertainty は同じ（k が支配）
    def _set_with_k_exceedances(k, n, delta=0.5):
        m = np.full(n, 10.0)          # 全員大マージン
        m[:k] = 0.5 * delta           # k 件だけ margin<2δ に置く
        # margin(logits)=top1-top2 を m にするため 2 クラス logit [m, 0] を作る
        return np.stack([m, np.zeros(n)], axis=-1)

    for k in (4, 30, 100):
        s_small = flip_bound_support(_set_with_k_exceedances(k, 1000), 0.5)
        s_large = flip_bound_support(_set_with_k_exceedances(k, 100000), 0.5)
        assert s_small["exceedances"] == k and s_large["exceedances"] == k
        # rel_uncertainty ≈ 1/√k, n に依らずほぼ同じ
        assert abs(s_small["rel_uncertainty"] - 1.0 / np.sqrt(k)) < 1e-9
        assert abs(s_small["rel_uncertainty"] - s_large["rel_uncertainty"]) < 1e-9
    # k=0 は inf（外挿しかできない）——全員 margin=100（大）で誰も境界を踏まない
    far = np.stack([np.full(50, 100.0), np.zeros(50)], axis=-1)
    assert flip_bound_support(far, 0.01)["rel_uncertainty"] == float("inf")


def test_min_exceedances_threshold_is_sensitive_to_its_constant():
    """定数 _MIN_EXCEEDANCES が well_supported 境界を *実際に* 支配する（Q5 の感度テスト同型）。"""
    from tsugi.decision import _MIN_EXCEEDANCES

    def _set_with_k(k, n=2000):
        m = np.full(n, 10.0)
        m[:k] = 0.001
        return np.stack([m, np.zeros(n)], axis=-1)

    assert flip_bound_support(_set_with_k(_MIN_EXCEEDANCES), 0.5)["well_supported"]
    assert not flip_bound_support(_set_with_k(_MIN_EXCEEDANCES - 1), 0.5)["well_supported"]


def test_flip_bound_support_from_divergence_matches_bridge_delta():
    """flip_bound_support_from_divergence は flip_bound_from_divergence と同じ δ を使う（Q21）。"""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((1000, 32))
    # 同じ相対発散で、bound と support が同一の δ（=_abs_delta）に基づくこと
    from tsugi.decision import _abs_delta
    d = _abs_delta(z, 0.02)
    assert np.allclose(
        flip_bound_support_from_divergence(z, 0.02)["exceedances"],
        flip_bound_support(z, d)["exceedances"])
    # 空入力でも壊れない
    empty = flip_bound_support_from_divergence(np.zeros((0, 4)), 0.02)
    assert empty["exceedances"] == 0 and not empty["well_supported"]


def main() -> int:
    ok = True
    tests = [
        test_margin_is_top1_minus_top2,
        test_tie_rate_flags_convention_dependent_decisions,
        test_topk_flip_rate_generalizes_argmax,
        test_nucleus_flip_rate_is_probability_dependent,
        test_decision_flips_are_scale_invariant,
        test_flips_concentrate_in_low_margin_tail,
        test_near_tie_threshold_is_sensitive_to_its_constant,
        test_predicted_bound_is_upper_bound,
        test_predicted_bound_uses_wilson_upper_bound_for_small_representative_set,
        test_numerical_divergence_not_sufficient_for_task_divergence,
        test_compare_decisions_reports_topk,
        test_compare_decisions_blocks_high_flip_rate,
        test_compare_decisions_uses_flip_rate_ub_for_small_batch,
        test_systematic_affine_divergence_does_not_flip,
        test_residual_bound_tighter_than_total_for_systematic,
        test_flip_bound_from_divergence_bridges_propagation_to_task,
        test_flip_bound_from_divergence_does_not_underestimate_high_scale_outlier,
        test_regression_flip_rate_matches_value_closeness,
        test_binary_flip_rate_detects_threshold_crossing,
        test_ranking_flip_rate_measures_top_k_set_change,
        test_compare_task_regression_blocks_large_value_drift,
        test_compare_task_uses_flip_rate_ub_for_small_batch,
        test_compare_task_ranking_single_query_no_wilson_widening,
        test_compare_task_binary_ok_for_large_margin,
        test_compare_task_binary_warns_when_flips_not_near_tie,
        test_compare_task_unknown_raises,
        test_tv_bound_is_upper_bound_across_temperature_and_eps,
        test_sampling_epsilon_is_shift_invariant_but_scale_sensitive,
        test_sampling_flip_rate_converges_to_argmax_flip_rate_as_temperature_falls,
        test_tv_bound_reports_itself_vacuous_at_low_temperature,
        test_compare_task_sampling_blocks_high_distribution_divergence,
        test_tv_bound_from_divergence_bridges_propagation_to_sampling,
        test_flip_bound_support_exposes_unrepresentative_calibration_set,
        test_flip_bound_support_relative_uncertainty_tracks_one_over_sqrt_k,
        test_min_exceedances_threshold_is_sensitive_to_its_constant,
        test_flip_bound_support_from_divergence_matches_bridge_delta,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: eps を振った数値発散 vs 判断フリップ率（max_abs は判断を測れない）
    print("\n--- 数値発散 vs 判断フリップ（タスクレベル）---")
    z = _logits()
    for eps in (1e-3, 1e-2, 1e-1):
        a, b = _vendor(z, eps, 1), _vendor(z, eps, 2)
        rep = compare_decisions(a, b, flip_budget=0.001)
        print(f"  eps={eps:.0e} max_abs={np.abs(a - b).max():.3f}  "
              + rep.to_text().splitlines()[0])
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
