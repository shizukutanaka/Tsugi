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
    flip_rate,
    margin,
    nucleus_flip_rate,
    predicted_flip_bound,
    ranking_flip_rate,
    regression_flip_rate,
    residual_divergence_rms,
    tie_rate,
    topk_flip_rate,
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
        test_regression_flip_rate_matches_value_closeness,
        test_binary_flip_rate_detects_threshold_crossing,
        test_ranking_flip_rate_measures_top_k_set_change,
        test_compare_task_regression_blocks_large_value_drift,
        test_compare_task_uses_flip_rate_ub_for_small_batch,
        test_compare_task_ranking_single_query_no_wilson_widening,
        test_compare_task_binary_ok_for_large_margin,
        test_compare_task_binary_warns_when_flips_not_near_tie,
        test_compare_task_unknown_raises,
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
