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
    compare_decisions,
    decision_flips,
    decompose_divergence,
    flip_rate,
    margin,
    predicted_flip_bound,
    residual_divergence_rms,
    topk_flip_rate,
)


def _logits(seed=0, n=4000, c=200):
    return np.random.default_rng(seed).standard_normal((n, c)).astype(np.float32)


def _vendor(z, eps, seed):
    return z + eps * np.random.default_rng(seed).standard_normal(z.shape).astype(np.float32)


def test_margin_is_top1_minus_top2():
    x = np.array([[1.0, 5.0, 3.0]])
    assert abs(float(margin(x)[0]) - 2.0) < 1e-9   # 5 - 3


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


def test_predicted_bound_is_upper_bound():
    z = _logits()
    for eps in (1e-3, 1e-2, 1e-1):
        a, b = _vendor(z, eps, 1), _vendor(z, eps, 2)
        from tsugi.decision import divergence_rms
        actual = flip_rate(a, b)
        bound = predicted_flip_bound(a, divergence_rms(a, b))
        assert actual <= bound + 1e-9     # P(margin<2δ) は実フリップ率の上界


def test_numerical_divergence_not_sufficient_for_task_divergence():
    # マージンが大きい(確信)モデルは、相当な数値発散(max_abs~0.37)でもタスク影響は無視可能
    z = _logits() * 100.0   # logit を増幅 → マージン大
    a, b = _vendor(z, 5e-2, 1), _vendor(z, 5e-2, 2)
    assert np.abs(a - b).max() > 0.1      # 数値 abs 発散は相当大きい
    assert flip_rate(a, b) < 0.005        # だがタスク(判断)影響は無視可能
    assert compare_decisions(a, b, flip_budget=0.01).ok  # タスク予算内で等価


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


def test_compare_decisions_blocks_high_flip_rate():
    z = _logits()
    a, b = _vendor(z, 1e-1, 1), _vendor(z, 1e-1, 2)  # 多数フリップ
    rep = compare_decisions(a, b, flip_budget=0.001)
    assert not rep.ok                     # 予算超で BLOCK


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
        test_topk_flip_rate_generalizes_argmax,
        test_decision_flips_are_scale_invariant,
        test_flips_concentrate_in_low_margin_tail,
        test_predicted_bound_is_upper_bound,
        test_numerical_divergence_not_sufficient_for_task_divergence,
        test_compare_decisions_reports_topk,
        test_compare_decisions_blocks_high_flip_rate,
        test_systematic_affine_divergence_does_not_flip,
        test_residual_bound_tighter_than_total_for_systematic,
        test_flip_bound_from_divergence_bridges_propagation_to_task,
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
