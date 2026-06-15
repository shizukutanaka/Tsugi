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
    flip_rate,
    margin,
    predicted_flip_bound,
)


def _logits(seed=0, n=4000, c=200):
    return np.random.default_rng(seed).standard_normal((n, c)).astype(np.float32)


def _vendor(z, eps, seed):
    return z + eps * np.random.default_rng(seed).standard_normal(z.shape).astype(np.float32)


def test_margin_is_top1_minus_top2():
    x = np.array([[1.0, 5.0, 3.0]])
    assert abs(float(margin(x)[0]) - 2.0) < 1e-9   # 5 - 3


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


def test_compare_decisions_blocks_high_flip_rate():
    z = _logits()
    a, b = _vendor(z, 1e-1, 1), _vendor(z, 1e-1, 2)  # 多数フリップ
    rep = compare_decisions(a, b, flip_budget=0.001)
    assert not rep.ok                     # 予算超で BLOCK


def main() -> int:
    ok = True
    tests = [
        test_margin_is_top1_minus_top2,
        test_decision_flips_are_scale_invariant,
        test_flips_concentrate_in_low_margin_tail,
        test_predicted_bound_is_upper_bound,
        test_numerical_divergence_not_sufficient_for_task_divergence,
        test_compare_decisions_blocks_high_flip_rate,
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
