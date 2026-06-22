"""rollout（新視点9: 自己回帰的発散）のテスト。

per-token 等価 ⇏ per-sequence 等価。フリップ率を生成長へ合成し、解析式を
Monte Carlo で確認、verdict が長さに応じて反転することを実証する。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tsugi.report import Risk  # noqa: E402
from tsugi.rollout import (  # noqa: E402
    analyze_rollout,
    expected_divergence_step,
    rollout_from_logits,
    safe_generation_length,
    sequence_survival,
    simulate_rollout,
)


def test_survival_compounds_over_length():
    # per-token 許容(1%)がシーケンスでは破綻: survival は length で複利的に減衰
    assert abs(sequence_survival(0.01, 1) - 0.99) < 1e-12
    assert sequence_survival(0.01, 100) < sequence_survival(0.01, 10)
    assert abs(sequence_survival(0.01, 100) - 0.99 ** 100) < 1e-12
    # 単調減少（length が増えれば survival は増えない）
    prev = 1.0
    for L in (0, 1, 10, 100, 1000):
        s = sequence_survival(0.02, L)
        assert s <= prev + 1e-12
        prev = s


def test_perfect_alignment_never_diverges():
    # p=0 は決して分岐しない（任意長で安全）
    assert sequence_survival(0.0, 10**6) == 1.0
    assert math.isinf(expected_divergence_step(0.0))
    assert safe_generation_length(0.0) > 10**12
    assert analyze_rollout(0.0, 10**6).max_risk == Risk.OK


def test_expected_divergence_is_geometric_mean():
    # 初回発散の期待位置 = 1/p（幾何分布）
    assert abs(expected_divergence_step(0.01) - 100.0) < 1e-9
    assert abs(expected_divergence_step(0.5) - 2.0) < 1e-9


def test_safe_length_matches_confidence():
    # safe_len は survival ≥ confidence を保つ最大長（境界で survival が confidence を跨ぐ）
    p, conf = 0.01, 0.99
    L = safe_generation_length(p, conf)
    assert sequence_survival(p, L) >= conf
    assert sequence_survival(p, L + 1) < conf


def test_monte_carlo_confirms_analytic_survival():
    # 解析式 (1−p)^L を独立 Monte Carlo が再現する（モデルの自己検証）
    for p, L in ((0.01, 50), (0.05, 20), (0.1, 10)):
        analytic = sequence_survival(p, L)
        empirical = simulate_rollout(p, L, trials=20000, seed=1)
        assert abs(analytic - empirical) < 0.02, f"p={p} L={L}: {analytic} vs {empirical}"


def test_verdict_flips_with_generation_length():
    # 同じ per-token フリップ率でも、生成長で verdict が OK→WARN→BLOCK と反転する
    p = 0.01
    assert analyze_rollout(p, 1).max_risk == Risk.OK            # 1 トークンなら許容
    mid = analyze_rollout(p, 100).max_risk
    long = analyze_rollout(p, 5000).max_risk
    assert mid >= Risk.WARN                                     # 100 トークンで赤信号化
    assert long == Risk.BLOCK                                   # 長文生成では分岐優勢
    assert long >= mid                                          # 長いほど厳しい（単調）


def test_rollout_from_logits_uses_per_token_flip_rate():
    # 代表 logit から per-token フリップ率を測り長さへ合成（decision との接続）
    rng = np.random.default_rng(0)
    base = rng.standard_normal((2000, 100)).astype(np.float32)
    # near-tie を少数仕込み、片ベンダーでだけ僅かに摂動 → 一定のフリップ率
    b = base + 1e-3 * rng.standard_normal(base.shape).astype(np.float32)
    short = rollout_from_logits(base, b, target_length=1)
    longr = rollout_from_logits(base, b, target_length=2000)
    assert 0.0 <= short.flip_rate <= 1.0
    assert longr.survival <= short.survival                     # 長いほど survival 低下
    assert longr.flip_rate == short.flip_rate                   # 同じ p を別長へ合成


def main() -> int:
    ok = True
    tests = [
        test_survival_compounds_over_length,
        test_perfect_alignment_never_diverges,
        test_expected_divergence_is_geometric_mean,
        test_safe_length_matches_confidence,
        test_monte_carlo_confirms_analytic_survival,
        test_verdict_flips_with_generation_length,
        test_rollout_from_logits_uses_per_token_flip_rate,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: per-token 1% が生成長でどう破綻するか
    print("\n--- per-token 1% フリップの自己回帰的破綻 ---")
    for L in (1, 10, 100, 1000):
        print("  " + analyze_rollout(0.01, L).to_text())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
