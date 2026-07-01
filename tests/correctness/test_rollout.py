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
    divergence_step_quantile,
    expected_divergence_step,
    flip_rate_upper_bound,
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


def test_median_divergence_step_is_smaller_than_mean():
    """初回発散ステップの中央値は平均より系統的に小さい（幾何分布の右裾・第19回）。

    divergence_step_quantile は実装・テスト済みだったが、どのレポートからも
    参照されていなかった（過剰実装＝実質デッドコード）。expected_step（平均=1/p）
    だけ見ると「典型的にはもっと長く保つ」と楽観視しやすい —— 右裾の少数の長生存
    run に平均が引っ張られるため。analyze_rollout に接続し両方報告することで、
    「平均は大きいが半数の run はもっと早く分岐する」という fail-safe な事実を隠さない。
    """
    for p in (0.001, 0.01, 0.1, 0.5):
        mean = expected_divergence_step(p)
        median = divergence_step_quantile(p, 0.5)
        assert median < mean, f"p={p}: median({median}) が mean({mean}) 以上（幾何分布の性質に反する）"
        assert median > 0

    # 解析的な検証値: p=0.01 → mean=100, median=ceil(ln(0.5)/ln(0.99))=69
    assert abs(expected_divergence_step(0.01) - 100.0) < 1e-9
    assert divergence_step_quantile(0.01, 0.5) == 69.0

    # analyze_rollout の RolloutReport に median_step が接続されている
    rep = analyze_rollout(0.01, 100)
    assert rep.median_step == 69.0
    assert "中央値" in rep.to_text()


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


def test_zero_observed_flips_is_not_false_confidence():
    # 改善: 0 フリップ観測でも p=0 と過信しない（fail-safe）。
    # 点推定なら survival=100%/safe_len=∞ だが、保守版は上限 p>0 で有限に倒す。
    rng = np.random.default_rng(0)
    # 完全一致 logit（フリップ 0）を少標本で
    a = rng.standard_normal((200, 50)).astype(np.float32)
    b = a.copy()
    point = rollout_from_logits(a, b, target_length=10**6, conservative=False)
    safe = rollout_from_logits(a, b, target_length=10**6, conservative=True)
    assert point.flip_rate == 0.0 and point.survival == 1.0       # 点推定は過信
    assert safe.flip_rate > 0.0                                   # 保守版は不確実性を計上
    assert safe.survival < 1.0                                    # 過信しない
    assert safe.max_risk >= point.max_risk                        # fail-safe 側


def test_decode_mode_matches_generation_sampling():
    # 改善: 生成は argmax だけでない。サンプリング生成では候補集合の分岐が per-token
    # 発散になる。nucleus/top-k の集合フリップ率は greedy argmax フリップ率以上になりうる
    # （argmax が同じでも候補集合は分岐しうる）→ サンプリング生成の発散を過小評価しない。
    rng = np.random.default_rng(3)
    a = rng.standard_normal((1500, 80)).astype(np.float32)
    b = a + 5e-2 * rng.standard_normal(a.shape).astype(np.float32)
    greedy = rollout_from_logits(a, b, target_length=256, decode="greedy")
    nucleus = rollout_from_logits(a, b, target_length=256, decode="nucleus",
                                  top_p=0.9, temperature=1.0)
    topk = rollout_from_logits(a, b, target_length=256, decode="topk", topk=5)
    # 集合フリップは argmax フリップ以上 → survival は greedy 以下（より厳しい/honest）
    assert nucleus.flip_rate >= greedy.flip_rate
    assert topk.flip_rate >= greedy.flip_rate
    assert nucleus.survival <= greedy.survival + 1e-12
    # 未知のデコード方式は弾く
    try:
        rollout_from_logits(a, b, target_length=10, decode="beam")
        raise AssertionError("unknown decode mode should raise")
    except ValueError:
        pass


def test_upper_bound_properties():
    # rule of three 近傍: 0/n の上限 ~ 3/n オーダー、標本増で縮小、点推定以上
    assert flip_rate_upper_bound(0, 0) == 1.0                     # データ無し=最大不確実
    ub_small = flip_rate_upper_bound(0, 100)
    ub_large = flip_rate_upper_bound(0, 10000)
    assert 0.0 < ub_large < ub_small < 0.1                        # n↑ で上限縮小
    assert flip_rate_upper_bound(5, 100) >= 5 / 100               # 点推定以上（保守）


def main() -> int:
    ok = True
    tests = [
        test_survival_compounds_over_length,
        test_perfect_alignment_never_diverges,
        test_expected_divergence_is_geometric_mean,
        test_median_divergence_step_is_smaller_than_mean,
        test_safe_length_matches_confidence,
        test_monte_carlo_confirms_analytic_survival,
        test_verdict_flips_with_generation_length,
        test_rollout_from_logits_uses_per_token_flip_rate,
        test_zero_observed_flips_is_not_false_confidence,
        test_decode_mode_matches_generation_sampling,
        test_upper_bound_properties,
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
