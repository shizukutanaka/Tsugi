"""非決定実行とノイズフロアのテスト（出力は点でなく分布）。

GPU の atomic 非決定で run-to-run に揺れること、単一 run 比較がノイズと発散を
混同すること、クロス差がノイズ未満なら INDISTINGUISHABLE になることを実証。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tsugi.nondeterminism import (  # noqa: E402
    DIVERGENT,
    EQUIVALENT,
    INDISTINGUISHABLE,
    attribute,
    compare_stable,
    measure_batch_variance,
    measure_noise_floor,
    simulate_batch_variant_reduction,
    simulate_nondeterministic_reduction,
)


def _parts(seed: int, K: int = 4096) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(K).astype(np.float32)


def test_batch_variance_is_deterministic_but_batch_dependent():
    # batch-invariance（Thinking Machines 2025）: バッチ依存タイルで結果が変わるが、
    # 同じバッチなら毎回同じ（atomic ノイズと違い決定論的）。
    p = _parts(0)
    r128_a = float(simulate_batch_variant_reduction(p, 128))
    r128_b = float(simulate_batch_variant_reduction(p, 128))
    r256 = float(simulate_batch_variant_reduction(p, 256))
    assert r128_a == r128_b          # 同じバッチ → 決定論的（再現する）
    assert r128_a != r256            # バッチが違う → 結果が変わる


def test_batch_variance_floor_is_positive():
    p = _parts(0)
    bv = measure_batch_variance(lambda t: simulate_batch_variant_reduction(p, t))
    assert bv["spread"] > 0.0        # バッチ変動は独立した床を生む
    assert bv["n_batches"] >= 2


def test_batch_floor_folds_into_effective_floor():
    # batch_floor が run-to-run を支配すると実効床になり batch-limited を WARN する
    p = _parts(0, K=64).reshape(4, 16)

    def run(s):
        g = np.random.default_rng(1000 + s).standard_normal(p.shape).astype(np.float32)
        return p + 1e-6 * g

    rep = compare_stable(run, run, K=256, n_runs=8, batch_floor=1e-2)
    assert rep.noise_floor >= 1e-2          # batch_floor が実効床に合流
    assert any(f.op == "batch" for f in rep.findings)


def test_reduction_is_nondeterministic():
    # 同一入力でも seed(=atomic順)で結果が揺れる＝出力は点でない
    p = _parts(0)
    r1 = float(simulate_nondeterministic_reduction(p, 1))
    r2 = float(simulate_nondeterministic_reduction(p, 2))
    assert r1 != r2                       # run-to-run で異なる
    assert abs(r1 - r2) < 1e-2 * abs(r1)  # だが小さい（順序差のみ）


def test_noise_floor_is_positive():
    p = _parts(0)
    nf = measure_noise_floor(lambda s: simulate_nondeterministic_reduction(p, s), n_runs=20)
    assert nf["spread"] > 0.0   # 決定論仮定（noise=0）は誤り


def test_attribute_three_regimes():
    assert attribute(0.5e-4, 1e-4, 1e-2) == INDISTINGUISHABLE  # ノイズ未満
    assert attribute(5e-3, 1e-4, 1e-2) == EQUIVALENT           # ノイズ超・許容内
    assert attribute(2e-2, 1e-4, 1e-2) == DIVERGENT            # 許容超


def test_indistinguishable_for_truly_equivalent_vendors():
    # 真に等価な 2 ベンダー（同じ部分和・atomic 順だけ違う）→ クロス差はノイズ程度
    # 累積は fp32 なので dtype=float32 が忠実（fp16 許容は粗すぎて何でも EQ にする）
    p = _parts(0)
    rep = compare_stable(
        lambda s: simulate_nondeterministic_reduction(p, 100 + s),
        lambda s: simulate_nondeterministic_reduction(p, 900 + s),
        K=4096, dtype="float32", n_runs=16)
    assert rep.verdict in (INDISTINGUISHABLE, EQUIVALENT)
    assert rep.verdict != DIVERGENT       # ノイズを発散と誤判定しない


def test_real_divergence_still_detected_above_noise():
    # 片ベンダーが系統的に大きくずれる（部分和を 5% 底上げ）→ ノイズを超え DIVERGENT
    p = _parts(0)
    rep = compare_stable(
        lambda s: simulate_nondeterministic_reduction(p, 100 + s),
        lambda s: simulate_nondeterministic_reduction(p * 1.05, 900 + s),
        K=4096, dtype="float32", n_runs=16)
    assert rep.verdict == DIVERGENT


def test_single_run_comparison_is_flaky():
    # 真に等価な 2 ベンダーの単一 run 比較は、許容を観測差の中間に置くと判定が割れる
    p = _parts(0)
    diffs = [abs(float(simulate_nondeterministic_reduction(p, 100 + t))
                 - float(simulate_nondeterministic_reduction(p, 900 + t)))
             for t in range(24)]
    tol = (min(diffs) + max(diffs)) / 2   # 観測差の中間（ノイズ帯の内側）
    verdicts = {"EQ" if d <= tol else "DV" for d in diffs}
    assert verdicts == {"EQ", "DV"}   # 同じ真に等価な対が run の引きで EQ/DV に揺れる


def main() -> int:
    ok = True
    tests = [
        test_batch_variance_is_deterministic_but_batch_dependent,
        test_batch_variance_floor_is_positive,
        test_batch_floor_folds_into_effective_floor,
        test_reduction_is_nondeterministic,
        test_noise_floor_is_positive,
        test_attribute_three_regimes,
        test_indistinguishable_for_truly_equivalent_vendors,
        test_real_divergence_still_detected_above_noise,
        test_single_run_comparison_is_flaky,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: ノイズフロア実測と 3 状態の境界
    p = _parts(0)
    nf = measure_noise_floor(lambda s: simulate_nondeterministic_reduction(p, s), 20)
    print("\n--- run-to-run ノイズフロア（atomic 非決定）---")
    print(f"  spread={nf['spread']:.3e}  rel={nf['rel']:.2e}  (n={nf['n_runs']})")
    rep = compare_stable(
        lambda s: simulate_nondeterministic_reduction(p, 100 + s),
        lambda s: simulate_nondeterministic_reduction(p, 900 + s), 4096, "float16", 16)
    print("  " + rep.to_text().replace("\n", "\n  "))
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
