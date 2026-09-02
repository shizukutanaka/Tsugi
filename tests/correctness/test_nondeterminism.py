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
    classify_nondeterminism,
    collect_runs,
    compare_stable,
    measure_batch_variance,
    measure_noise_floor,
    noise_floor_from_runs,
    nondeterminism_reason,
    pair_deviations,
    op_is_nondeterministic,
    simulate_batch_variant_reduction,
    simulate_nondeterministic_reduction,
)
from tsugi.report import Risk  # noqa: E402


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


def test_robust_floor_rejects_single_outlier():
    # 1 回の測定グリッチで max-min 床は膨張するが robust 床（10-90%）は頑健（Q49）
    p = _parts(0, K=64).reshape(4, 16)

    def run(s):
        g = np.random.default_rng(1000 + s).standard_normal(p.shape).astype(np.float32)
        return p + (5e-2 if s == 0 else 1e-6) * g   # seed0 だけ外れ値

    nf = measure_noise_floor(run, 16)
    assert nf["spread_robust"] < nf["spread"] * 1e-2   # robust は外れ値で膨張しない
    assert nf["spread"] > 1e-3                          # max-min は外れ値で膨張


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


def test_atomic_op_catalog_flags_pytorch_nondeterministic_ops():
    # PyTorch 公式が atomicAdd 由来で非決定と明示する op を静的に識別する
    # （https://pytorch.org/docs/stable/notes/randomness.html）
    for name in ("scatter_add", "index_add", "bincount", "embedding_bag", "ctc_loss"):
        assert op_is_nondeterministic(name), f"{name} を非決定と識別できない"
        assert nondeterminism_reason(name) is not None
    # 決定論的な op は誤検出しない
    for name in ("matmul", "softmax", "add", "relu", "layernorm"):
        assert not op_is_nondeterministic(name), f"{name} を誤って非決定扱い"
        assert nondeterminism_reason(name) is None


def test_histc_and_put_are_atomic_nondet_ops():
    # histc(CUDA ヒストグラム)・put_(accumulate=True) も atomicAdd 由来（PyTorch 公式 doc 掲載）
    for name in ("histc", "put_"):
        assert op_is_nondeterministic(name), f"{name} を非決定と識別できない"
        reason = nondeterminism_reason(name)
        assert reason is not None and "atomicAdd" in reason


def test_cumsum_kthvalue_median_are_nondet_but_not_atomic():
    # cumsum/kthvalue/median は atomicAdd でなく CUDA の並列縮約・tie-break 順序で揺れる
    # （PyTorch 公式 doc 掲載）。機構が違うため reason 文字列は "atomicAdd" と書いてはいけない
    # （catalog の requires_noise_floor 判定を偽情報にしないため・機構の正確な区別）。
    for name in ("cumsum", "kthvalue", "median"):
        assert op_is_nondeterministic(name), f"{name} を非決定と識別できない"
        reason = nondeterminism_reason(name)
        assert reason is not None
        assert "atomicAdd" not in reason, f"{name} は非atomicなのに reason が atomicAdd と誤記: {reason}"
        assert "非atomic" in reason


def test_op_catalog_tolerates_naming_variants():
    # 表記揺れ（末尾 _・aten 修飾・次元サフィックス）を前方一致で吸収する
    assert op_is_nondeterministic("scatter_add_")
    assert op_is_nondeterministic("aten.scatter_add.default")
    assert op_is_nondeterministic("max_pool2d")
    assert op_is_nondeterministic("ADAPTIVE_AVG_POOL3D")   # 大文字でも一致


def test_classify_nondeterminism_requires_noise_floor():
    # 非決定 op を含むグラフは noise floor 実測が必須と宣言される
    graph = ["matmul", "softmax", "scatter_add", "add"]
    rep = classify_nondeterminism(graph)
    assert rep.requires_noise_floor
    assert rep.nondet_ops == ("scatter_add",)
    assert rep.max_risk == Risk.WARN
    assert any("noise floor" in f.message for f in rep.findings)

    # 決定論的グラフは静的許容で十分（noise floor 不要）
    det = classify_nondeterminism(["matmul", "softmax", "layernorm", "add"])
    assert not det.requires_noise_floor
    assert det.nondet_ops == ()
    assert det.max_risk == Risk.OK


def test_runs_to_resolve_turns_indistinguishable_into_a_next_step():
    """INDISTINGUISHABLE を終端でなく「あと N run」で決着する状態にする。

    背景（文献）: 単発比較では良性の浮動小数ノイズと真の差を区別できないが、独立な run を
    平均すれば平均のノイズは σ/√N に縮み、系統差 d は縮まない——よって SNR = d·√N/σ は
    伸び、いずれ分離できる。必要条件 d > z·σ/√N より N > (z·σ/d)²。DiFR
    ("Inference Verification Despite Nondeterminism") が多数トークンに証拠を累積して
    設定誤りを検出するのと同型の枠組み（docs/SOURCES.md）。
    """
    import math

    from tsugi.nondeterminism import _erfinv, runs_to_resolve

    # 逆誤差関数近似の精度（標準的な z 値と一致すること）
    assert abs(math.sqrt(2) * _erfinv(2 * 0.95 - 1) - 1.6449) < 0.01
    assert abs(math.sqrt(2) * _erfinv(2 * 0.99 - 1) - 2.3263) < 0.01

    # 分離不要なケースは 0（差が無い・ノイズが無い・既に分離済み）
    assert runs_to_resolve(0.0, 1e-2) == 0
    assert runs_to_resolve(1e-3, 0.0) == 0
    assert runs_to_resolve(2e-2, 1e-2) == 0      # d > σ は既に分離済み

    # N は (σ/d)² に比例（d を半分にすると 4 倍）
    n1 = runs_to_resolve(1e-3, 1e-2)
    n2 = runs_to_resolve(5e-4, 1e-2)
    assert n1 > 1 and abs(n2 / n1 - 4.0) < 0.1, (n1, n2)

    # 信頼水準を上げるほど多くの run が要る
    assert runs_to_resolve(1e-3, 1e-2, 0.99) > runs_to_resolve(1e-3, 1e-2, 0.95)


def test_compare_stable_reports_runs_needed_when_indistinguishable():
    """compare_stable が INDISTINGUISHABLE 時に「あと何 run で決着するか」を出す。

    従来この判定は「等価判定は未定義」で行き止まりだった（ユーザーに次手が無い）。
    """
    from tsugi.nondeterminism import INDISTINGUISHABLE, compare_stable

    rng = np.random.default_rng(0)
    base = rng.standard_normal((32, 32)).astype(np.float32)

    # 両ベンダーとも同程度の run-to-run ノイズを持ち、クロス差がノイズに埋もれる構成
    def run_a(seed):
        return base + np.random.default_rng(1000 + seed).standard_normal(base.shape).astype(np.float32) * 1e-3

    def run_b(seed):
        return base + np.random.default_rng(2000 + seed).standard_normal(base.shape).astype(np.float32) * 1e-3

    rep = compare_stable(run_a, run_b, K=256, dtype="float16", n_runs=8)
    if rep.verdict == INDISTINGUISHABLE:
        assert rep.runs_needed >= 0
        txt = rep.to_text()
        assert "区別不能" in txt
        # 差があるなら具体的な run 数の提示、無いなら「差が無い」旨を出す
        assert ("run を平均すれば分離可能" in txt) or ("差が無い" in txt), txt


def test_pair_deviations_samples_pair_differences_not_center_deviations():
    """校正標本は「run 対の差」であって「中心からの偏差」ではない（単位の一致）。

    等価判定が比較するのは max|a-b|（2 つの単発 run の差）。もし校正標本に
    中心（平均/中央値）からの偏差を使うと、測る量が比較される量の約半分になり、
    要求 SAFETY を系統的に約 2 倍過小評価する ——「許容は十分」と誤って言う
    偽OK 方向の誤り。ここでは既知の分布（±d の 2 値）で両者の比を固定する。
    """
    d = 3.0
    stack = np.stack([np.full((4,), -d if i % 2 == 0 else d) for i in range(8)])

    pairs = pair_deviations(stack)
    assert pairs.shape == (4,), f"重ならない対 floor(8/2)=4 個のはずが {pairs.shape}"
    assert np.allclose(pairs, 2 * d), f"対の差は 2d={2 * d} のはず: {pairs}"

    center_dev = float(np.abs(stack - stack.mean(axis=0)).max())
    assert np.isclose(center_dev, d)
    assert np.allclose(pairs, 2 * center_dev), \
        "中心偏差は対の差の半分——校正に使うと要求 SAFETY を 2 倍過小評価する"

    # 対が作れない（run が 1 本）なら空を返す（0 を「揺れなし」と誤らせない）
    assert pair_deviations(stack[:1]).size == 0


def test_noise_floor_from_runs_matches_measure_noise_floor():
    """スタック再利用版が従来の再実行版と同じ床を出す（実機 run を二度走らせない）。"""
    def run(s):
        return np.random.default_rng(1234 + s).standard_normal((4, 8))

    stack = collect_runs(run, n_runs=8)
    assert stack.shape == (8, 4, 8)
    reused = noise_floor_from_runs(stack)
    fresh = measure_noise_floor(run, n_runs=8)
    for key in ("spread", "spread_robust", "std", "rel", "n_runs"):
        assert np.isclose(reused[key], fresh[key]), f"{key}: {reused[key]} vs {fresh[key]}"


def main() -> int:
    ok = True
    tests = [
        test_batch_variance_is_deterministic_but_batch_dependent,
        test_robust_floor_rejects_single_outlier,
        test_batch_variance_floor_is_positive,
        test_batch_floor_folds_into_effective_floor,
        test_reduction_is_nondeterministic,
        test_noise_floor_is_positive,
        test_attribute_three_regimes,
        test_indistinguishable_for_truly_equivalent_vendors,
        test_real_divergence_still_detected_above_noise,
        test_single_run_comparison_is_flaky,
        test_atomic_op_catalog_flags_pytorch_nondeterministic_ops,
        test_histc_and_put_are_atomic_nondet_ops,
        test_cumsum_kthvalue_median_are_nondet_but_not_atomic,
        test_op_catalog_tolerates_naming_variants,
        test_classify_nondeterminism_requires_noise_floor,
        test_runs_to_resolve_turns_indistinguishable_into_a_next_step,
        test_compare_stable_reports_runs_needed_when_indistinguishable,
        test_pair_deviations_samples_pair_differences_not_center_deviations,
        test_noise_floor_from_runs_matches_measure_noise_floor,
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
