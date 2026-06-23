"""worstcase（新視点10: 最悪ケース発散探索）のテスト。

平均ケース等価 ⇏ 最悪ケース等価。代表データ上は良性に見えるベンダー対でも、
認証エンベロープ内に発散を最大化する反例（near-cancellation）が能動探索で見つかる
ことを実証する。真に同一なベンダーには偽陽性を出さないことも確認。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tsugi.report import Risk  # noqa: E402
from tsugi.worstcase import (  # noqa: E402
    analyze_worst_case,
    divergence,
    search_worst_input,
)


def _accum_precision_vendors():
    """累積精度が違う 2 ベンダーの二乗和。入力規模が増すと fp16 累積は swamping/飽和し、
    発散が *単調に* 増える（代表的な小入力では良性・大入力で破綻）。"""
    def fp16(x):
        acc = np.float16(0.0)
        for v in np.asarray(x, dtype=np.float16):
            acc = np.float16(acc + np.float16(v * v))
        return np.array([acc], dtype=np.float64)

    def fp32(x):
        return np.array([np.sum(np.asarray(x, dtype=np.float32) ** 2)], dtype=np.float64)

    return fp16, fp32


def test_worst_case_exceeds_typical():
    # 能動探索の最悪発散は代表サンプルの典型発散を上回る（平均は最悪を過小評価）
    fp16, fp32 = _accum_precision_vendors()
    rng = np.random.default_rng(0)
    samples = [rng.standard_normal(64) for _ in range(32)]
    rep = analyze_worst_case(fp16, fp32, samples, tol=1e-2, radius=8.0, steps=600, seed=1)
    assert rep.worst_divergence >= rep.typical_divergence
    assert rep.amplification > 1.0
    assert rep.n_samples == 32


def test_in_envelope_counterexample_is_block():
    # 代表サンプルは tol 内（平均ケースは合格）でも、エンベロープ内に許容超過の反例が
    # 能動探索で見つかれば BLOCK。平均ケース等価 ⇏ 最悪ケース等価。
    fp16, fp32 = _accum_precision_vendors()
    rng = np.random.default_rng(0)
    samples = [rng.standard_normal(64) for _ in range(16)]
    rep = analyze_worst_case(fp16, fp32, samples, tol=1e-3, bounds=(-30.0, 30.0),
                             steps=900, seed=1)
    assert rep.typical_divergence < rep.tol          # 平均ケースは合格に見える
    assert rep.worst_divergence > rep.tol            # だが反例が許容を超える
    assert rep.max_risk == Risk.BLOCK
    assert rep.x_worst is not None                   # 再現可能な反例を返す


def test_identical_vendor_has_no_false_positive():
    # 真に同一なベンダー（同じ関数）は探索しても発散 0 → OK（偽陽性を出さない）
    fp16, _ = _accum_precision_vendors()
    rng = np.random.default_rng(0)
    samples = [rng.standard_normal(32) for _ in range(8)]
    rep = analyze_worst_case(fp16, fp16, samples, tol=1e-6, steps=200, seed=0)
    assert rep.worst_divergence == 0.0
    assert rep.max_risk == Risk.OK


def test_search_is_reproducible():
    # seed 固定で反例は再現する（反例は監査可能な成果物）
    fp16, fp32 = _accum_precision_vendors()
    x0 = np.linspace(-3, 3, 32)
    x1, d1 = search_worst_input(fp16, fp32, x0, steps=300, seed=7)
    x2, d2 = search_worst_input(fp16, fp32, x0, steps=300, seed=7)
    assert d1 == d2 and np.allclose(x1, x2)


def test_search_never_worsens_start():
    # ヒルクライムは開始点以上の最悪発散を返す（探索は悪化しない）
    fp16, fp32 = _accum_precision_vendors()
    x0 = np.linspace(0.1, 2.0, 48)                    # 良性な小入力の開始点
    d_start = divergence(fp16, fp32, x0)
    _, d_w = search_worst_input(fp16, fp32, x0, radius=8.0, steps=500, seed=1)
    assert d_w >= d_start


def test_bounds_constrain_the_search():
    # 探索領域（エンベロープ）を狭めると最悪発散は緩む（box 制約が効く＝envelope と接続）
    fp16, fp32 = _accum_precision_vendors()
    rng = np.random.default_rng(5)
    samples = [rng.standard_normal(32) for _ in range(8)]
    wide = analyze_worst_case(fp16, fp32, samples, tol=1e-2, steps=400, seed=0,
                              bounds=(-30.0, 30.0))
    tight = analyze_worst_case(fp16, fp32, samples, tol=1e-2, steps=400, seed=0,
                               bounds=(-0.5, 0.5))
    assert wide.worst_divergence >= tight.worst_divergence


def main() -> int:
    ok = True
    tests = [
        test_worst_case_exceeds_typical,
        test_in_envelope_counterexample_is_block,
        test_identical_vendor_has_no_false_positive,
        test_search_is_reproducible,
        test_search_never_worsens_start,
        test_bounds_constrain_the_search,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: 代表は tol 内でもエンベロープ内に反例が在る
    fp16, fp32 = _accum_precision_vendors()
    rng = np.random.default_rng(0)
    samples = [rng.standard_normal(64) for _ in range(16)]
    print("\n--- 平均ケース等価 ⇏ 最悪ケース等価 ---")
    print(analyze_worst_case(fp16, fp32, samples, tol=1e-3, bounds=(-30.0, 30.0),
                             steps=900, seed=1).to_text())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
