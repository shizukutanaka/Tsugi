"""error propagation の検証（新視点・第4ラウンド）。

per-kernel 等価 ⇏ per-model 等価 を、合成モデルと実 numpy シミュレーションの
両方で実証する。深さに比例して発散が累積し、ill-conditioned op で増幅されること、
そして単一カーネルの許容では足りないことを示す。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tsugi.equivalence import simulate_vendor_matmul  # noqa: E402
from tsugi.propagation import (  # noqa: E402
    GraphOp,
    amplification,
    empirical_cond,
    is_amplifier,
    merge_divergence,
    model_tolerance,
    propagate,
    propagate_dag,
)
from tsugi.tolerance import expected_gemm_abs_error  # noqa: E402


def test_depth_accumulates_divergence():
    # 同じ matmul を 1 層 vs 12 層: モデル発散は深さに比例して増える
    one = model_tolerance([GraphOp("matmul", K=256)])
    deep = model_tolerance([GraphOp("matmul", K=256)] * 12)
    assert deep > one
    # ほぼ線形（12 層は 1 層の 10〜14 倍程度）
    assert 10 < deep / one < 14, f"depth scaling off: {deep / one}"


def test_naive_per_kernel_underestimates():
    # 増幅 op を挟むと素朴な local 和より合成発散が大きい
    ops = [GraphOp("matmul", K=512),
           GraphOp("softmax", cond=8.0),   # ill-conditioned（大 logit）
           GraphOp("matmul", K=512)]
    rep = propagate(ops)
    assert rep.model_divergence > rep.naive_sum, \
        "composition must exceed naive per-kernel sum"


def test_ill_conditioned_op_is_dominant():
    ops = [GraphOp("matmul", K=128),
           GraphOp("reduce", cond=50.0),   # 強い相殺
           GraphOp("add")]
    rep = propagate(ops)
    assert rep.dominant.kind == "reduce", f"dominant should be reduce, got {rep.dominant.kind}"


def _run_chain(L, n, K, seed=0, residual=False, relative=False):
    """L 層の matmul+rmsnorm を 2 ベンダー（累積順序違い）で流し発散を返す。

    residual=True なら pre-norm 残差ブロック x = x + matmul(rmsnorm(x), w)。
    relative=True なら相対発散（||a-b||/||a||）、既定は最大絶対発散。
    """
    rng = np.random.default_rng(seed)
    x_a = rng.standard_normal((n, K)).astype(np.float16)
    x_b = x_a.copy()
    weights = [rng.standard_normal((K, K)).astype(np.float16) * (1.0 / np.sqrt(K))
               for _ in range(L)]

    def rmsnorm(x):
        return x / (np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=1, keepdims=True)) + 1e-6)

    for w in weights:
        # vendor A: f32 累積 / vendor B: split-k（累積順序差）= 別ベンダー相当
        if residual:  # pre-norm 残差: x = x + matmul(rmsnorm(x), w)
            fa = simulate_vendor_matmul(rmsnorm(x_a).astype(np.float16), w, accum="f32")
            fb = simulate_vendor_matmul(rmsnorm(x_b).astype(np.float16), w, accum="f32", split_k=8)
            x_a = x_a.astype(np.float64) + fa
            x_b = x_b.astype(np.float64) + fb
        else:         # 平坦チェーン: x = rmsnorm(matmul(x, w))
            x_a = rmsnorm(simulate_vendor_matmul(x_a.astype(np.float16), w, accum="f32"))
            x_b = rmsnorm(simulate_vendor_matmul(x_b.astype(np.float16), w, accum="f32", split_k=8))
    a, b = x_a.astype(np.float64), x_b.astype(np.float64)
    if relative:
        return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
    return float(np.abs(a - b).max())


def test_per_kernel_pass_but_model_diverges_numpy():
    # 実 numpy: 同じ許容内の単一カーネル発散が、深さで実際に累積することを実証。
    # per-kernel(1層)では小さい発散が、モデル(12層)では桁違いに育つ。
    n, K = 64, 64
    one_layer = _run_chain(1, n, K)
    deep = _run_chain(12, n, K)

    # (1) 合成で実測発散が深さとともに育つ（per-kernel 検証では見えない）
    assert deep > one_layer * 3, \
        f"model divergence should grow with depth: 1層={one_layer:.2e} 12層={deep:.2e}"
    # (2) 合成予測（深さ込み・safety 付き上界）は実測を上回る
    predicted = model_tolerance([GraphOp("matmul", K=K)] * 12)
    assert predicted >= deep, \
        f"propagated bound {predicted:.2e} should bound measured {deep:.2e}"
    # (3) 単一カーネル許容を「12 層分」誤って流用すると過小評価になる
    single = expected_gemm_abs_error(K, "float16", scale=1.0)
    assert predicted > single, "model-level tolerance must exceed single-kernel tolerance"


def test_residual_dilutes_model_divergence():
    # 残差トポロジ（skip 接続）は同じ深さでも発散を希釈する（一次モデル）。
    plain = [GraphOp("matmul", K=512) for _ in range(24)]
    resid = [GraphOp("matmul", K=512, residual=True) for _ in range(24)]
    mp = propagate(plain).model_divergence
    mr = propagate(resid).model_divergence
    assert mr < mp * 0.5            # 残差は線形累積よりずっと小さい
    # 単一ブロックでは差が出ない（深さ効果）
    one_p = propagate([GraphOp("matmul", K=512)]).model_divergence
    one_r = propagate([GraphOp("matmul", K=512, residual=True)]).model_divergence
    assert abs(one_p - one_r) < 1e-12


def test_residual_lower_than_plain_numpy():
    # 実 numpy（pre-norm）: 深い残差スタックの相対発散は平坦チェーンより小さい。
    n, K, L = 16, 64, 24
    plain = _run_chain(L, n, K, residual=False, relative=True)
    resid = _run_chain(L, n, K, residual=True, relative=True)
    assert resid < plain, f"residual {resid:.2e} should be < plain {plain:.2e}"


def test_empty_graph():
    assert model_tolerance([]) == 0.0


def test_merge_dilutes_uncorrelated_vs_correlated():
    # 合流則: 独立(random-walk)は二乗平均で希釈、相関(worst-case)は線形和
    divs = [0.1, 0.1, 0.1]
    assert abs(merge_divergence(divs, correlated=True) - 0.3) < 1e-12
    assert abs(merge_divergence(divs, correlated=False) - math.sqrt(0.03)) < 1e-12
    assert merge_divergence(divs, correlated=False) < merge_divergence(divs, correlated=True)
    assert merge_divergence([]) == 0.0


def test_dag_reduces_to_linear_chain():
    # propagate_dag は GraphOp だけの列なら propagate と一致（一般化が線形を包含）
    ops = [GraphOp("matmul", K=256), GraphOp("softmax", cond=4.0), GraphOp("matmul", K=128)]
    assert abs(propagate_dag(ops).model_divergence - propagate(ops).model_divergence) < 1e-18


def test_dag_identity_branch_carries_input_divergence():
    # 恒等(空)ブランチ＋f ブランチの合流は δ_in を運ぶ（skip 経路＝残差の DAG 表現）
    upstream = [GraphOp("matmul", K=256)]
    base = propagate(upstream).model_divergence
    fork = [*upstream, [[], [GraphOp("matmul", K=256)]]]
    merged = propagate_dag(fork, correlated=False).model_divergence
    assert merged >= base                      # 恒等枝が下界を与える
    # 相関ありはさらに大きい（保守側）
    merged_c = propagate_dag(fork, correlated=True).model_divergence
    assert merged_c >= merged


def _run_merge(n, K, seed=0):
    """2 ブランチ合流 y = matmul(x,w1)+matmul(x,w2) を 2 ベンダーで流し相対発散を返す。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, K)).astype(np.float16)
    w1 = (rng.standard_normal((K, K)) / np.sqrt(K)).astype(np.float16)
    w2 = (rng.standard_normal((K, K)) / np.sqrt(K)).astype(np.float16)
    ya = (simulate_vendor_matmul(x, w1, accum="f32")
          + simulate_vendor_matmul(x, w2, accum="f32"))
    yb = (simulate_vendor_matmul(x, w1, accum="f32", split_k=8)
          + simulate_vendor_matmul(x, w2, accum="f32", split_k=8))
    a, b = ya.astype(np.float64), yb.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))


def test_dag_branch_merge_bounds_measured_divergence_numpy():
    # 実 numpy: 2 つの計算ブランチが合流する DAG。線形列では表現できない位相を、
    # propagate_dag のフォーク→merge が（保守側 correlated で）実測発散を上界する。
    n, K = 64, 64
    measured = _run_merge(n, K)
    fork = [[[GraphOp("matmul", K=K)], [GraphOp("matmul", K=K)]]]
    predicted = propagate_dag(fork, correlated=True).model_divergence
    assert predicted >= measured, f"DAG bound {predicted:.2e} should bound {measured:.2e}"
    # 合流は trace に merge ノードを残す（可視性）
    assert any(o.kind.startswith("merge") for o in propagate_dag(fork).ops)


def test_only_genuine_relative_amplifiers():
    # 相対誤差を増幅するのは reduce/softmax/exp/layer_norm。div/reciprocal/add は相対 ~1。
    # layer_norm は平均優勢入力で amp≈RMS/σ に増幅（A-5 の数値実験で検証）。
    # rms_norm は無条件安定（≤1）だが 1 未満の減衰係数は入れないので非増幅・amp=1 固定。
    assert is_amplifier("reduce") and is_amplifier("exp") and is_amplifier("softmax")
    assert is_amplifier("layer_norm") and not is_amplifier("rms_norm")
    assert not is_amplifier("div") and not is_amplifier("reciprocal")
    assert not is_amplifier("add") and not is_amplifier("matmul")


def test_empirical_cond_is_data_driven():
    rng = np.random.default_rng(0)
    signed = rng.standard_normal((4, 256))
    positive = np.abs(rng.standard_normal((4, 256)))
    # 符号付き和は相殺で κ≫1、正の和は κ≈1（相殺なし）
    assert empirical_cond(signed, "reduce", axis=1) > 5.0
    assert empirical_cond(positive, "reduce", axis=1) < 1.5
    # max reduction は well-conditioned
    assert empirical_cond(signed, "reduce", axis=1, reduce_kind="max") == 1.0
    # exp の相対条件数は max|x|
    logits = rng.standard_normal((4, 16)) * 3
    assert abs(empirical_cond(logits, "exp") - float(np.abs(logits).max())) < 1e-9
    # div は相対では増幅しない
    assert empirical_cond(signed, "div") == 1.0


def test_empirical_cond_makes_amplification_fire():
    # 静的 cond=1 では amp=1 だが、実測 cond を入れると合成発散が naive 和を超える
    rng = np.random.default_rng(1)
    signed = rng.standard_normal((4, 512))
    kappa = empirical_cond(signed, "reduce", axis=1)
    ops = [GraphOp("matmul", K=256), GraphOp("reduce", cond=kappa)]
    rep = propagate(ops)
    assert rep.model_divergence > rep.naive_sum   # 実測 cond で増幅が発火


def test_depth_amplification_is_stable_across_seeds():
    """深さによる発散増大が seed に依らず頑健であることを実測で固定（SOCRATIC-50 Q48）。

    `docs/PERSPECTIVE-error-propagation.md` は「12 層で約 2000 倍」と書いていたが、
    これは *単一 seed の例示* であって統計ではなかった。多 seed で測ると:
      - 結論（1 層では無視できる発散が 2 桁以上に育つ）は seed に依らず頑健
      - **倍率そのものは seed で 2 倍以上ばらつく**（p10-p90 で ~1,600-3,300 倍）
    例示値を統計に置き換え、「約 2000 倍」が精密な定数でなく代表値であることを固定する。
    """
    from tsugi.equivalence import simulate_vendor_matmul

    def rmsnorm(x):
        return x / (np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True)) + 1e-6)

    def chain(seed, depth, N=64, K=64):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((N, K)).astype(np.float16)
        ws = [rng.standard_normal((K, K)).astype(np.float16) for _ in range(depth)]
        a = x.astype(np.float32)
        b = x.astype(np.float32)
        for w in ws:
            # 2 ベンダーの違いは累積順序（split_k）のみ
            a = rmsnorm(simulate_vendor_matmul(a.astype(np.float16), w, split_k=1))
            b = rmsnorm(simulate_vendor_matmul(b.astype(np.float16), w, split_k=4))
        return float(np.max(np.abs(a - b)))

    n_seeds = 15
    ratios = sorted(chain(s, 12) / max(chain(s, 1), 1e-30) for s in range(n_seeds))
    med = ratios[len(ratios) // 2]

    # 結論の頑健性: 全 seed で 2 桁以上の増大（これは seed に依らない）
    assert ratios[0] > 100.0, f"最小 seed でも 100 倍超のはず: {ratios[0]:.0f}"
    # 中央値は文書の記載レンジ（~1,600-3,300 倍）に収まる
    assert 1000.0 < med < 5000.0, f"中央値が想定レンジ外: {med:.0f}"
    # ばらつきの存在自体を固定（単一 seed を精密な定数として扱わない根拠）
    assert ratios[-1] / ratios[0] > 1.3, (
        f"seed 間のばらつきが消えている（実験構成の変化を疑う）: "
        f"{ratios[0]:.0f}-{ratios[-1]:.0f}")


# --- A-5: 正規化層の相対発散増幅（数値実験で検証してから導入） ---

def _layernorm(x, eps=1e-5):
    """実 LayerNorm（torch.nn.LayerNorm と同じ式・eps 既定値も同じ 1e-5）。"""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def _measured_norm_amp(norm_fn, x, seed=0, rel=1e-4, delta=None):
    """norm_fn を通した相対 RMS 発散の増幅率を実測する（δ_out/δ_in）。

    摂動の seed は入力 x の seed と **独立** でなければならない（+1000 のずらし）。
    同一 seed だと摂動が x の雑音成分と厳密に平行になり、LayerNorm がそれを完全に
    消してしまう（amp≈0）——スケール方向と平均方向は LN ヤコビアンの *零空間* だから。
    その落とし穴自体は test_layer_norm_jacobian_nullspace_is_annihilated で資産化した。
    delta を明示すれば任意方向の摂動を注入できる（零空間の実証に使う）。
    """
    rng = np.random.default_rng(seed + 1000)
    if delta is None:
        delta = rel * rng.standard_normal(x.shape) * np.sqrt(np.mean(x ** 2))
    d_in = np.sqrt(np.mean(delta ** 2)) / np.sqrt(np.mean(x ** 2))
    y, y2 = norm_fn(x), norm_fn(x + delta)
    d_out = np.sqrt(np.mean((y2 - y) ** 2)) / np.sqrt(np.mean(y ** 2))
    return d_out / d_in


def test_layer_norm_amplification_matches_jacobian_bound():
    """LayerNorm は平均優勢入力で相対発散を増幅し、RMS/√(σ²+eps) がそれを上界する。

    理論: y=(x−μ)/√(σ²+eps) のヤコビアンは J=(1/√(σ²+eps))(I−11ᵀ/d−ŷŷᵀ/d)——
    2 方向（平均・半径）の特異値が消え、最大特異値は 1/√(σ²+eps)。相対 RMS 増幅は
    δ_out/δ_in ≤ RMS(x)/√(σ²+eps) = 1/√(1−(μ/RMS)²) （eps→0）。零平均なら ≈1、
    平均優勢（μ/RMS→1）なら ≫1。旧モデル（norm→reduce・amp 実質 1）はこの増幅を
    見逃す偽OK だった——shift=10 で実測 amp≈10 に対し旧 cond≈1。
    """
    for shift in (0.0, 1.0, 3.0, 10.0):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((32, 512)) + shift
            meas = _measured_norm_amp(_layernorm, x, seed=seed)
            bound = empirical_cond(x, "layer_norm")
            assert meas <= bound * 1.05, \
                f"shift={shift} seed={seed}: 実測 {meas:.2f} > 上界 {bound:.2f}"
    # 増幅が実在する（旧 amp=1 が偽OK だった証拠）と、境界の挙動を固定
    x0 = np.random.default_rng(0).standard_normal((32, 512))
    x10 = x0 + 10.0
    assert empirical_cond(x0, "layer_norm") < 2.0          # 零平均: 増幅なし
    assert empirical_cond(x10, "layer_norm") > 5.0         # 平均優勢: 強い増幅
    assert _measured_norm_amp(_layernorm, x10) > 2.0       # 実測でも増幅が起きている


def test_layer_norm_jacobian_nullspace_is_annihilated():
    """LayerNorm ヤコビアンの 2 消失特異値を実測で示す（理論の直接検証）。

    J=(1/√(σ²+eps))(I−11ᵀ/d−ŷŷᵀ/d) は「平均方向（1）」と「半径＝スケール方向（ŷ）」の
    2 つを射影で落とす。よって **x に平行な摂動**（スケール変化）と **定数ベクトルの摂動**
    （平均シフト）は出力を一切変えない: LN((1+c)x + b·1) = LN(x)。
    一方、独立方向の摂動は RMS/σ 倍に増幅される。同じ x・同じ大きさの摂動でも
    「向き」で増幅が 0 倍にも 10 倍にもなる——増幅は入力の大きさだけでなく
    摂動の方向にも依存する（このテストは実際に踏んだ落とし穴の資産化）。
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 512)) + 10.0
    mag = 1e-4 * np.sqrt(np.mean(x ** 2))

    # (1) スケール方向（x に平行）→ 消える
    parallel = 1e-4 * x
    assert _measured_norm_amp(_layernorm, x, delta=parallel) < 0.01
    # (2) 平均方向（定数ベクトル）→ 消える
    mean_shift = np.full_like(x, mag)
    assert _measured_norm_amp(_layernorm, x, delta=mean_shift) < 0.01
    # (3) 独立方向 → RMS/σ 倍に増幅（上界の内側）
    generic = _measured_norm_amp(_layernorm, x)
    assert generic > 2.0
    assert generic <= empirical_cond(x, "layer_norm") * 1.05


def test_rms_norm_unconditional_stability_measured():
    """RMSNorm の相対増幅は無条件に ≤1（実測）——だが減衰係数(<1)は入れず amp=1 固定。

    J=(1/r)(I−ŷŷᵀ) は半径方向のみを落とす射影×縮小で、相対 RMS 増幅は厳密に ≤1
    （文献: unconditional forward stability・docs/SOURCES.md）。平均優勢入力でも
    増幅しない点が LayerNorm と決定的に違う（μ を引かないので σ でなく RMS で割る）。
    1 未満の係数を入れないのは未検証係数の禁止（設計ガードレール 2）——amp=1.0 は
    実測 ≤1 を決して過小評価しない保守側。
    """
    def rmsnorm(x, eps=1e-6):
        return x / (np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True)) + eps)

    for shift in (0.0, 3.0, 100.0):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((32, 512)) + shift
            assert _measured_norm_amp(rmsnorm, x, seed=seed) <= 1.05
    # モデル側: rms_norm は増幅も減衰もしない（cond を与えても厳密に 1.0）
    assert amplification(GraphOp("rms_norm", cond=100.0)) == 1.0
    assert amplification(GraphOp("layer_norm", cond=7.0)) == 7.0
    assert empirical_cond(np.zeros((4, 8)), "rms_norm") == 1.0


def test_empirical_cond_layer_norm_statistic():
    """layer_norm の cond は行ごとの RMS/√(σ²+eps) の max（median だと外れ行で偽OK）。"""
    rng = np.random.default_rng(0)
    zero_mean = rng.standard_normal((32, 512))
    assert empirical_cond(zero_mean, "layer_norm") < 2.0
    assert empirical_cond(zero_mean + 10.0, "layer_norm") > 5.0
    # 外れ行 1 本（massive activation 型）: 零平均 31 行 + 平均優勢 1 行。
    # median なら <2 に埋もれるが max は外れ行を反映する（偽OK 封じ）。
    mixed = zero_mean.copy()
    mixed[7] += 10.0
    rms = np.sqrt(np.mean(mixed ** 2, axis=-1))
    sd = np.sqrt(np.maximum(mixed.var(axis=-1), 0.0) + 1e-5)
    assert float(np.median(rms / sd)) < 2.0        # median は隠す
    assert empirical_cond(mixed, "layer_norm") > 5.0   # max は捕まえる
    # 定数行でも有限（eps ガード・inf/nan を出さない）
    const = np.full((4, 64), 7.0)
    c = empirical_cond(const, "layer_norm")
    assert np.isfinite(c) and c > 100.0            # 近定数行は強増幅として報告（保守側）


def test_layer_norm_model_prediction_bounds_measured_divergence_numpy():
    """matmul→LayerNorm 連鎖で「実測 cond 込みの予測 ≥ 実測発散」を両レジームで固定。

    偽OK ピン（受け入れ基準）: 平均優勢側は旧 reduce-cond≈1 で予測が実測を下回っていた
    （このテストを旧写像で走らせると赤）。零平均側は逆に旧 cond が ~20-30 に爆発して
    いた——新統計で予測は下がるが、それでも実測を上界することをここで保証する
    （下がっても偽OK にならないことの機械的な証拠）。
    """
    def chain(shift, L=4, n=32, K=64, seed=0):
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal((n, K)) + shift).astype(np.float16)
        a = b = x
        for i in range(L):
            w = (rng.standard_normal((K, K)) / np.sqrt(K)).astype(np.float16)
            a = _layernorm(simulate_vendor_matmul(a.astype(np.float16), w, accum="f32"))
            b = _layernorm(simulate_vendor_matmul(b.astype(np.float16), w,
                                                  accum="f32", split_k=8))
        af, bf = a.astype(np.float64), b.astype(np.float64)
        meas = float(np.linalg.norm(af - bf) / (np.linalg.norm(af) + 1e-30))
        cond = empirical_cond(np.asarray(x, dtype=np.float64), "layer_norm")
        ops = [GraphOp("matmul", K=K), GraphOp("layer_norm", cond=cond)] * L
        return model_tolerance(ops), meas

    for shift in (0.0, 10.0):
        pred, meas = chain(shift)
        assert pred >= meas, \
            f"shift={shift}: 予測 {pred:.2e} が実測 {meas:.2e} を下回る（偽OK）"


def main() -> int:
    ok = True
    tests = [
        test_depth_accumulates_divergence,
        test_naive_per_kernel_underestimates,
        test_ill_conditioned_op_is_dominant,
        test_per_kernel_pass_but_model_diverges_numpy,
        test_residual_dilutes_model_divergence,
        test_residual_lower_than_plain_numpy,
        test_empty_graph,
        test_merge_dilutes_uncorrelated_vs_correlated,
        test_dag_reduces_to_linear_chain,
        test_dag_identity_branch_carries_input_divergence,
        test_dag_branch_merge_bounds_measured_divergence_numpy,
        test_only_genuine_relative_amplifiers,
        test_empirical_cond_is_data_driven,
        test_empirical_cond_makes_amplification_fire,
        test_depth_amplification_is_stable_across_seeds,
        test_layer_norm_amplification_matches_jacobian_bound,
        test_layer_norm_jacobian_nullspace_is_annihilated,
        test_rms_norm_unconditional_stability_measured,
        test_empirical_cond_layer_norm_statistic,
        test_layer_norm_model_prediction_bounds_measured_divergence_numpy,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: 12 層 transformer 風グラフの合成発散
    ops = ([GraphOp("matmul", K=256), GraphOp("softmax", cond=4.0),
            GraphOp("matmul", K=256), GraphOp("add")] * 3)
    print("\n--- 12-ish op グラフの発散伝播 ---")
    print(propagate(ops).to_text())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
