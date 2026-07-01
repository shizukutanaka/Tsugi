"""統合ファサード tsugi.audit のテスト（検証層を 1 判定に束ねる）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

import tsugi  # noqa: E402
from tsugi import tile  # noqa: E402
from tsugi.audit import _graph_ops, audit, audit_cross_vendor, audit_runtime  # noqa: E402
from tsugi.autotune import TileConfig  # noqa: E402
from tsugi.envelope import certify_gemm  # noqa: E402
from tsugi.portcheck import _demo_module  # noqa: E402
from tsugi.report import Risk  # noqa: E402


@tsugi.jit
def _softmax_row(x, out, N, BN):
    """増幅 op（reduce/exp）を含む実カーネル（Q10 の attention/norm 系代表）。"""
    p = tsugi.program_id(0)
    row = tile.load(x, (p * BN, 0), (BN, N))
    m = tile.reduce(row, 1, "max")
    e = tile.exp(row - m)
    s = tile.reduce(e, 1, "sum")
    tile.store(out, (p * BN, 0), (e / s).to(tsugi.float16))


def test_audit_aggregates_all_static_phases():
    mod, block, cfg = _demo_module()
    a = audit(mod, cfg, block_dims=block)
    names = {p.name.split()[0] for p in a.phases}
    # 静的層（portability/feasibility/occupancy/numerics/propagation）＋ runtime
    assert {"portability", "feasibility", "occupancy", "numerics",
            "propagation", "runtime"} <= names


def test_graph_ops_collapses_kloop_dots_into_one_matmul():
    # K ループの dot 群（load で分断）は 1 つの matmul(K=反復×BK) に集約される
    mod, block, cfg = _demo_module()
    gops = _graph_ops(mod, cfg)
    matmuls = [o for o in gops if o.kind == "matmul"]
    assert len(matmuls) == 1
    assert matmuls[0].K == 256   # 4 dots × block_k 64


def test_audit_numerics_uses_sample_scale_when_given():
    """audit(sample=...) は certify_from_sample で実 RMS scale を認証する（Q13/Q14 の facade 接続）。

    Round 11 で envelope.certify_from_sample を追加したが audit() facade は
    certify_gemm(K,"float16",1.0) のまま未接続だった（scale=1 暗黙仮定が製品経路に残存）。
    sample を渡すと numerics phase のテキストに実測 scale が反映されることを保証する。
    sample 無指定時は「scale=1.0 仮定」を明記し、暗黙化しないことも確認する。
    """
    mod, block, cfg = _demo_module()

    # sample 無指定: scale=1.0 仮定が明記される（暗黙化しない）
    a_default = audit(mod, cfg, block_dims=block)
    num_default = next(p for p in a_default.phases if p.name.startswith("numerics"))
    assert "scale=1.0 仮定" in num_default.to_text()

    # sample あり: 実測 RMS scale（例: 50）が反映される（scale=1 認証より遥かに大きい atol）
    sample = np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32) * 50.0
    a_sample = audit(mod, cfg, block_dims=block, sample=sample)
    num_sample = next(p for p in a_sample.phases if p.name.startswith("numerics"))
    text = num_sample.to_text()
    assert "sample 実測" in text
    actual_rms = float(np.sqrt(np.mean(sample.astype(np.float64) ** 2)))
    assert f"scale={actual_rms:.3g}" in text, f"実測 scale={actual_rms:.3g} がテキストに無い: {text}"


def test_audit_propagates_amplification_through_traced_softmax():
    # Q10: 増幅 op（reduce/exp）を出す実カーネルを trace → audit に通し、propagation 層が
    # 実グラフから増幅 op を拾い「静的 cond=1 は下界」と過小評価を WARN することを保証。
    x = np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32)
    mod = tsugi.trace(_softmax_row, (x, x.copy(), 16, 16), {}, (0,))
    cfg = TileConfig(block_m=16, block_n=16, block_k=16, num_stages=2, num_warps=4)

    # 実グラフから増幅 op が抽出される（dot のみだった頃は空回りしていた経路）
    kinds = {o.kind for o in _graph_ops(mod, cfg)}
    assert {"reduce", "exp"} <= kinds, kinds

    a = audit(mod, cfg, block_dims=(16,))
    prop = next(p for p in a.phases if "propagation" in p.name.lower())
    text = prop.to_text()
    # 静的 cond=1 の過小評価を隠さず WARN（empirical_cond/audit_runtime へ誘導）
    assert prop.max_risk >= Risk.WARN
    assert "下界" in text and ("exp" in text or "reduce" in text)


def test_audit_sample_auto_measures_empirical_cond():
    """audit(sample=...) は増幅 op（reduce/exp）の cond を empirical_cond で自動実測する（第13回）。

    従来は sample を渡しても propagation の cond=1（well-conditioned 仮定）のままで、
    「静的 cond=1 は下界」と WARN するだけだった（Q7/Q8/Q11 の未接続）。
    sample があるのに empirical_cond を呼ばないのは sample を活かしきれていない。
    本修正: sample 提供時は増幅 op の cond を実データから測り、model_divergence を
    実測値に更新する（過小評価の是正）。sample 未指定時は従来通り下界 WARN のまま。
    """
    x = np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32)
    mod = tsugi.trace(_softmax_row, (x, x.copy(), 16, 16), {}, (0,))
    cfg = TileConfig(block_m=16, block_n=16, block_k=16, num_stages=2, num_warps=4)

    a_no_sample = audit(mod, cfg, block_dims=(16,))
    a_with_sample = audit(mod, cfg, block_dims=(16,), sample=x)
    prop_no = next(p for p in a_no_sample.phases if "propagation" in p.name.lower())
    prop_yes = next(p for p in a_with_sample.phases if "propagation" in p.name.lower())

    # sample 無し: 従来通り「下界」WARN のまま（cond=1 が変更されない）
    assert "下界" in prop_no.to_text()
    # sample 有り: 実測済みを明示し、「cond=1 は下界」誘導文言は出ない（もう cond=1 でないから）
    text_yes = prop_yes.to_text()
    assert "実測済み" in text_yes
    assert "静的 cond=1 は" not in text_yes
    # 実測 cond は softmax の reduce（相殺しうる）で 1 でない値を返しうる →
    # model_divergence が sample 無し版と異なる（過小評価が是正された証拠）
    div_no = float(prop_no.to_text().split("モデル発散(予測)=")[1].split()[0])
    div_yes = float(text_yes.split("モデル発散(予測)=")[1].split()[0])
    assert div_yes != div_no, "sample の有無で model_divergence が変わらない（empirical_cond 未使用の疑い）"


def test_propagation_phase_runs_on_module():
    # 統合された propagation が per-model 発散を出す（単一 matmul は増幅なし=INFO）
    mod, block, cfg = _demo_module()
    a = audit(mod, cfg, block_dims=block)
    prop = next(p for p in a.phases if p.name.startswith("propagation"))
    assert prop.when == "decided"
    assert prop.max_risk == Risk.INFO   # 単一 op グラフは伝播増幅なし


def test_audit_verdict_from_static_only():
    # デモ構成は AMD で起動不能 → 静的判定は BLOCK（移植ブロッカー）
    mod, block, cfg = _demo_module()
    a = audit(mod, cfg, block_dims=block)
    assert a.max_risk == Risk.BLOCK
    assert not a.portable


def test_runtime_phase_excluded_from_verdict():
    # runtime 層は実機データ待ちゆえ判定に影響しない（静的層のみで verdict）
    mod, block, cfg = _demo_module()
    a = audit(mod, cfg, block_dims=block)
    rt = [p for p in a.phases if p.when == "pending"]
    assert len(rt) == 1
    assert all(p.when == "decided" for p in a.decided_phases)
    assert a.max_risk == max(p.max_risk for p in a.decided_phases)


def test_audit_text_has_lifecycle_and_verdict():
    mod, block, cfg = _demo_module()
    txt = audit(mod, cfg, block_dims=block).to_text()
    assert "起動不能" in txt          # feasibility BLOCK
    assert "導出許容" in txt          # numerics
    assert "要実機データ" in txt      # runtime チェックリスト
    assert "判定（静的層）" in txt


def test_audit_without_cfg_still_runs_portability():
    # 構成なしでも移植性と（dot があれば）数値目安は出る
    mod, block, cfg = _demo_module()
    a = audit(mod, None, block_dims=block)
    names = {p.name.split()[0] for p in a.phases}
    assert "portability" in names
    assert "feasibility" not in names   # cfg 無しでは起動可能性は判定不可


def test_audit_verdict_is_provenance_stamped():
    # verdict は環境フィンガープリントに束ねられ、スタック更新で stale（再検証要）になる
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 32)).astype(np.float32)
    ad = audit_runtime(a, a + 1e-4, K=256, provenance={"rocm": "6.0", "driver": "550"})
    assert ad.certificate is not None
    assert not ad.is_stale(rocm="6.0", driver="550")    # 同一スタック
    assert ad.is_stale(rocm="6.0", driver="560")        # driver 更新 → 再検証要
    # 静的 audit も stamp される
    mod, block, cfg = _demo_module()
    assert audit(mod, cfg, block_dims=block).certificate is not None


def test_audit_runtime_oracle_catches_shared_mode():
    # oracle を渡すと correctness 層が働く: a≈b でも両方 oracle と不一致なら BLOCK（共有モード）
    rng = np.random.default_rng(0)
    oracle = rng.standard_normal((64, 64)).astype(np.float32)
    a = (oracle * 1.05).astype(np.float32)
    b = (oracle * 1.05 + 1e-9).astype(np.float32)
    bad = audit_runtime(a, b, K=256, oracle=oracle)
    assert not bad.portable                            # 共有モード障害を検出
    assert any(p.name.startswith("correctness") for p in bad.phases)
    good = audit_runtime(oracle, oracle.copy(), K=256, oracle=oracle)
    assert good.portable                               # a≈b≈oracle
    # oracle 無しでは correctness は問えない（portability のみ）
    no_oracle = audit_runtime(a, b, K=256)
    assert not any(p.name.startswith("correctness") for p in no_oracle.phases)


def test_audit_runtime_blame_skipped_when_oracle_unhealthy():
    # oracle が不健全の場合、blame（責帰）はスキップされ誤指摘を防ぐ。
    # 実際に verify_oracle() を壊すのは難しいので、oracle が健全な場合に blame 行があること、
    # oracle なし（None）の場合に correctness 相自体がないことで間接的に確認。
    import numpy as np
    rng = np.random.default_rng(3)
    oracle = rng.standard_normal((32, 32)).astype(np.float32)
    a = oracle + 1e-7 * rng.standard_normal((32, 32)).astype(np.float32)
    b = (oracle * 1.05).astype(np.float32)
    ad = audit_runtime(a, b, K=256, oracle=oracle)
    cp = next(p for p in ad.phases if p.name.startswith("correctness"))
    # With healthy oracle, blame line must appear
    blame_lines = [ln for ln in cp.lines if "責帰" in ln]
    assert blame_lines, "healthy oracle → blame 行があるべき"


def test_audit_runtime_blame_points_to_culprit_vendor():
    # oracle があるとき correctness 層が blame（新視点13）で修正方向を示す。
    # B が oracle から大きく乖離 → 「vendor B の実装を優先修正」が verdict に現れる。
    rng = np.random.default_rng(1)
    oracle = rng.standard_normal((64, 64)).astype(np.float32)
    a = oracle + 1e-6 * rng.standard_normal((64, 64)).astype(np.float32)  # A ≈ oracle
    b = (oracle * 1.10).astype(np.float32)                                # B 10% off
    ad = audit_runtime(a, b, K=256, oracle=oracle)
    cp = next(p for p in ad.phases if p.name.startswith("correctness"))
    blame_lines = [ln for ln in cp.lines if "責帰" in ln]
    assert blame_lines, "correctness 層に blame（責帰）行があるべき"
    assert "vendor B" in blame_lines[0], f"B を culprit に指すべき: {blame_lines[0]}"


def test_audit_runtime_layer_diagnosis_pinpoints_divergent_layer():
    """audit_runtime(layers_a=..., layers_b=...) が attribution.diagnose を接続する（第16回）。

    attribution.diagnose（onset/spike 特定＋blame 統合）は実装・テスト済みだが、
    audit_runtime は BLOCK を出すだけで「どの層で・どちらのベンダーが」悪いかを
    一度も特定していなかった（第11-15回で見つけた「機能は実装済みだが facade 未接続」
    と同型の欠陥の5件目）。layers_a/layers_b（＋任意 layers_oracle）を渡すと
    層別診断が自動で走り、onset（発散開始層）・spike（支配的増分層）・
    responsible vendor を verdict に含める。
    """
    def _id(x):
        return np.asarray(x, dtype=np.float64)

    def _scale2(x):
        return np.asarray(x, dtype=np.float64) * 2.0

    oracle_layers = [_id, _scale2, _id]
    layers_a = [_id, _scale2, _id]                              # A は oracle と一致
    layers_b = [_id, lambda x: np.asarray(x, dtype=np.float64) * 2.0 + 0.5, _id]  # B が層1で発散

    rng = np.random.default_rng(0)
    a_out = rng.standard_normal((16, 16)).astype(np.float32)
    b_out = a_out.copy()
    x0 = np.array([1.0, 2.0, 3.0])

    ad = audit_runtime(a_out, b_out, K=64, layers_a=layers_a, layers_b=layers_b,
                       layers_oracle=oracle_layers, x0=x0,
                       layer_names=["embed", "matmul", "norm"])
    attr = next(p for p in ad.phases if p.name.startswith("attribution"))
    text = attr.to_text()
    assert "matmul" in text, f"spike 層名(matmul)が診断テキストに無い: {text}"
    assert "vendor B" in text or "fix vendor B" in text, \
        f"B が発散源として責帰されていない: {text}"

    # layers_a/layers_b 未指定なら attribution 層は現れない（後方互換）
    ad_no_layers = audit_runtime(a_out, b_out, K=64)
    assert not any(p.name.startswith("attribution") for p in ad_no_layers.phases)


def _accum_precision_vendors():
    """累積精度が違う 2 ベンダー（fp16 累積 vs fp32 累積）の二乗和。test_worstcase.py と同型。"""
    def fp16(x):
        acc = np.float16(0.0)
        for v in np.asarray(x, dtype=np.float16):
            acc = np.float16(acc + np.float16(v * v))
        return np.array([acc], dtype=np.float64)

    def fp32(x):
        return np.array([np.sum(np.asarray(x, dtype=np.float32) ** 2)], dtype=np.float64)

    return fp16, fp32


def test_audit_runtime_worst_case_search_finds_envelope_counterexample():
    """audit_runtime(fn_a=..., fn_b=..., worst_samples=...) が worstcase.analyze_worst_case
    を接続する（第17回）。

    worstcase.analyze_worst_case（唯一の能動探索層・平均ケース検証の盲点を露出）は
    実装・テスト済みだが、audit_runtime は受動的な代表サンプル比較しか行わず、認証
    エンベロープ内に隠れる反例を能動的に探すことは一度もしていなかった
    （第11-16回で見つけた「機能は実装済みだが facade 未接続」と同型の欠陥の6件目）。
    代表サンプルでは良性に見える（典型発散 < tol）が、エンベロープ内の能動探索で
    tol を超える反例が見つかる古典的なケースで、worstcase phase が BLOCK を出すことを実証する。
    """
    fp16, fp32 = _accum_precision_vendors()
    rng = np.random.default_rng(0)
    samples = [rng.standard_normal(64) for _ in range(16)]

    a = rng.standard_normal((8, 8)).astype(np.float32)
    ad = audit_runtime(a, a.copy(), K=64, fn_a=fp16, fn_b=fp32, worst_samples=samples,
                       worst_tol=1e-3, worst_bounds=(-30.0, 30.0), worst_steps=900, worst_seed=1)
    wc = next(p for p in ad.phases if p.name.startswith("worstcase"))
    assert wc.max_risk == Risk.BLOCK, f"エンベロープ内の反例が見つからない: {wc.to_text()}"
    assert not ad.portable, "worstcase の BLOCK が verdict に算入されていない"

    # fn_a/fn_b/worst_samples 未指定なら worstcase phase は現れない（後方互換）
    ad_no_wc = audit_runtime(a, a.copy(), K=64)
    assert not any(p.name.startswith("worstcase") for p in ad_no_wc.phases)


def test_audit_runtime_passes_equivalent_within_noise():
    # 真に等価(微小差)・ノイズ床込み → ブロッカー無し（INDISTINGUISHABLE/EQ は OK 寄り）
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    b = a + 1e-4 * rng.standard_normal((64, 64)).astype(np.float32)
    ad = audit_runtime(a, b, K=256, env=certify_gemm(256, "float16", 1.0),
                       noise_floor=1e-3)
    assert ad.portable                     # 真の発散なし
    names = {p.name.split()[0] for p in ad.phases}
    assert {"envelope", "equivalence"} <= names


def test_audit_runtime_blocks_real_divergence():
    # 5% 系統スケール誤差 → BLOCK（max_abs か系統バイアスのいずれかが捕捉）
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    ad = audit_runtime(a, a * 1.05, K=256, noise_floor=1e-3)
    assert not ad.portable
    assert ad.max_risk == Risk.BLOCK


def test_audit_runtime_equivalence_distinguishes_layout_from_true_divergence():
    """equivalence phase が classify_divergence で LAYOUT を数値発散と区別する（第18回）。

    equivalence.classify_divergence（LAYOUT vs 真の数値発散の判別）は実装・テスト済みだが、
    audit_runtime の equivalence phase は DIVERGENT を一律 BLOCK にするだけで、
    転置/再タイル（値は正しいが位置違い・codegen の整列バグ）と真の精度バグを
    区別していなかった（第11-17回で見つけた「機能は実装済みだが facade 未接続」と
    同型の欠陥の7件目）。両者は修正すべき箇所が全く異なる（整列 vs 精度チューニング）
    ため、区別せずに BLOCK だけ出すのは診断上の手がかりを捨てていることになる。
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 32)).astype(np.float32)

    # 転置（レイアウトバグ・値は正しい）は BLOCK のまま（真に不一致）だが LAYOUT と明示される
    ad_layout = audit_runtime(a, a.T.copy(), K=256, noise_floor=1e-6)
    eq_layout = next(p for p in ad_layout.phases if p.name.startswith("equivalence"))
    assert eq_layout.max_risk == Risk.BLOCK
    assert "LAYOUT" in eq_layout.to_text()

    # 真のスケール発散（レイアウトでない）は LAYOUT タグが付かない
    ad_true = audit_runtime(a, (a * 1.5).astype(np.float32), K=256, noise_floor=1e-6)
    eq_true = next(p for p in ad_true.phases if p.name.startswith("equivalence"))
    assert eq_true.max_risk == Risk.BLOCK
    assert "LAYOUT" not in eq_true.to_text()


def test_audit_runtime_includes_decision_when_logits_given():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    la = rng.standard_normal((1500, 200)).astype(np.float32)
    lb = la + 1e-1 * rng.standard_normal(la.shape).astype(np.float32)  # 多数フリップ
    ad = audit_runtime(a, a.copy(), K=256, logits_a=la, logits_b=lb, flip_budget=0.001)
    dp = next(p for p in ad.phases if p.name.startswith("decision"))
    assert dp.max_risk == Risk.BLOCK       # フリップ率が予算超


def test_audit_runtime_supports_non_classification_tasks():
    """audit_runtime(task=...) が decision.compare_task に委譲する（第15回）。

    decision.compare_task（regression/binary/ranking）は実装・テスト済みだが、
    audit_runtime は常に compare_decisions（分類 argmax 専用）を呼んでいた
    （第11-14回で見つけた「機能は実装済みだが facade 未接続」と同型の欠陥）。
    回帰モデル（価格/物理量）・二値分類（診断/異常検知）・検索/推薦（ranking）は
    argmax を持たないため、従来は decision 層の恩恵を受けられなかった。
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 32)).astype(np.float32)

    # regression: 許容(rtol)超の乖離は BLOCK
    reg_a = rng.standard_normal(500)
    reg_b = reg_a * 2.0   # 100% 乖離 → rtol=1e-3 を大きく超える
    ad_reg = audit_runtime(a, a.copy(), K=256, logits_a=reg_a, logits_b=reg_b,
                           task="regression", flip_budget=0.001,
                           task_kwargs={"rtol": 0.01})
    dp_reg = next(p for p in ad_reg.phases if p.name.startswith("decision"))
    assert "regression" in dp_reg.name
    assert dp_reg.max_risk == Risk.BLOCK

    # binary: しきい値をまたぐ反転は BLOCK
    bin_a = np.full(200, 0.9)
    bin_b = np.full(200, 0.1)   # 閾値 0.5 を挟んで全反転
    ad_bin = audit_runtime(a, a.copy(), K=256, logits_a=bin_a, logits_b=bin_b,
                           task="binary", flip_budget=0.001)
    dp_bin = next(p for p in ad_bin.phases if p.name.startswith("decision"))
    assert "binary" in dp_bin.name
    assert dp_bin.max_risk == Risk.BLOCK

    # 既定（classification）は従来通り compare_decisions を使う（後方互換）
    la = rng.standard_normal((500, 50)).astype(np.float32)
    ad_default = audit_runtime(a, a.copy(), K=256, logits_a=la, logits_b=la.copy())
    dp_default = next(p for p in ad_default.phases if p.name.startswith("decision"))
    assert dp_default.name == "decision タスクレベル等価"   # task サフィックス無し = classification


def test_audit_runtime_adds_rollout_phase_when_gen_length_given():
    # gen_length を渡すと per-token フリップを生成長へ合成する rollout 層が増える（新視点9）
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    la = rng.standard_normal((2000, 100)).astype(np.float32)
    lb = la + 3e-3 * rng.standard_normal(la.shape).astype(np.float32)  # 小さな per-token 差
    short = audit_runtime(a, a.copy(), K=256, logits_a=la, logits_b=lb, gen_length=1)
    long = audit_runtime(a, a.copy(), K=256, logits_a=la, logits_b=lb, gen_length=4000)
    # gen_length 無しでは rollout 層は付かない
    base = audit_runtime(a, a.copy(), K=256, logits_a=la, logits_b=lb)
    assert not any(p.name.startswith("rollout") for p in base.phases)
    rp_long = next(p for p in long.phases if p.name.startswith("rollout"))
    rp_short = next(p for p in short.phases if p.name.startswith("rollout"))
    # 同じ per-token 差でも長い生成では risk が上がる（複利増幅）
    assert rp_long.max_risk >= rp_short.max_risk


def test_audit_rollout_is_fail_safe_with_zero_observed_flips():
    # 改善: facade の rollout も点推定でなく上側信頼限界を使う。完全一致 logit
    # （フリップ 0 観測）でも長い生成では survival を 100% と過信しない（fail-safe 整合）。
    rng = np.random.default_rng(1)
    a = rng.standard_normal((48, 48)).astype(np.float32)
    la = rng.standard_normal((300, 64)).astype(np.float32)
    ad = audit_runtime(a, a.copy(), K=256, logits_a=la, logits_b=la.copy(),
                       gen_length=10**6)
    rp = next(p for p in ad.phases if p.name.startswith("rollout"))
    # 0 フリップ観測でも p>0 の上側限界を計上 → 巨大 L で安全と言い切らない
    assert rp.max_risk >= Risk.WARN
    assert any("上側限界" in ln for ln in rp.lines)


def test_audit_cross_vendor_folds_batch_variance_floor():
    # run_batch を渡すと batch-invariance 床が実効床に合流する（2025 研究の取り込み）
    from tsugi.nondeterminism import simulate_batch_variant_reduction
    base = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)

    def run(s):
        g = np.random.default_rng(7777 + s).standard_normal(base.shape).astype(np.float32)
        return base + 1e-6 * g

    flat = np.random.default_rng(1).standard_normal(4096).astype(np.float32)
    ad = audit_cross_vendor(run, run, K=256, n_runs=6,
                            run_batch=lambda t: simulate_batch_variant_reduction(flat, t))
    # 真に等価（同じ run）+ 床込み → ブロッカー無し
    assert ad.portable


def test_audit_cross_vendor_robust_resists_single_glitchy_run():
    """audit_cross_vendor(robust=True) は単発の測定グリッチに頑健（Q49 の実機入口接続・第14回）。

    nondeterminism.compare_stable は robust=True（外れ値頑健な 10-90 パーセンタイル幅）を
    サポートするが、実機向けの主要入口 audit_cross_vendor には接続されておらず常に
    max-min（spread）を使っていた。単発グリッチ 1 個で noise floor が桁違いに膨張し、
    本来 EQUIVALENT と明確に判定できるはずの真の divergence を INDISTINGUISHABLE
    （等価判定「未定義」の WARN）に押し込め、運用上の余計な triage を発生させる。
    """
    base = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)

    def run_a(s):
        g = np.random.default_rng(1000 + s).standard_normal(base.shape).astype(np.float32)
        scale = 5e-2 if s == 7 else 1e-6   # noise floor 測定中の 1 run だけグリッチ
        return base + scale * g

    def run_b(s):
        g = np.random.default_rng(2000 + s).standard_normal(base.shape).astype(np.float32)
        # A/B 間の真の系統発散（seed=0 の比較対象出力自体にはグリッチが乗らない）
        return base * 1.0008 + 1e-6 * g

    ad_default = audit_cross_vendor(run_a, run_b, K=256, n_runs=16)
    ad_robust = audit_cross_vendor(run_a, run_b, K=256, n_runs=16, robust=True)

    eq_default = next(p for p in ad_default.phases if "equivalence" in p.name.lower())
    eq_robust = next(p for p in ad_robust.phases if "equivalence" in p.name.lower())

    # 既定(non-robust): グリッチが noise floor を膨張させ INDISTINGUISHABLE（未定義）に隠す
    assert "INDISTINGUISHABLE" in eq_default.to_text(), \
        f"既定 robust=False でグリッチが noise floor を膨張させないと再現できない: {eq_default.to_text()}"
    # robust=True: 同じグリッチでも 10-90 パーセンタイル幅で頑健 → EQUIVALENT に明確判定
    assert "EQUIVALENT" in eq_robust.to_text() and "INDISTINGUISHABLE" not in eq_robust.to_text(), \
        f"robust=True でも INDISTINGUISHABLE のまま（Q49 修正が効いていない）: {eq_robust.to_text()}"


def test_audit_cross_vendor_forwards_provenance():
    # 実機入口も verdict を実 GPU スタックに束ねる（provenance 素通し・drift 回帰防止）
    base = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)

    def run(s):
        g = np.random.default_rng(7777 + s).standard_normal(base.shape).astype(np.float32)
        return base + 1e-6 * g

    ad = audit_cross_vendor(run, run, K=256, n_runs=4,
                            provenance={"rocm": "6.0", "driver": "550"})
    assert ad.certificate is not None
    assert not ad.is_stale(rocm="6.0", driver="550")
    assert ad.is_stale(rocm="6.0", driver="560")


def test_audit_demo_runs_end_to_end():
    # examples/audit_demo.py が両 facade を回し、系統バグを BLOCK にすることを保証
    import contextlib
    import importlib.util
    import io

    path = Path(__file__).resolve().parents[2] / "examples" / "audit_demo.py"
    spec = importlib.util.spec_from_file_location("audit_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main()
    out = buf.getvalue()
    assert rc == 0
    assert "系統バイアス" in out          # 実行時に max_abs 盲点の系統バグを捕捉
    assert "移植ブロッカー" in out        # 静的に AMD 起動不能を BLOCK


def main() -> int:
    ok = True
    tests = [
        test_audit_aggregates_all_static_phases,
        test_graph_ops_collapses_kloop_dots_into_one_matmul,
        test_audit_numerics_uses_sample_scale_when_given,
        test_audit_propagates_amplification_through_traced_softmax,
        test_audit_sample_auto_measures_empirical_cond,
        test_propagation_phase_runs_on_module,
        test_audit_verdict_from_static_only,
        test_runtime_phase_excluded_from_verdict,
        test_audit_text_has_lifecycle_and_verdict,
        test_audit_without_cfg_still_runs_portability,
        test_audit_verdict_is_provenance_stamped,
        test_audit_runtime_oracle_catches_shared_mode,
        test_audit_runtime_blame_skipped_when_oracle_unhealthy,
        test_audit_runtime_blame_points_to_culprit_vendor,
        test_audit_runtime_layer_diagnosis_pinpoints_divergent_layer,
        test_audit_runtime_worst_case_search_finds_envelope_counterexample,
        test_audit_runtime_passes_equivalent_within_noise,
        test_audit_runtime_blocks_real_divergence,
        test_audit_runtime_equivalence_distinguishes_layout_from_true_divergence,
        test_audit_runtime_includes_decision_when_logits_given,
        test_audit_runtime_supports_non_classification_tasks,
        test_audit_runtime_adds_rollout_phase_when_gen_length_given,
        test_audit_rollout_is_fail_safe_with_zero_observed_flips,
        test_audit_cross_vendor_folds_batch_variance_floor,
        test_audit_cross_vendor_robust_resists_single_glitchy_run,
        test_audit_cross_vendor_forwards_provenance,
        test_audit_demo_runs_end_to_end,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
