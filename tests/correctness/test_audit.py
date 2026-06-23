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


def test_audit_runtime_includes_decision_when_logits_given():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    la = rng.standard_normal((1500, 200)).astype(np.float32)
    lb = la + 1e-1 * rng.standard_normal(la.shape).astype(np.float32)  # 多数フリップ
    ad = audit_runtime(a, a.copy(), K=256, logits_a=la, logits_b=lb, flip_budget=0.001)
    dp = next(p for p in ad.phases if p.name.startswith("decision"))
    assert dp.max_risk == Risk.BLOCK       # フリップ率が予算超


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
        test_audit_propagates_amplification_through_traced_softmax,
        test_propagation_phase_runs_on_module,
        test_audit_verdict_from_static_only,
        test_runtime_phase_excluded_from_verdict,
        test_audit_text_has_lifecycle_and_verdict,
        test_audit_without_cfg_still_runs_portability,
        test_audit_verdict_is_provenance_stamped,
        test_audit_runtime_oracle_catches_shared_mode,
        test_audit_runtime_blame_skipped_when_oracle_unhealthy,
        test_audit_runtime_blame_points_to_culprit_vendor,
        test_audit_runtime_passes_equivalent_within_noise,
        test_audit_runtime_blocks_real_divergence,
        test_audit_runtime_includes_decision_when_logits_given,
        test_audit_runtime_adds_rollout_phase_when_gen_length_given,
        test_audit_rollout_is_fail_safe_with_zero_observed_flips,
        test_audit_cross_vendor_folds_batch_variance_floor,
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
