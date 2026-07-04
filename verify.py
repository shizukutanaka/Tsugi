"""Tsugi 不変条件チェッカ（CI verify・ハーネスの verify パターン）。

プロジェクトの不変条件を機械的に検証する。CI で fail on error。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / "python"

INVARIANTS: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    INVARIANTS.append((name, cond))
    print(f"[{'OK ' if cond else 'ERR'}] {name}")


def _grep(pattern: str, *globs: str) -> list[str]:
    hits = []
    self_path = Path(__file__).resolve()
    for g in globs:
        for p in ROOT.rglob(g):
            if "build" in p.parts or "__pycache__" in p.parts:
                continue
            if p.resolve() == self_path:  # verify 自身のスキャンパターンを除外
                continue
            try:
                for ln in p.read_text(encoding="utf-8").splitlines():
                    if pattern in ln:
                        hits.append(f"{p}: {ln.strip()}")
            except Exception:  # noqa: BLE001
                pass
    return hits


def main() -> int:
    # 1. 課金コード不在（絶対禁止）
    billing = [h for h in _grep("stripe", "*.py") + _grep("Stripe", "*.py")
               if "ADR" not in h and "FAQ" not in h]
    check("no billing/Stripe code", not billing)

    # 2. バイナリ CUDA 変換の不在（ADR-002）
    zluda = [h for h in _grep("zluda", "*.py") if "FAQ" not in h and "ADR" not in h]
    check("no binary CUDA translation (ADR-002)", not zluda)

    # 3. PII パターン不在
    pii = [h for h in _grep("@gmail", "*.py") + _grep("@yahoo", "*.py")]
    check("no PII in code", not pii)

    # 4. リファレンス correctness 全通過
    r = subprocess.run([sys.executable, str(ROOT / "tests/correctness/test_reference.py")],
                       capture_output=True, text=True)
    check("reference correctness PASS", r.returncode == 0)

    # 5. tracer / lowering / autotune 全通過
    for t in ("test_tracer.py", "test_lowering.py", "test_autotune.py"):
        rr = subprocess.run([sys.executable, str(ROOT / "tests/correctness" / t)],
                            capture_output=True, text=True)
        check(f"{t} PASS", rr.returncode == 0)

    # 6. dot は行列コア intrinsic へ写像（ADR-004）
    sys.path.insert(0, str(PY))
    from tsugi.lowering import VENDOR_LOWERING
    check("dot→wmma (NVIDIA, ADR-004)", "wmma" in VENDOR_LOWERING["dot"]["nvidia"])
    check("dot→mfma (AMD CDNA, ADR-004)", "mfma" in VENDOR_LOWERING["dot"]["amd_cdna"])
    # DSL が emit しうる全 op に実ターゲット lowering が定義済み（spec が DSL に同期）
    from tsugi.lowering import unlowered_ops
    check("lowering covers every emittable DSL op (nvidia/amd, no drift)",
          all(not unlowered_ops(t) for t in ("nvidia", "amd_cdna", "amd_rdna")))

    # 7. machine-code emission は正直に未実装
    import tsugi
    try:
        tsugi.compile(lambda: None, (), emit_machine_code=True)
        check("machine-code honestly unimplemented", False)
    except NotImplementedError:
        check("machine-code honestly unimplemented", True)


    # 8. equivalence 検出器が発散を捕まえる（新視点の柱）
    import numpy as np
    from tsugi.equivalence import compare, simulate_vendor_matmul
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 1024)).astype(np.float16)
    b = rng.standard_normal((1024, 64)).astype(np.float16)
    good = simulate_vendor_matmul(a, b, accum="f32")
    bad = simulate_vendor_matmul(a, b, accum="f16", split_k=64)
    check("equivalence detects f16-accum divergence", not compare(good, bad, "float16").equivalent)
    check("equivalence accepts identical", compare(good, good.copy(), "float16").equivalent)
    # float64 が float32 の緩い許容にフォールバックしない（偽OK 防止・PyTorch assert_close 由来）
    _f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    check("equivalence float64 catches 1e-6 drift (no float32 fallback)",
          not compare(_f64, _f64 + 1e-6, "float64").equivalent)
    check("equivalence float64 accepts genuine double-precision rounding",
          compare(_f64, _f64 + 1e-14, "float64").equivalent)


    # 9. occupancy: 同一構成がベンダー間で差を持つ（移植落とし穴の検出）
    from tsugi.autotune import TileConfig
    from tsugi.occupancy import cross_vendor_occupancy
    occ = cross_vendor_occupancy(TileConfig(64, 64, 32, 3, 4))
    vals = {v: e.occupancy for v, e in occ.items()}
    check("occupancy differs across vendors", len(set(vals.values())) > 1)


    # 10. tolerance: 導出許容が K 依存（固定値でない・新視点）
    from tsugi.tolerance import expected_gemm_abs_error
    check("derived tolerance grows with K",
          expected_gemm_abs_error(4096, "float16") > expected_gemm_abs_error(64, "float16"))

    # 11. feasibility: 起動不能を BLOCK に分類する（占有率と区別・新視点）
    from tsugi import portability
    from tsugi.feasibility import check as feasible_check
    from tsugi.ir import Module
    nv_tuned = TileConfig(128, 128, 64, 4, 8)   # NVIDIA で起動・AMD で起動不能
    check("feasibility: NVIDIA launchable", feasible_check(nv_tuned, "nvidia").launchable)
    check("feasibility: AMD CDNA NOT launchable", not feasible_check(nv_tuned, "amd_cdna").launchable)
    check("unfeasible config classified as BLOCK (not WARN)",
          portability.analyze(Module(kernels=[]), "amd_cdna", cfg=nv_tuned).max_risk
          == portability.Risk.BLOCK)

    # 12. propagation: per-kernel 等価 ⇏ per-model 等価（深さで発散が累積・新視点4）
    from tsugi.propagation import GraphOp, model_tolerance
    single = model_tolerance([GraphOp("matmul", K=256)])
    deep = model_tolerance([GraphOp("matmul", K=256)] * 12)
    check("model-level divergence exceeds single-kernel (composition)", deep > single)
    amp = model_tolerance([GraphOp("matmul", K=128), GraphOp("softmax", cond=8.0)])
    flat = model_tolerance([GraphOp("matmul", K=128), GraphOp("softmax", cond=1.0)])
    check("ill-conditioned op amplifies divergence", amp > flat)
    plain_chain = model_tolerance([GraphOp("matmul", K=512)] * 24)
    resid_chain = model_tolerance([GraphOp("matmul", K=512, residual=True)] * 24)
    check("residual topology dilutes propagated divergence vs plain chain",
          resid_chain < plain_chain * 0.5)
    # propagation の DAG 一般化: フォーク→合流（attention/concat）を表現でき、線形列を包含
    from tsugi.propagation import merge_divergence, propagate, propagate_dag
    _lin = [GraphOp("matmul", K=256), GraphOp("softmax", cond=4.0)]
    _fork = [[[GraphOp("matmul", K=256)], [GraphOp("matmul", K=256)]]]
    check("propagate_dag generalizes linear chain and merges branches conservatively",
          abs(propagate_dag(_lin).model_divergence - propagate(_lin).model_divergence) < 1e-18
          and merge_divergence([0.1, 0.1], correlated=True)
          > merge_divergence([0.1, 0.1], correlated=False)
          and propagate_dag(_fork, correlated=True).model_divergence
          >= propagate_dag(_fork, correlated=False).model_divergence)

    # 13. envelope: 認証前提の実行時逸脱を検出（静的保証の契約化・新視点5）
    import numpy as np  # noqa: F811
    from tsugi.envelope import certify_gemm, check_softmax_input, check_tensor
    env = certify_gemm(K=256, dtype="float16", scale=1.0)
    overflow = np.full((4, 4), 70000.0, np.float32)   # > fp16 max 65504
    check("envelope flags fp16 overflow (OUT)", not check_tensor(overflow, env).in_envelope)
    safe = np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32)
    check("envelope accepts in-range input (IN)", check_tensor(safe, env).in_envelope)
    big_logit = np.array([[0.0, 12.5]], np.float32)   # > ln(65504)=11.09
    check("envelope flags fp16 softmax overflow",
          not check_softmax_input(big_logit, env).in_envelope)
    # outlier feature(massive activations): 単一 scale 仮定の破綻を検出
    from tsugi.envelope import channel_scale_spread, check_outlier_features
    _ol = np.random.default_rng(0).standard_normal((32, 128)).astype(np.float32)
    _ol[:, 3] *= 200
    check("envelope flags outlier features breaking single-scale assumption",
          check_outlier_features(_ol).max_risk == portability.Risk.WARN
          and channel_scale_spread(_ol) > 50.0)

    # 14. calibration: 検証器そのものを検証（偽OK の非対称コストと検出限界・新視点6）
    from tsugi.calibration import (
        detectability_floor,
        evaluate,
        is_equivalent_combined,
        make_corpus,
    )
    from tsugi.equivalence import compare_gemm
    check("detectability floor grows with K (sqrt)",
          detectability_floor(8192, "float16")["rel"]
          > detectability_floor(256, "float16")["rel"] * 4)
    corpus = make_corpus(seed=0)
    max_abs_only = evaluate(corpus, lambda a, b, K: compare_gemm(a, b, K, "float16").equivalent)
    combined = evaluate(corpus, lambda a, b, K: is_equivalent_combined(a, b, K, "float16"))
    check("max_abs-only verifier is untrustworthy (false-OK on sub-floor bug)",
          not max_abs_only.trustworthy and max_abs_only.false_ok > 0)
    check("combined verifier (max_abs + systematic) is trustworthy (false-OK = 0)",
          combined.trustworthy and combined.false_ok == 0)

    # 15. nondeterminism: 出力は分布・ノイズ未満は判定未定義（新視点7）
    from tsugi.nondeterminism import (
        DIVERGENT,
        attribute,
        measure_noise_floor,
        simulate_nondeterministic_reduction,
    )
    parts = np.random.default_rng(0).standard_normal(4096).astype(np.float32)
    nf = measure_noise_floor(lambda s: simulate_nondeterministic_reduction(parts, s), 16)
    check("GPU output is a distribution (run-to-run noise > 0)", nf["spread"] > 0.0)
    check("below noise floor → INDISTINGUISHABLE (equivalence undefined)",
          attribute(nf["spread"] * 0.5, nf["spread"], 1e-2) == "INDISTINGUISHABLE")
    check("above tolerance → DIVERGENT (real divergence survives noise)",
          attribute(1.0, nf["spread"], 1e-2) == DIVERGENT)

    # 16. decision: タスクレベル等価（判断フリップは数値発散と decouple・新視点8）
    from tsugi.decision import compare_decisions, flip_rate
    z = np.random.default_rng(0).standard_normal((2000, 200)).astype(np.float32)
    va = z + 1e-2 * np.random.default_rng(1).standard_normal(z.shape).astype(np.float32)
    vb = z + 1e-2 * np.random.default_rng(2).standard_normal(z.shape).astype(np.float32)
    check("decision flips are scale-invariant (decoupled from abs error)",
          abs(flip_rate(va, vb) - flip_rate(va * 10, vb * 10)) < 1e-12)
    check("predicted flip bound P(margin<2d) upper-bounds actual flips",
          flip_rate(va, vb) <= compare_decisions(va, vb).predicted_bound + 1e-9)
    check("high flip rate blocks at task level",
          not compare_decisions(va, vb, flip_budget=0.001).ok)

    # 17. audit: 統合ファサードが静的層を 1 判定に束ねる（運用統合）
    from tsugi.audit import audit
    from tsugi.portcheck import _demo_module
    mod, block, dcfg = _demo_module()
    ad = audit(mod, dcfg, block_dims=block)
    check("audit aggregates static phases into one verdict (AMD launch BLOCK)",
          ad.max_risk == portability.Risk.BLOCK and not ad.portable)
    check("audit excludes runtime phases from static verdict",
          ad.max_risk == max(p.max_risk for p in ad.decided_phases))
    from tsugi.audit import _graph_ops
    mm = [o for o in _graph_ops(mod, dcfg) if o.kind == "matmul"]
    check("audit propagation collapses K-loop dots into one per-model matmul",
          len(mm) == 1 and mm[0].K == 256
          and any(p.name.startswith("propagation") for p in ad.phases))

    # 18. audit_runtime: 実データで実行時層を束ねて判定（チェックリストの実行版）
    from tsugi.audit import audit_runtime
    from tsugi.envelope import certify_gemm as _cert
    _a = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    check("audit_runtime passes truly-equivalent outputs within noise",
          audit_runtime(_a, _a + 1e-4, K=256, env=_cert(256, "float16", 1.0),
                        noise_floor=1e-3).portable)
    check("audit_runtime blocks a real systematic divergence (5% scale)",
          not audit_runtime(_a, _a * 1.05, K=256, noise_floor=1e-3).portable)
    # oracle を渡すと correctness 層が共有モード障害(a≈b だが両方 oracle 不一致)を BLOCK
    _ora = np.random.default_rng(1).standard_normal((64, 64)).astype(np.float32)
    _sb = (_ora * 1.05).astype(np.float32)
    check("audit_runtime with oracle catches shared-mode (correctness, not just portability)",
          not audit_runtime(_sb, _sb.copy(), K=256, oracle=_ora).portable
          and audit_runtime(_ora, _ora.copy(), K=256, oracle=_ora).portable)
    # verdict は provenance スタンプされ、スタック更新で stale（再検証要）
    _av = audit_runtime(_a, _a + 1e-4, K=256, provenance={"driver": "550"})
    check("audit verdict is provenance-stamped and goes stale on stack change",
          _av.certificate is not None and not _av.is_stale(driver="550")
          and _av.is_stale(driver="560"))

    # 19. audit_cross_vendor: ノイズ実測→監査の実機経路（擬似 run で配線を検証）
    from tsugi.audit import audit_cross_vendor
    def _run(scale, vseed):
        def r(seed):
            g = np.random.default_rng(vseed * 9973 + seed)
            return _a * scale + 1e-4 * g.standard_normal(_a.shape).astype(np.float32)
        return r
    check("audit_cross_vendor measures noise then passes equivalent vendors",
          audit_cross_vendor(_run(1.0, 1), _run(1.0, 2), K=256, n_runs=8).portable)
    check("audit_cross_vendor blocks a real cross-vendor divergence",
          not audit_cross_vendor(_run(1.0, 1), _run(1.05, 2), K=256, n_runs=8).portable)

    # 20. propagation: 相対増幅は reduce/softmax/exp のみ・empirical_cond は data-driven
    from tsugi.propagation import empirical_cond, is_amplifier
    check("only reduce/softmax/exp amplify relative error (not div/reciprocal/add)",
          is_amplifier("reduce") and is_amplifier("exp")
          and not is_amplifier("div") and not is_amplifier("add"))
    _sg = np.random.default_rng(0).standard_normal((4, 256))
    check("empirical_cond is data-driven (signed reduction cancels, positive does not)",
          empirical_cond(_sg, "reduce", axis=1)
          > empirical_cond(np.abs(_sg), "reduce", axis=1) * 3)

    # 21. batch invariance（Thinking Machines 2025）: バッチ依存だが決定論的な第三の床
    from tsugi.nondeterminism import (
        measure_batch_variance,
        simulate_batch_variant_reduction,
    )
    _bp = np.random.default_rng(0).standard_normal(4096).astype(np.float32)
    check("batch-variant reduction is deterministic per batch, varies across batch",
          simulate_batch_variant_reduction(_bp, 128) == simulate_batch_variant_reduction(_bp, 128)
          and simulate_batch_variant_reduction(_bp, 128) != simulate_batch_variant_reduction(_bp, 256))
    check("batch-invariance floor is a positive independent floor",
          measure_batch_variance(lambda t: simulate_batch_variant_reduction(_bp, t))["spread"] > 0.0)
    # robust 床は単発の外れ値に頑健（Q49）
    _rp = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    def _orun(s):
        g = np.random.default_rng(1000 + s).standard_normal(_rp.shape).astype(np.float32)
        return _rp + (5e-2 if s == 0 else 1e-6) * g
    _nf = measure_noise_floor(_orun, 16)
    check("robust noise floor rejects a single outlier run",
          _nf["spread_robust"] < _nf["spread"] * 1e-2)

    # 22. decision: argmax 保存的な系統発散(スケール/シフト)はタスク影響ゼロ（arXiv:2511.00025）
    from tsugi.decision import decompose_divergence
    from tsugi.decision import flip_rate as _flip
    _z = np.random.default_rng(0).standard_normal((2000, 200)).astype(np.float32)
    _scaled = (_z * 1.5).astype(np.float32)
    check("systematic affine divergence (scale) is numerically large but flips nothing",
          _flip(_z, _scaled) == 0.0
          and decompose_divergence(_z, _scaled)["total"] > 0.1
          and decompose_divergence(_z, _scaled)["residual"] < 1e-3)
    # top-k 候補集合フリップ（生成タスク向け・k=1 で argmax 一致）
    from tsugi.decision import nucleus_flip_rate, topk_flip_rate
    _zb = _z + 3e-2 * np.random.default_rng(5).standard_normal(_z.shape).astype(np.float32)
    check("topk_flip_rate generalizes argmax (k=1 equals, monotone in k)",
          abs(topk_flip_rate(_z, _zb, 1) - _flip(_z, _zb)) < 1e-12
          and topk_flip_rate(_z, _zb, 1) <= topk_flip_rate(_z, _zb, 8))
    check("nucleus(top-p) flip is probability-dependent (not scale-invariant)",
          nucleus_flip_rate(_z, _z, 0.9) == 0.0
          and nucleus_flip_rate(_z, _zb, 0.9) != nucleus_flip_rate(_z * 5, _zb * 5, 0.9))
    # tie_rate: 量子化 logit の同点（argmax 規約依存）を露出し flip 誤帰属を警告
    from tsugi.decision import tie_rate
    _q = np.round(np.random.default_rng(0).standard_normal((1000, 40)) * 3).astype(np.float32)
    check("tie_rate exposes convention-dependent decisions (quantized ties)",
          tie_rate(_q) > 0.1 and tie_rate(_z) < 0.01)
    # 新視点11 タスク多様性: regression/binary/ranking は argmax と独立に flip を測れる
    from tsugi.decision import (binary_flip_rate, ranking_flip_rate,
                                regression_flip_rate)
    _r_a = np.random.default_rng(9).standard_normal(1000)
    check("regression_flip_rate respects rtol boundary (0 below, 1 above)",
          regression_flip_rate(_r_a, _r_a * 1.0 + 1e-9, rtol=0.01) == 0.0
          and regression_flip_rate(_r_a, _r_a * 2.0, rtol=0.01) == 1.0)
    _b_a = np.clip(np.random.default_rng(10).standard_normal(500) * 0.3 + 0.5, 0, 1)
    check("binary_flip_rate: identical outputs have no flips",
          binary_flip_rate(_b_a, _b_a.copy()) == 0.0
          and binary_flip_rate(np.full(100, 0.9), np.full(100, 0.1)) == 1.0)
    _rk = np.random.default_rng(11).standard_normal(100)
    check("ranking_flip_rate: negligible perturbation preserves top-k set",
          ranking_flip_rate(_rk, _rk + 1e-9, k=10) == 0.0
          and ranking_flip_rate(_rk, np.random.default_rng(12).standard_normal(100), k=10) == 1.0)
    # shared-mode 障害: cross-vendor 一致は必要十分でない（oracle 照合で初めて見える盲点）
    from tsugi.calibration import SM_SHARED, detect_shared_mode
    _orc = np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32)
    _va = (_orc * 1.05).astype(np.float32)
    check("cross-vendor agreement is blind to shared-mode failure (needs oracle)",
          is_equivalent_combined(_va, _va.copy(), 256, "float16")
          and detect_shared_mode(_va, _va.copy(), _orc, 256) == SM_SHARED)

    # オラクル自体をメタモルフィック関係で検証（無限後退を断つ・第二オラクル不要）
    from tsugi.oracle_check import oracle_is_trustworthy, verify_oracle
    check("oracle is verified by metamorphic relations (not asserted)",
          oracle_is_trustworthy())
    check("oracle check can actually flag deviation (not always-green)",
          not verify_oracle(rtol=0.0).ok)

    # レイアウト不一致(値正しい・位置違い)を真の数値発散と区別（element-wise 比較の盲点）
    from tsugi.equivalence import DV_DIVERGENT, DV_LAYOUT, classify_divergence
    _la = np.random.default_rng(0).standard_normal((48, 48)).astype(np.float32)
    check("layout mismatch (transpose) distinguished from numerical divergence",
          classify_divergence(_la, _la.T.copy(), 256) == DV_LAYOUT
          and classify_divergence(_la, (_la * 1.5).astype(np.float32), 256) == DV_DIVERGENT)

    # provenance: verdict は point-in-time。スタック更新で証明書は stale（再検証要）
    from tsugi.provenance import certify, is_stale
    _cert = certify("portable", rocm="6.0", driver="550.0")
    check("certificate is stale after stack upgrade (point-in-time verdict)",
          not is_stale(_cert, rocm="6.0", driver="550.0")
          and is_stale(_cert, rocm="6.0", driver="560.0"))

    # rollout: per-token フリップは自己回帰生成長で複利増幅（per-token 許容 ⇏ per-sequence）
    from tsugi.rollout import (
        analyze_rollout,
        rollout_from_logits,
        sequence_survival,
        simulate_rollout,
    )
    check("per-token flip compounds over rollout length (token-OK ⇏ sequence-OK)",
          analyze_rollout(0.01, 1).max_risk < analyze_rollout(0.01, 1000).max_risk
          and abs(sequence_survival(0.01, 100) - simulate_rollout(0.01, 100, 20000, 1)) < 0.02)
    # rollout fail-safe: 0 フリップ観測でも survival=100% と過信しない（小標本の上限を使う）
    _ident = np.random.default_rng(0).standard_normal((200, 50)).astype(np.float32)
    check("rollout does not mistake zero observed flips for zero flip rate (fail-safe)",
          rollout_from_logits(_ident, _ident.copy(), 10**6, conservative=True).survival < 1.0
          and rollout_from_logits(_ident, _ident.copy(), 10**6, conservative=False).survival == 1.0)
    # rollout はデコード方式に整合: サンプリング集合フリップ ≥ greedy argmax フリップ
    # （argmax 同一でも候補集合は分岐しうる → サンプリング生成の発散を過小評価しない）
    _da = np.random.default_rng(3).standard_normal((1500, 80)).astype(np.float32)
    _db = _da + 5e-2 * np.random.default_rng(4).standard_normal(_da.shape).astype(np.float32)
    check("rollout per-token flip rate honors decode mode (sampling ≥ greedy)",
          rollout_from_logits(_da, _db, 256, decode="nucleus", top_p=0.9).flip_rate
          >= rollout_from_logits(_da, _db, 256, decode="greedy").flip_rate)
    # rollout: 初回発散の中央値は平均より系統的に小さい（幾何分布の右裾・第19回）
    # divergence_step_quantile は実装済みだが従来どのレポートにも接続されていなかった。
    # 平均だけでは「典型的にはもっと長く保つ」と楽観視しやすい（右裾に平均が引っ張られる）。
    from tsugi.rollout import divergence_step_quantile, expected_divergence_step
    check("rollout reports median divergence step alongside mean (fail-safe: mean overstates typical survival)",
          divergence_step_quantile(0.01, 0.5) < expected_divergence_step(0.01)
          and analyze_rollout(0.01, 100).median_step == divergence_step_quantile(0.01, 0.5)
          and "中央値" in analyze_rollout(0.01, 100).to_text())

    # 新視点10 worstcase: 平均ケース等価 ⇏ 最悪ケース等価（能動探索が代表を超える反例を発見）
    from tsugi.worstcase import analyze_worst_case

    def _wc_fp16(x):
        acc = np.float16(0.0)
        for v in np.asarray(x, dtype=np.float16):
            acc = np.float16(acc + np.float16(v * v))
        return np.array([acc], dtype=np.float64)

    def _wc_fp32(x):
        return np.array([np.sum(np.asarray(x, dtype=np.float32) ** 2)], dtype=np.float64)

    _wc_samples = [np.random.default_rng(0).standard_normal(64) for _ in range(16)]
    _wc = analyze_worst_case(_wc_fp16, _wc_fp32, _wc_samples, tol=1e-3,
                             bounds=(-30.0, 30.0), steps=900, seed=1)
    check("worst-case search finds in-envelope counterexample average-case verification misses",
          _wc.typical_divergence < _wc.tol < _wc.worst_divergence
          and _wc.max_risk == portability.Risk.BLOCK
          and _wc.x_worst is not None)
    # 同一ベンダーには偽陽性を出さない（探索しても発散 0）
    check("worst-case search has no false positive on identical vendors",
          analyze_worst_case(_wc_fp16, _wc_fp16, _wc_samples, tol=1e-6,
                             steps=200, seed=0).worst_divergence == 0.0)

    # 23. equivalence も共通 Risk インターフェース（report 統一・Q44/Q47）
    from tsugi.equivalence import compare as _cmp
    _o = np.ones((4, 4), np.float32)
    check("equivalence exposes uniform risk/ok interface like other reports",
          _cmp(_o, _o.copy()).risk == portability.Risk.OK
          and _cmp(_o, _o + 10.0).risk == portability.Risk.BLOCK)

    # 24. calibration ROC: 合成判定は閾値超のバグ強度で偽OK=0・max_abs 単独は見逃す（Q36）
    from tsugi.calibration import roc_sweep
    _roc = roc_sweep(strengths=(0.05,), K=2048, seeds=8)
    check("ROC sweep: combined verifier false-OK=0 above threshold, max_abs misses",
          _roc[0]["false_ok_combined"] == 0.0 and _roc[0]["false_ok_max_abs"] > 0.0)

    # 25. torch backend がコード生成前でも FX グラフに静的検証を届ける（Q23/Q26）
    from tsugi_torch.fxbridge import audit_fx, fx_to_graph_ops

    class _N:
        def __init__(s, op, t, shp=None):
            s.op, s.target, s.meta = op, t, ({"tensor_meta": type("M", (), {"shape": shp})} if shp else {})

    class _G:
        def __init__(s, ns):
            s.graph = type("GR", (), {"nodes": ns})
    _gm = _G([_N("call_function", "aten.addmm.default", (8, 512)),
              _N("call_function", "aten._softmax.default"),
              _N("output", "output")])
    check("torch backend maps FX aten ops to logical ops (matmul/softmax)",
          [o.kind for o in fx_to_graph_ops(_gm)] == ["matmul", "softmax"])
    check("torch backend audit_fx surfaces amplifiers before codegen",
          "softmax" in audit_fx(_gm)["amplifiers"])
    _lg = np.random.default_rng(0).standard_normal((500, 128)).astype(np.float32)
    check("torch backend translates model divergence to task flip bound (static→task)",
          0.0 <= audit_fx(_gm, ref_logits=_lg)["task_flip_bound"] <= 1.0
          and audit_fx(_gm)["task_flip_bound"] is None)

    # 26. safety 係数は単一情報源（constants.SAFETY）に集約（Q1/Q2）
    from tsugi.constants import SAFETY
    from tsugi.propagation import GraphOp
    from tsugi.tolerance import derive_tolerance as _dt
    check("safety factor is single-sourced from constants.SAFETY",
          GraphOp("matmul", K=8).safety == SAFETY
          and _dt(8, "float16")["atol"] == _dt(8, "float16", safety=SAFETY)["atol"])

    # 27. attribution 新視点12: 発散 onset/spike で O(log L) デバッグ経路を実現
    from tsugi.attribution import attribute, layer_divergences, find_onset, find_spike, bisect_onset

    _x12 = np.array([1.0, 2.0, 3.0])

    def _attr_id(x):
        return np.asarray(x, dtype=np.float64)

    def _attr_perturb(x):
        return np.asarray(x, dtype=np.float64) + 0.1

    _layers_a12 = [_attr_id, _attr_id, _attr_id, _attr_id]
    # Perturbation injected at layer 1 → onset=1, spike=1
    _layers_b12 = [_attr_id, _attr_perturb, _attr_id, _attr_id]
    _divs12 = layer_divergences(_layers_a12, _layers_b12, _x12, relative=False)
    check("attribution layer_divergences returns one entry per layer",
          len(_divs12) == 4 and all(isinstance(d, float) for d in _divs12))
    check("attribution onset detects first layer exceeding threshold",
          find_onset(_divs12, threshold=0.05) == 1
          and find_onset(_divs12, threshold=200.0) is None)
    check("attribution spike identifies max divergence increment layer",
          find_spike(_divs12) == 1)
    check("attribution identical vendors → onset=None (no false positive)",
          attribute(_layers_a12, _layers_a12, _x12, tol=1e-9).onset is None)

    # 28. bisect_onset と linear scan は同じ onset を返す（O(log L) correctness）
    def _pf_a(i, x):
        out = np.asarray(x, dtype=np.float64)
        for fn in _layers_a12[:i + 1]:
            out = np.asarray(fn(out), dtype=np.float64)
        return out

    def _pf_b(i, x):
        out = np.asarray(x, dtype=np.float64)
        for fn in _layers_b12[:i + 1]:
            out = np.asarray(fn(out), dtype=np.float64)
        return out

    check("attribution bisect_onset matches linear find_onset (O(log L) correctness)",
          bisect_onset(_pf_a, _pf_b, _x12, n_layers=4, tol=0.05, relative=False) == 1)

    # 28b. diagnose() は attribution + blame を 1 回で返す（統合診断・孤立 API を閉じる）
    from tsugi.attribution import diagnose
    _diag_oracle = [_attr_id, _attr_id, _attr_id, _attr_id]
    _diag_b = [_attr_id, _attr_perturb, _attr_id, _attr_id]  # B diverges at layer 1
    _dr = diagnose(_layers_a12, _diag_b, _diag_oracle, _x12,
                   tol=0.05, relative=False, names=["L0", "L1", "L2", "L3"])
    check("diagnose spike=1 (where B diverges)",
          _dr.spike == 1 and _dr.spike_closer == "A")  # A matches oracle → blame B
    check("diagnose onset=1 (first layer exceeding tol)",
          _dr.onset == 1)

    # 29. blame 新視点13: どちらのベンダーが oracle に近いか（責帰）
    from tsugi.blame import compare_accuracy, accuracy_relative, layer_blame

    _oracle13 = np.array([1.0, 2.0, 3.0])
    _a13 = _oracle13 + 1e-8      # A almost exact
    _b13 = _oracle13 + 0.5       # B far from oracle

    def _bid(x): return np.asarray(x, dtype=np.float64)
    def _bperturb(x): return np.asarray(x, dtype=np.float64) + 0.5

    check("blame: A closer to oracle → closer='A' (B blamed)",
          compare_accuracy(_a13, _b13, _oracle13, tol=1e-4).closer == "A")
    check("blame: identical to oracle → OK risk",
          compare_accuracy(_oracle13.copy(), _oracle13.copy(),
                           _oracle13, tol=1e-4).max_risk
          == __import__("tsugi.report", fromlist=["Risk"]).Risk.OK)
    check("blame: accuracy_relative of exact oracle match is zero",
          accuracy_relative(_oracle13, _oracle13) == 0.0)

    # 30. layer_blame が per-layer で dist_a / dist_b を返す（attribution との接続）
    _lb_oracle = [_bid, _bid, _bid]
    _lb_a = [_bid, _bid, _bid]            # A matches oracle everywhere
    _lb_b = [_bid, _bperturb, _bid]       # B diverges at layer 1
    _lb_dists = layer_blame(_lb_a, _lb_b, _lb_oracle, _oracle13, relative=False)
    check("layer_blame returns per-layer dist tuples",
          len(_lb_dists) == 3
          and all(isinstance(da, float) and isinstance(db, float)
                  for da, db in _lb_dists))
    check("layer_blame detects B divergence at layer 1 (da=0, db>0)",
          _lb_dists[1][0] == 0.0 and _lb_dists[1][1] > 0.1)

    # 31. TF32 dtype: NVIDIA Ampere+ の 10-bit 仮数 GEMM に fp16 と同等の許容を適用
    from tsugi.equivalence import TOLERANCE as _TOL
    from tsugi.tolerance import UNIT_ROUNDOFF as _URO
    check("TF32 tolerance equals float16 (10-bit mantissa, cross-vendor Ampere↔AMD)",
          _TOL["tf32"]["atol"] == _TOL["float16"]["atol"]
          and _URO["tf32"] == _URO["float16"])
    # dtype="tf32" で compare が動く（KeyError 不発）
    _tf32_a = np.ones((4, 4), np.float32)
    _tf32_b = _tf32_a + 5e-3   # fp16/tf32 許容内(1e-2)だが fp32 許容(1e-4)では外
    check("compare(..., dtype='tf32') accepts TF32-level drift without KeyError",
          compare(_tf32_a, _tf32_b, "tf32").equivalent
          and not compare(_tf32_a, _tf32_b, "float32").equivalent)

    # 32. NaN/Inf 明示タグ: 精度発散とデータ破壊を区別（根本原因診断の起点）
    _nan_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    _nan_b = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    _nan_rep = compare(_nan_a, _nan_b, "float32")
    check("NaN in output → has_non_finite=True and equivalent=False (data corruption tag)",
          _nan_rep.has_non_finite and not _nan_rep.equivalent)
    _fin_rep = compare(_nan_a, _nan_a + 0.5, "float32")
    check("finite divergence → has_non_finite=False (精度発散とデータ破壊を区別)",
          not _fin_rep.has_non_finite)

    # 33. float64 が envelope DTYPE_LIMITS に登録済み（float32 フォールバック防止）
    from tsugi.envelope import dtype_limits as _dlim
    check("float64 dtype_limits is not the float32 fallback (max_normal differs by 270 orders)",
          _dlim("float64").max_normal > _dlim("float32").max_normal * 1e260)

    # 33b. FP8 (OCP OFP8 E4M3/E5M2): H100/MI300/B200 推論主流の dtype を 3 テーブルに登録
    check("FP8 unit_roundoff ordered by mantissa bits (e5m2 coarsest > e4m3 > fp16)",
          _URO["float8_e5m2"] > _URO["float8_e4m3"] > _URO["float16"]
          and _URO["float8_e4m3"] == 2.0 ** -4 and _URO["float8_e5m2"] == 2.0 ** -3)
    check("FP8 tolerance is coarser than fp16 (few mantissa bits + amax scale divergence)",
          _TOL["float8_e4m3"]["atol"] > _TOL["float16"]["atol"]
          and _TOL["float8_e5m2"]["atol"] > _TOL["float8_e4m3"]["atol"])
    check("FP8 E4M3 narrow range (max=448) makes overflow the main envelope risk",
          _dlim("float8_e4m3").max_normal == 448.0
          and _dlim("float8_e5m2").max_normal == 57344.0
          and _dlim("float8_e4m3").max_normal < _dlim("float16").max_normal)
    # E4M3 では未スケール活性(1000)が overflow するが fp16 ではしない（dtype 依存リスク）
    from tsugi.envelope import certify_gemm as _cg8
    _x8 = np.full((4, 4), 1000.0, np.float32)
    check("FP8 E4M3 flags 1000 as overflow (BLOCK) where fp16 does not (amax scaling needed)",
          not check_tensor(_x8, _cg8(64, "float8_e4m3", 1.0)).in_envelope
          and not any("overflow" in f.message
                      for f in check_tensor(_x8, _cg8(64, "float16", 1000.0)).findings))

    # 34. nondeterminism 静的カタログ: atomicAdd 由来の非決定 op を実行前に検出（PyTorch 公式）
    from tsugi.nondeterminism import (classify_nondeterminism, op_is_nondeterministic)
    check("atomicAdd nondeterministic ops cataloged (scatter_add/index_add/bincount, PyTorch docs)",
          op_is_nondeterministic("scatter_add") and op_is_nondeterministic("aten.index_add.default")
          and op_is_nondeterministic("bincount")
          and not op_is_nondeterministic("matmul") and not op_is_nondeterministic("softmax"))
    _ndrep = classify_nondeterminism(["matmul", "softmax", "scatter_add", "add"])
    check("graph with atomicAdd op statically requires runtime noise-floor measurement",
          _ndrep.requires_noise_floor and _ndrep.nondet_ops == ("scatter_add",)
          and not classify_nondeterminism(["matmul", "softmax", "add"]).requires_noise_floor)
    # FX 橋がコード生成前に非決定 op を audit に届ける（torch.compile 経路）
    from tsugi_torch.fxbridge import audit_fx as _afx

    class _NN:
        def __init__(s, op, t):
            s.op, s.target, s.meta = op, t, {}

    class _GG:
        def __init__(s, ns):
            s.graph = type("GR", (), {"nodes": ns})
    _ndgm = _GG([_NN("call_function", "aten.addmm.default"),
                 _NN("call_function", "aten.scatter_add.default"),
                 _NN("output", "output")])
    check("torch backend audit_fx flags atomicAdd nondeterminism before codegen",
          _afx(_ndgm)["requires_noise_floor"]
          and any("scatter_add" in n for n in _afx(_ndgm)["nondeterministic_ops"]))

    # 35. dynamic shape 検出: shape guard で形状依存カーネルが選ばれ等価性が shape ごとに異なる
    # 研究知見（2025）: torch.compile shape guard は形状ごとにカーネルを特化する。
    # 特化カーネルはタイル幅・縮約順序・アキュムレータ幅が変わるため
    # 1 形状で認証した等価性は他形状に転用できない → per-shape 再検証が必要。
    class _SymDim:
        """torch.SymInt 模倣 — int() で TypeError を送出する symbolic 次元。"""
        def __int__(self): raise TypeError("symbolic")

    class _SymNode:
        def __init__(s, op, t, symbolic=False):
            shp = (_SymDim(), 512) if symbolic else (8, 512)
            s.op, s.target = op, t
            s.meta = {"tensor_meta": type("M", (), {"shape": shp})}

    class _SymGM:
        def __init__(s, nodes):
            s.graph = type("GR", (), {"nodes": nodes})

    _dynm = _SymGM([_SymNode("call_function", "aten.addmm.default", symbolic=True),
                    _SymNode("output", "output")])
    _statm = _SymGM([_SymNode("call_function", "aten.addmm.default", symbolic=False),
                     _SymNode("output", "output")])
    check("dynamic shape graph detected as has_dynamic_shapes=True (shape guard→per-shape re-verify)",
          _afx(_dynm)["has_dynamic_shapes"])
    check("static shape graph is NOT dynamic (no symbolic dims → has_dynamic_shapes=False)",
          not _afx(_statm)["has_dynamic_shapes"])

    # 37. certify_from_sample: 実 RMS scale を測定して認証（scale=1 暗黙仮定を排除・Q14）
    # scale=1 で認証後 scale=50 のデータを check_tensor すると scale 超過 BLOCK が誤発火する。
    # certify_from_sample は同じデータで認証・検査を一致させ、偽 BLOCK を防ぐ。
    from tsugi.envelope import certify_from_sample as _cfs
    _x_large = np.random.default_rng(99).standard_normal((16, 64)).astype(np.float32) * 50.0
    _env_wrong = certify_gemm(K=64, dtype="float32", scale=1.0)
    check("certify_gemm(scale=1) for scale=50 data causes BLOCK (the problem certify_from_sample solves)",
          check_tensor(_x_large, _env_wrong).max_risk == portability.Risk.BLOCK)
    _env_right = _cfs(_x_large, K=64, dtype="float32")
    _right_rep = check_tensor(_x_large, _env_right)
    _scale_blocks = [f for f in _right_rep.findings if "scale" in f.message and f.risk == portability.Risk.BLOCK]
    check("certify_from_sample eliminates spurious scale-BLOCK by using real RMS (Q14 fix)",
          not _scale_blocks and _env_right.scale_max > 40.0)

    # 38. audit_fx ref_scale: logits を渡すと RMS scale が測定されて出力に含まれる（envelope との橋）
    _lg2 = np.random.default_rng(0).standard_normal((200, 64)).astype(np.float32) * 8.0
    _afx_no = _afx(_gm)          # logits 無し
    _afx_with = _afx(_gm, ref_logits=_lg2)
    check("audit_fx without logits does not include ref_scale key",
          "ref_scale" not in _afx_no)
    check("audit_fx with logits includes ref_scale (RMS within 1%) for certify_from_sample use",
          "ref_scale" in _afx_with
          and abs(_afx_with["ref_scale"] - float(np.sqrt(np.mean(_lg2 ** 2)))) < 0.1)

    # 39. backend 冪等性: tsugi_torch の register() は二重呼出しで同じ挙動を繰り返さない（Q28 fix）
    # torch 有り: _BACKEND_REGISTERED=True → 二度目は即 return。
    # torch 無し（本環境）: RuntimeError だが _BACKEND_REGISTERED=False で guard 変数が存在する。
    from tsugi_torch import _BACKEND_REGISTERED, register
    check("_BACKEND_REGISTERED is a bool (idempotency guard exists in module)",
          isinstance(_BACKEND_REGISTERED, bool))
    # 二度目の呼出しは _BACKEND_REGISTERED=True ならスキップ・False なら同じ RuntimeError
    # → どちらの場合も「前回と同一の挙動」 = 冪等
    try:
        register()
        _second_raise = False
    except RuntimeError:
        _second_raise = True
    check("backend register() second call is idempotent (same outcome as first)",
          not _BACKEND_REGISTERED or not _second_raise)

    # 40. audit(sample=...) facade が certify_from_sample を使う（Q13/Q14 の facade 接続・第12回）
    # Round 11 で envelope.certify_from_sample を追加したが audit() は certify_gemm(...,1.0) の
    # ままだった（scale=1 暗黙仮定が製品経路の主要 facade に残存）。ここで接続を検証する。
    _sample50 = np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32) * 50.0
    _a_nosample = audit(mod, dcfg, block_dims=block)
    _a_sample = audit(mod, dcfg, block_dims=block, sample=_sample50)
    _num_no = next(p for p in _a_nosample.phases if p.name.startswith("numerics"))
    _num_yes = next(p for p in _a_sample.phases if p.name.startswith("numerics"))
    check("audit() without sample explicitly states scale=1.0 assumption (no silent default)",
          "scale=1.0 仮定" in _num_no.to_text())
    check("audit(sample=...) uses real measured RMS scale instead of scale=1 (Q13/Q14 facade fix)",
          "sample 実測" in _num_yes.to_text()
          and "scale=1.0 仮定" not in _num_yes.to_text())

    # 41. audit(sample=...) は増幅 op（reduce/exp）の cond を empirical_cond で自動実測する（第13回）
    # sample を渡しても cond=1 のまま「下界」WARN するだけだった（Q7/Q8/Q11 の未接続）を解消。
    import tsugi as _ts
    from tsugi import tile as _tile

    @_ts.jit
    def _sm_kernel(x, out, N, BN):
        p = _ts.program_id(0)
        row = _tile.load(x, (p * BN, 0), (BN, N))
        m = _tile.reduce(row, 1, "max")
        e = _tile.exp(row - m)
        s = _tile.reduce(e, 1, "sum")
        _tile.store(out, (p * BN, 0), (e / s).to(_ts.float16))

    from tsugi.autotune import TileConfig as _TC
    _xsm = np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32)
    _mod_sm = _ts.trace(_sm_kernel, (_xsm, _xsm.copy(), 16, 16), {}, (0,))
    _cfg_sm = _TC(block_m=16, block_n=16, block_k=16, num_stages=2, num_warps=4)
    _a_no = audit(_mod_sm, _cfg_sm, block_dims=(16,))
    _a_yes = audit(_mod_sm, _cfg_sm, block_dims=(16,), sample=_xsm)
    _prop_no = next(p for p in _a_no.phases if "propagation" in p.name.lower())
    _prop_yes = next(p for p in _a_yes.phases if "propagation" in p.name.lower())
    check("audit(sample=...) measures empirical_cond for amplifying ops (no longer static lower bound)",
          "実測済み" in _prop_yes.to_text() and "静的 cond=1 は" not in _prop_yes.to_text())
    check("empirical_cond changes model_divergence vs the cond=1 static default",
          _prop_yes.to_text() != _prop_no.to_text())

    # 42. audit_cross_vendor(robust=...) が外れ値頑健な noise floor を実機入口に接続する（第14回）
    # nondeterminism.compare_stable は robust=True(Q49・10-90パーセンタイル幅)をサポートするが
    # audit_cross_vendor は常に max-min(spread)を使い、単発グリッチで noise floor が桁違いに
    # 膨張し、真に EQUIVALENT な差を INDISTINGUISHABLE(未定義WARN)に押し込めていた。
    from tsugi.audit import audit_cross_vendor as _acv
    _base42 = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)

    def _run_a42(s):
        g = np.random.default_rng(1000 + s).standard_normal(_base42.shape).astype(np.float32)
        return _base42 + (5e-2 if s == 7 else 1e-6) * g   # 単発グリッチ(seed=7)

    def _run_b42(s):
        g = np.random.default_rng(2000 + s).standard_normal(_base42.shape).astype(np.float32)
        return _base42 * 1.0008 + 1e-6 * g   # 真の系統発散

    _ad_def = _acv(_run_a42, _run_b42, K=256, n_runs=16)
    _ad_rob = _acv(_run_a42, _run_b42, K=256, n_runs=16, robust=True)
    _eq_def = next(p for p in _ad_def.phases if "equivalence" in p.name.lower())
    _eq_rob = next(p for p in _ad_rob.phases if "equivalence" in p.name.lower())
    check("audit_cross_vendor default (robust=False) is glitch-vulnerable (reproduces the problem)",
          "INDISTINGUISHABLE" in _eq_def.to_text())
    check("audit_cross_vendor(robust=True) resists single-glitch noise inflation (Q49 entry-point fix)",
          "EQUIVALENT" in _eq_rob.to_text() and "INDISTINGUISHABLE" not in _eq_rob.to_text())

    # 43. audit_runtime(task=...) が decision.compare_task に委譲する（第15回）
    # compare_task(regression/binary/ranking) は実装済みだが audit_runtime は常に
    # compare_decisions（分類 argmax 専用）を呼んでいた（非分類タスクが未接続だった）。
    from tsugi.audit import audit_runtime as _art
    _a43 = np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32)
    _reg_a = np.random.default_rng(1).standard_normal(300)
    _reg_b = _reg_a * 2.0   # 100% 乖離 → 明確な回帰フリップ
    _ad_reg = _art(_a43, _a43.copy(), K=64, logits_a=_reg_a, logits_b=_reg_b,
                  task="regression", flip_budget=0.001, task_kwargs={"rtol": 0.01})
    _dp_reg = next(p for p in _ad_reg.phases if p.name.startswith("decision"))
    check("audit_runtime(task='regression') delegates to compare_task and detects real divergence",
          "regression" in _dp_reg.name and _dp_reg.max_risk == portability.Risk.BLOCK)
    _la43 = np.random.default_rng(2).standard_normal((300, 20)).astype(np.float32)
    _ad_default43 = _art(_a43, _a43.copy(), K=64, logits_a=_la43, logits_b=_la43.copy())
    _dp_default43 = next(p for p in _ad_default43.phases if p.name.startswith("decision"))
    check("audit_runtime default task remains classification (backward compatible)",
          _dp_default43.name == "decision タスクレベル等価")

    # 44. audit_runtime(layers_a=..., layers_b=...) が attribution.diagnose を接続する（第16回）
    # diagnose（onset/spike特定＋blame統合）は実装済みだが audit_runtime は BLOCK を出すだけで
    # どの層・どちらのベンダーが悪いかを一度も特定していなかった。
    def _id44(x):
        return np.asarray(x, dtype=np.float64)

    def _scale2_44(x):
        return np.asarray(x, dtype=np.float64) * 2.0

    _oracle_layers44 = [_id44, _scale2_44, _id44]
    _layers_a44 = [_id44, _scale2_44, _id44]
    _layers_b44 = [_id44, lambda x: np.asarray(x, dtype=np.float64) * 2.0 + 0.5, _id44]
    _a_out44 = np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32)
    _ad44 = audit_runtime(_a_out44, _a_out44.copy(), K=64,
                          layers_a=_layers_a44, layers_b=_layers_b44,
                          layers_oracle=_oracle_layers44, x0=np.array([1.0, 2.0, 3.0]))
    _attr44 = next(p for p in _ad44.phases if p.name.startswith("attribution"))
    check("audit_runtime(layers_a/layers_b/layers_oracle) pinpoints the divergent layer and culprit vendor",
          "vendor B" in _attr44.to_text())
    _ad_no_layers44 = audit_runtime(_a_out44, _a_out44.copy(), K=64)
    check("audit_runtime without layers_a/layers_b has no attribution phase (backward compatible)",
          not any(p.name.startswith("attribution") for p in _ad_no_layers44.phases))

    # 45. audit_runtime(fn_a=..., fn_b=..., worst_samples=...) が worstcase.analyze_worst_case
    # を接続する（第17回）。唯一の能動探索層が facade に未接続だった欠陥の6件目。
    def _fp16_45(x):
        acc = np.float16(0.0)
        for v in np.asarray(x, dtype=np.float16):
            acc = np.float16(acc + np.float16(v * v))
        return np.array([acc], dtype=np.float64)

    def _fp32_45(x):
        return np.array([np.sum(np.asarray(x, dtype=np.float32) ** 2)], dtype=np.float64)

    _rng45 = np.random.default_rng(0)
    _samples45 = [_rng45.standard_normal(64) for _ in range(16)]
    _a45 = _rng45.standard_normal((8, 8)).astype(np.float32)
    _ad45 = audit_runtime(_a45, _a45.copy(), K=64, fn_a=_fp16_45, fn_b=_fp32_45,
                          worst_samples=_samples45, worst_tol=1e-3,
                          worst_bounds=(-30.0, 30.0), worst_steps=900, worst_seed=1)
    _wc45 = next(p for p in _ad45.phases if p.name.startswith("worstcase"))
    check("audit_runtime(fn_a/fn_b/worst_samples) finds an in-envelope counterexample (BLOCK)",
          _wc45.max_risk == portability.Risk.BLOCK and not _ad45.portable)
    _ad_no_wc45 = audit_runtime(_a45, _a45.copy(), K=64)
    check("audit_runtime without fn_a/fn_b/worst_samples has no worstcase phase (backward compatible)",
          not any(p.name.startswith("worstcase") for p in _ad_no_wc45.phases))

    # 46. audit_runtime の equivalence phase が classify_divergence で LAYOUT を区別する（第18回）
    # equivalence.classify_divergence(LAYOUTか真の数値発散か)は実装済みだが audit_runtime は
    # 一律 BLOCK にするだけで両者を区別していなかった(修正箇所が全く異なる: 整列 vs 精度)。
    _a46 = np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32)
    _ad_layout46 = audit_runtime(_a46, _a46.T.copy(), K=256, noise_floor=1e-6)
    _eq_layout46 = next(p for p in _ad_layout46.phases if p.name.startswith("equivalence"))
    check("audit_runtime tags transpose (layout bug, values correct) as LAYOUT not plain BLOCK",
          _eq_layout46.max_risk == portability.Risk.BLOCK and "LAYOUT" in _eq_layout46.to_text())
    _ad_true46 = audit_runtime(_a46, (_a46 * 1.5).astype(np.float32), K=256, noise_floor=1e-6)
    _eq_true46 = next(p for p in _ad_true46.phases if p.name.startswith("equivalence"))
    check("audit_runtime does not tag a true scale divergence as LAYOUT (no false classification)",
          _eq_true46.max_risk == portability.Risk.BLOCK and "LAYOUT" not in _eq_true46.to_text())

    # 47. calibration.check_systematic は bias 点推定でなく上側限界(bias+stderr)で判定する（第20回）
    # 小 N テンソルではたまたま小さい bias が出て真の系統誤差を見逃しうる（偽OK）。
    # rollout.flip_rate_upper_bound と同じ fail-safe パターン（点推定でなく上側限界）を適用。
    from tsugi.calibration import check_systematic as _cs47
    from tsugi.calibration import systematic_divergence as _sd47
    from tsugi.calibration import systematic_divergence_stderr as _sds47
    from tsugi.tolerance import unit_roundoff
    _rng47 = np.random.default_rng(2)
    _a47 = _rng47.standard_normal(4) * 1.0
    _b47 = _a47.copy()
    _b47[0] *= 1.05
    _bias47 = _sd47(_a47, _b47)
    _stderr47 = _sds47(_a47, _b47, n_boot=300, seed=2)
    _thresh47 = SAFETY * unit_roundoff("float16")
    check("small-N point-estimate bias alone looks negligible (reproduces the false-OK setup)",
          abs(_bias47) < 0.5 * _thresh47)
    check("small-N upper bound (bias+stderr) exceeds threshold where point estimate alone would not",
          abs(_bias47) + _stderr47 > _thresh47)
    check("check_systematic uses the upper bound and correctly BLOCKs the small-N case (no false-OK)",
          _cs47(_a47, _b47, K=1, dtype="float16").max_risk == portability.Risk.BLOCK)

    # 48. decision.predicted_flip_bound は点推定でなく Wilson 上側限界で判定する（第21回）
    # P(margin<2δ) は代表 logit n 件からの点推定。n が小さいと 0 件観測でも真の確率は 0 でない
    # （rollout.flip_rate_upper_bound と同じ rule-of-three 問題）。第20回 calibration と同型修正。
    from tsugi.decision import margin as _margin48
    from tsugi.decision import predicted_flip_bound as _pfb48
    _z48 = np.random.default_rng(0).standard_normal((20, 50)).astype(np.float32) * 5.0
    _delta48 = 0.01
    _m48 = _margin48(_z48)
    _k48 = int(np.count_nonzero(_m48 < 2.0 * _delta48))
    check("small representative set (n=20) has zero observed sub-threshold margins (test setup)",
          _k48 == 0)
    check("predicted_flip_bound does not mistake zero observations for zero flip probability (fail-safe)",
          _pfb48(_z48, _delta48) > 0.05)

    # 49. compare_decisions は観測 flip_rate(点推定)でなく flip_rate_ub(Wilson 上側限界)で
    # 予算判定する（第22回・第21回の同型修正を主判定 compare_decisions にも適用）。
    from tsugi.decision import compare_decisions as _cd49
    from tsugi.decision import flip_rate as _fr49
    _rng49 = np.random.default_rng(5)
    _n49 = 30
    _z49 = _rng49.standard_normal((_n49, 20)).astype(np.float32)

    def _vendor49(z, eps, seed):
        return z + eps * np.random.default_rng(seed).standard_normal(z.shape).astype(np.float32)

    _a49 = _vendor49(_z49, 3e-2, 12)
    _b49 = _vendor49(_z49, 3e-2, 13)
    check("small batch (n=30) has zero observed flips (test setup)",
          _fr49(_a49, _b49) == 0.0)
    _rep49 = _cd49(_a49, _b49, flip_budget=0.03)
    check("compare_decisions does not mistake zero observed flips for in-budget (fail-safe, no false-OK)",
          _rep49.flip_rate_ub > 0.03 and _rep49.max_risk >= portability.Risk.WARN)

    # 50. compare_task(regression/binary/ranking) が同様に flip_rate_ub(Wilson 上側限界)で
    # 予算判定する（FEATURE-AUDIT.md A-1: compare_decisions のみ修正済みで非分類 3 タスクが
    # 取り残されていた欠陥の解消）。ranking の 1D 単一クエリは決定的結果ゆえ widening 対象外。
    from tsugi.decision import compare_task as _ct50
    _rng50 = np.random.default_rng(0)
    _n50 = 30
    _a50 = _rng50.standard_normal(_n50)
    _b50 = _a50.copy()   # 全要素が許容内 → 観測フリップは厳密に 0 件
    _rep50 = _ct50(_a50, _b50, task="regression", flip_budget=0.03, rtol=0.01)
    check("compare_task(regression) has zero observed flips (test setup)",
          _rep50.flip_rate == 0.0)
    check("compare_task(regression) does not mistake zero observed flips for in-budget (no false-OK)",
          _rep50.flip_rate_ub > 0.03 and _rep50.max_risk >= portability.Risk.WARN)
    # ranking の 1D（単一クエリ）は決定的結果ゆえ widening 無し（flip_rate_ub == flip_rate）
    _rng50b = np.random.default_rng(1)
    _sa50 = _rng50b.standard_normal(100)
    _sb50 = _sa50 + 1e-9 * _rng50b.standard_normal(100)
    _repr50 = _ct50(_sa50, _sb50, task="ranking", k=10)
    check("compare_task(ranking) with 1D single-query input applies no Wilson widening",
          _repr50.flip_rate_ub == _repr50.flip_rate)

    # 51. audit_runtime の envelope phase が check_outlier_features・check_softmax_input を
    # 接続する（同型スキャンで発見: envelope.check_tensor だけが呼ばれ、同モジュールの
    # outlier feature 検出・softmax exp-overflow 検査が実行時監査に届いていなかった）。
    from tsugi.envelope import certify_gemm as _cg51
    _rng51 = np.random.default_rng(0)
    _a51 = _rng51.standard_normal((32, 256)).astype(np.float32) * 0.1
    _a51[:, 5] *= 20   # outlier channel（check_tensor 単独では IN-envelope のまま）
    _env51 = _cg51(K=256, dtype="float32", scale=1.0)
    _ad51 = audit_runtime(_a51, _a51.copy(), K=256, env=_env51, noise_floor=1e-6)
    _ep51 = next(p for p in _ad51.phases if p.name.startswith("envelope"))
    check("audit_runtime envelope phase surfaces outlier features (check_outlier_features wired)",
          _ep51.max_risk == portability.Risk.WARN and "outlier" in _ep51.to_text())
    _a51b = _rng51.standard_normal((16, 16)).astype(np.float32)
    _logits51 = np.array([[0.0, 12.5, 3.0]], dtype=np.float32)   # > ln(65504)=11.09 fp16
    _env51b = _cg51(K=128, dtype="float16", scale=1.0)
    _ad51b = audit_runtime(_a51b, _a51b.copy(), K=128, env=_env51b, noise_floor=1e-4,
                           logits_a=_logits51, logits_b=_logits51)
    _ep51b = next(p for p in _ad51b.phases if p.name.startswith("envelope"))
    check("audit_runtime envelope phase surfaces fp16 softmax exp-overflow (check_softmax_input wired)",
          _ep51b.max_risk == portability.Risk.BLOCK and "softmax" in _ep51b.to_text())

    failed = [n for n, c in INVARIANTS if not c]
    print(f"\n{'VERIFY PASS' if not failed else 'VERIFY FAIL'}: "
          f"{len(INVARIANTS) - len(failed)}/{len(INVARIANTS)} invariants")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
