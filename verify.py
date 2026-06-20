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

    failed = [n for n, c in INVARIANTS if not c]
    print(f"\n{'VERIFY PASS' if not failed else 'VERIFY FAIL'}: "
          f"{len(INVARIANTS) - len(failed)}/{len(INVARIANTS)} invariants")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
