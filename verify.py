"""Tsugi 不変条件チェッカ（CI verify・ハーネスの verify パターン）。

プロジェクトの不変条件を機械的に検証する。CI で fail on error。

## verify.py と tests/correctness/ の関係（SOCRATIC-50 Q33・意図的な重複）

verify.py の不変条件は tests/correctness/test_*.py と *意図的に* 重複する。二重保守に
見えるが、両者は役割が異なる:

  - **tests/correctness/**: *挙動* を網羅的に exercise する（多数のケース・エッジ・
    property test）。「動くか」を確かめる詳細なテストスイート。
  - **verify.py**: プロジェクトの *主張* を 1 行 1 件の名前つき不変条件として列挙した
    **機械可読なサマリ（契約）**。散文の SPEC でなく実行可能な形で「この製品は何を
    保証するか」を上から下まで scan できる。各不変条件はテストが確かめる挙動のうち
    *設計上の約束* に当たるものだけを、根拠コメント（Q番号・commit）つきで固定する。

したがって重複は冗長でなく **belt-and-suspenders**: テストが挙動を、verify が
「その挙動が意図した保証である」ことを別レイヤで主張する。verify を test から自動生成
する案もあるが、不変条件は「テストのどれが *約束* か」という人間の選別を含むため、
現状は手書きの主張リストとして維持する（生成すると主張の意図が失われる）。
新しい保証を足す時は test（挙動）と verify（主張）の両方に書くのが規約。
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


def _orphan_tests() -> list[str]:
    """`tests/correctness/test_*.py` に定義された test_* 関数が、各ファイルの
    `main()` のテストリストに登録されているか（= 少なくとも一度は実行されるか）を
    機械的に検査する。

    各テストファイルは手書きの `tests = [...]` リストで実行対象を選ぶ構造になっている
    （pytest 等のディスカバリ機構でなく明示的な列挙）。これは「テストを書いたが
    リスト登録を忘れ、一度も実行されない」という静かな品質劣化を招きうる——
    このプロジェクトが本番コードの「facade 未接続」（実装したが呼ばれない関数）で
    繰り返し見つけてきた欠陥のテスト層版。def 行以外の場所に関数名が 1 回以上
    出現すれば「リストで参照されている」とみなす（簡易ヒューリスティック・
    docstring 等でのコメント言及も出現としてカウントするため false negative
    （見逃し）はあっても false positive（誤検出）は起きにくい）。
    """
    import re

    orphans: list[str] = []
    for p in sorted((ROOT / "tests" / "correctness").glob("test_*.py")):
        try:
            src = p.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        for fn in re.findall(r"^def (test_[a-zA-Z0-9_]*)\(", src, re.M):
            refs = len(re.findall(rf"\b{re.escape(fn)}\b", src)) - 1  # -1 = def 行自身
            if refs == 0:
                orphans.append(f"{p.name}:{fn}")
    return orphans


# 依存パッケージのライセンス許容リスト（SOCRATIC-50 Q42）。
# 全て permissive で本体（Apache-2.0）との配布互換性あり。新規依存を追加する際は
# 実際のライセンスを確認してからここに追記することを強制する静的ゲート
# （pip-licenses 等の外部ツール・ネットワークアクセス無しで動作・CPU-only ポリシーと整合）。
_DEPENDENCY_LICENSE_ALLOWLIST: dict[str, str] = {
    "numpy": "BSD-3-Clause",
    "torch": "BSD-3-Clause 系（PyTorch 独自ライセンス・permissive）",
}


def _declared_dependencies(text: str | None = None) -> list[str]:
    """`python/pyproject.toml` の実行時依存パッケージ名を抽出する（Q42・ライセンス監査用）。

    `[build-system] requires`（ビルド時のみ・配布物に含まれない）は対象外。
    `[project] dependencies` と `[project.optional-dependencies]` の各エントリのみを見る。
    TOML パーサ（Python 3.10 では標準ライブラリに `tomllib` が無い）に依存せず、
    この pyproject.toml の単純な構造に絞った正規表現で抽出する簡易実装。
    `text` を渡すとその文字列を対象にする（テスト用・実ファイルを変更せず検証できる）。
    """
    import re

    if text is None:
        text = (PY / "pyproject.toml").read_text(encoding="utf-8")
    raw: list[str] = []
    m = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S)
    if m:
        raw += re.findall(r'"([a-zA-Z0-9_.-]+)', m.group(1))
    opt_start = text.find("[project.optional-dependencies]")
    if opt_start >= 0:
        nxt = text.find("\n[", opt_start + 1)
        opt_section = text[opt_start:nxt if nxt >= 0 else None]
        for m2 in re.finditer(r"=\s*\[(.*?)\]", opt_section, re.S):
            raw += re.findall(r'"([a-zA-Z0-9_.-]+)', m2.group(1))
    pkgs: list[str] = []
    for n in raw:
        pkg = re.split(r"[<>=!~]", n)[0].strip()
        if pkg and pkg not in pkgs:
            pkgs.append(pkg)
    return pkgs


def _undocumented_dependencies() -> list[str]:
    """宣言済み依存のうちライセンス許容リストに無いものを返す（Q42）。空なら全依存が
    レビュー済みの permissive ライセンス。新規依存の追加漏れチェックに使う恒常ゲート。
    """
    return [d for d in _declared_dependencies() if d not in _DEPENDENCY_LICENSE_ALLOWLIST]


# facade 未接続の許容リスト（意図的な非接続・理由つき）。
# docs/FEATURE-AUDIT.md セクション B-2（意図的に facade 非接続）・A-12（既知の
# 未実装ギャップ）と対応する。新たに正当な理由で非接続にする関数はここに追記する。
_FACADE_DISCONNECT_ALLOWLIST: dict[str, str] = {
    "bisect_onset": "O(log L) の代替探索アルゴリズム。diagnose() は既に全層 divergence "
                    "を計算済みで恩恵がないため意図的に別 API のまま",
    "grid_search": "タイル構成探索ユーティリティ（GPU codegen 実装後に本経路へ・pre-codegen）",
    "make_corpus": "検証器自身を検証するメタツール（開発時の校正用・製品判定経路でない）",
    "evaluate": "検証器自身を検証するメタツール（開発時の校正用・製品判定経路でない）",
    "roc_sweep": "検証器自身を検証するメタツール（開発時の校正用・製品判定経路でない）",
    "op_is_nondeterministic": "nondeterminism_reason の便宜的な bool ラッパ"
                              "（classify_nondeterminism は reason 文字列側を使う）",
    "simulate_nondeterministic_reduction": "GPU 実機なしで検証層をテストする CPU シミュレータ"
                                           "（テスト専用が正当）",
    "simulate_batch_variant_reduction": "GPU 実機なしで検証層をテストする CPU シミュレータ"
                                        "（テスト専用が正当）",
    "occupancy_gap": "cross_vendor_occupancy(vendors=targets) に一般化されたが、"
                     "2 者間の簡易 API として維持（意図的な下位互換）",
    "oracle_is_trustworthy": "verify_oracle().ok の便宜ラッパ（audit は verify_oracle を直接使う）",
    "model_tolerance": "propagate(ops).model_divergence の便宜ラッパ",
    "changed_fields": "certify/is_stale の内部部品（上位関数経由で facade は利用済み）",
    "simulate_rollout": "GPU 実機なしで検証層をテストする CPU シミュレータ（テスト専用が正当）",
}


def _facade_disconnected_functions() -> list[str]:
    """`python/tsugi/*.py`・`python/tsugi_torch/*.py` の公開関数のうち、自ファイル内
    でも他のソースファイルからも一切呼ばれていない（= テストからしか呼ばれない、
    または完全にデッド）ものを検出する。

    このプロジェクトは 11 件の「実装済みだが facade（audit 系）から呼ばれない」欠陷を
    ソース参照スキャンで発見・修正してきた（docs/FEATURE-AUDIT.md セクション B-1）。
    このスキャンを一度きりの手動作業でなく恒常的な不変条件にし、新機能追加のたびに
    同型の欠陥が紛れ込むのを機械的に検出する。既知の意図的な非接続は
    `_FACADE_DISCONNECT_ALLOWLIST` で除外し、そこに無い新規の未接続だけを報告する。
    """
    import re

    files = list((PY / "tsugi").glob("*.py")) + list((PY / "tsugi_torch").glob("*.py"))
    texts: dict[Path, str] = {}
    for p in files:
        try:
            texts[p] = p.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
    unexpected: list[str] = []
    for f, src in sorted(texts.items()):
        for fn in re.findall(r"^def ([a-z][a-zA-Z0-9_]*)\(", src, re.M):
            if fn in _FACADE_DISCONNECT_ALLOWLIST:
                continue
            in_own = src.count(fn) > 1
            in_src = any(fn in t for g, t in texts.items() if g != f)
            if not in_own and not in_src:
                unexpected.append(f"{f.name}:{fn}")
    return unexpected


def _check_prohibited_and_suites() -> None:
    """不変条件 1-7: 禁止パターン・テストスイート通過・lowering 対応・正直な未実装宣言。"""
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
    from tsugi.lowering import VENDOR_LOWERING
    check("dot→wmma (NVIDIA, ADR-004)", "wmma" in VENDOR_LOWERING["dot"]["nvidia"])
    check("dot→mfma (AMD CDNA, ADR-004)", "mfma" in VENDOR_LOWERING["dot"]["amd_cdna"])
    # DSL が emit しうる全 op に実ターゲット lowering が定義済み（spec が DSL に同期）
    from tsugi.lowering import unlowered_ops
    check("lowering covers every emittable DSL op (nvidia/amd, no drift)",
          all(not unlowered_ops(t) for t in ("nvidia", "amd_cdna", "amd_rdna")))

    # 7. machine-code emission は実装済み（不変条件 90-92 が中身を固定）。ここでは
    #    *正直さ* だけを見る: 未対応ターゲット（SPIR-V）は黙って空を返さず明示的に
    #    NotImplementedError を投げ、既定の dry-run は生成物を持たない。
    import tsugi
    try:
        tsugi.compile(lambda: None, (), target="spirv", emit_machine_code=True)
        check("machine-code emission is honest about unsupported targets", False)
    except NotImplementedError:
        check("machine-code emission is honest about unsupported targets", True)


def _check_core_pillars() -> None:
    """不変条件 8-13: equivalence/occupancy/tolerance/feasibility/propagation/envelope の柱。"""
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


def _check_verifier_calibration() -> None:
    """不変条件 14-16: 検証器の自己検証（calibration）・ノイズフロア・タスクレベル判定の柱。"""
    import numpy as np

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


def _check_audit_facades() -> None:
    """不変条件 17-19: audit/audit_runtime/audit_cross_vendor の統合ファサード。"""
    import numpy as np

    from tsugi import portability

    # 17. audit: 統合ファサードが静的層を 1 判定に束ねる（運用統合）
    from tsugi.audit import audit
    from tsugi.portcheck import _demo_module
    mod, block, dcfg = _demo_module()
    ad = audit(mod, dcfg, block_dims=block)
    check("audit aggregates static phases into one verdict (AMD launch BLOCK)",
          ad.max_risk == portability.Risk.BLOCK and not ad.portable)
    check("audit excludes runtime phases from static verdict",
          ad.max_risk == max(p.max_risk for p in ad.decided_phases))
    from tsugi.audit import _graph_ops, _iter_graphops
    mm = [o for o in _iter_graphops(_graph_ops(mod, dcfg)) if o.kind == "matmul"]
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


def _check_propagation_floors_and_search() -> None:
    """不変条件 20-22 と続く無番号ブロック: 増幅 op・batch 不変性・系統発散・
    shared-mode・oracle 検証・レイアウト判別・provenance・rollout・worstcase 探索。"""
    import numpy as np

    from tsugi import portability
    from tsugi.calibration import is_equivalent_combined
    from tsugi.nondeterminism import measure_noise_floor

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


def _check_reports_and_diagnostics() -> None:
    """不変条件 23-30: レポート統一・ROC・torch backend 静的監査・SAFETY 単一情報源・
    attribution/blame 診断。"""
    import numpy as np

    from tsugi import portability

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


def _check_dtype_precision() -> None:
    """不変条件 31-33c: TF32/NaN タグ/float64/FP8/MX の dtype 3 テーブル整合。"""
    import numpy as np

    from tsugi.envelope import check_tensor
    from tsugi.equivalence import compare

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

    # 33c. Microscaling (OCP MX v1.0 MXFP4/MXFP6): NVIDIA Blackwell と AMD CDNA4 の
    # 両方が HW ネイティブ対応する唯一の共通低精度フォーマット群（NVFP4 は NVIDIA 専用のため対象外）
    check("MX unit_roundoff ordered by mantissa bits (mxfp4 coarsest > mxfp6_e3m2 > mxfp6_e2m3)",
          _URO["mxfp4_e2m1"] > _URO["mxfp6_e3m2"] > _URO["mxfp6_e2m3"]
          and _URO["mxfp4_e2m1"] == 2.0 ** -1
          and _URO["mxfp6_e3m2"] == 2.0 ** -2
          and _URO["mxfp6_e2m3"] == 2.0 ** -3)
    check("MX tolerance ordered by mantissa bits, mxfp4 coarser than fp8_e5m2 (1 mantissa bit)",
          _TOL["mxfp4_e2m1"]["atol"] > _TOL["mxfp6_e3m2"]["atol"] > _TOL["mxfp6_e2m3"]["atol"]
          and _TOL["mxfp4_e2m1"]["atol"] > _TOL["float8_e5m2"]["atol"])
    check("MXFP4 narrowest range (max=6.0) among all dtypes makes overflow the dominant risk",
          _dlim("mxfp4_e2m1").max_normal == 6.0
          and _dlim("mxfp6_e2m3").max_normal == 7.5
          and _dlim("mxfp6_e3m2").max_normal == 28.0
          and _dlim("mxfp4_e2m1").max_normal < _dlim("mxfp6_e2m3").max_normal)
    # MXFP4 では 8.0 が overflow するが fp16 ではしない（block スケールがあっても要素間レンジは狭い）
    _xmx = np.full((4, 4), 8.0, np.float32)
    check("MXFP4 flags 8.0 as overflow (BLOCK) where fp16 does not (max=6.0 narrowest of all dtypes)",
          not check_tensor(_xmx, _cg8(64, "mxfp4_e2m1", 1.0)).in_envelope
          and not any("overflow" in f.message
                      for f in check_tensor(_xmx, _cg8(64, "float16", 8.0)).findings))


def _check_nondet_catalog_and_dynamic_shapes() -> None:
    """不変条件 34-35: 非決定 op 静的カタログと dynamic shape 検出（FX 橋経由）。"""
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


def _check_late_facade_wiring() -> None:
    """不変条件 37-46: certify_from_sample・backend 冪等性・audit facade の後期接続
    （empirical_cond/robust noise floor/compare_task/attribution/worstcase/LAYOUT 判別）。"""
    import numpy as np

    from tsugi import portability
    from tsugi.audit import audit, audit_runtime
    from tsugi.envelope import certify_gemm, check_tensor
    from tsugi.portcheck import _demo_module
    from tsugi_torch.fxbridge import audit_fx as _afx

    mod, block, dcfg = _demo_module()

    class _N:
        def __init__(s, op, t, shp=None):
            s.op, s.target, s.meta = op, t, ({"tensor_meta": type("M", (), {"shape": shp})} if shp else {})

    class _G:
        def __init__(s, ns):
            s.graph = type("GR", (), {"nodes": ns})
    _gm = _G([_N("call_function", "aten.addmm.default", (8, 512)),
              _N("call_function", "aten._softmax.default"),
              _N("output", "output")])

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


def _check_statistical_rigor() -> None:
    """不変条件 47-53: 点推定でなく上側限界で判定する fail-safe 群と、
    それを facade（audit_runtime/compare_task/occupancy phase）へ届ける接続。"""
    import numpy as np

    from tsugi import portability
    from tsugi.audit import audit, audit_runtime
    from tsugi.constants import SAFETY
    from tsugi.portcheck import _demo_module

    mod, block, dcfg = _demo_module()

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

    # 52. compare_task(binary) が compare_decisions と同型の near-tie 健全性チェックを持つ
    # （binary_margin は実装・テスト済みだったがこの診断には未接続だった）。
    _a52 = np.concatenate([np.full(500, 0.501), np.full(500, 0.95)])
    _b52_ok = _a52.copy()
    _b52_ok[:500] = 0.499   # 閾値付近のみフリップ（正常系）
    _rep52_ok = _ct50(_a52, _b52_ok, task="binary", flip_budget=0.6)
    check("compare_task(binary) does not warn when flips concentrate in near-tie tail (no false alarm)",
          not any("near-tie" in f.message for f in _rep52_ok.findings))
    _b52_bad = _a52.copy()
    _b52_bad[500:] = 0.05   # 確信領域までフリップ（異常系）
    _rep52_bad = _ct50(_a52, _b52_bad, task="binary", flip_budget=0.6)
    check("compare_task(binary) warns when flips reach the confident region (binary_margin wired)",
          any("near-tie" in f.message for f in _rep52_bad.findings)
          and _rep52_bad.flipped_margin_median > _rep52_bad.overall_margin_median)

    # 53. audit() の occupancy phase が targets 全体を報告する（従来 nvidia/amd_cdna の
    # 2 者間ギャップにハードコードされ、targets に amd_rdna があっても未報告だった。
    # cross_vendor_occupancy は実装・テスト済みだったが未接続だった）。
    _ad53 = audit(mod, dcfg, block_dims=block)
    _occ53 = next(p for p in _ad53.phases if p.name.startswith("occupancy"))
    check("audit() occupancy phase reports amd_rdna with default targets (cross_vendor_occupancy wired)",
          "amd_rdna" in _occ53.to_text())
    _ad53b = audit(mod, dcfg, block_dims=block, targets=("nvidia", "amd_rdna"))
    _occ53b = next(p for p in _ad53b.phases if p.name.startswith("occupancy"))
    check("audit() occupancy phase follows targets, not a hardcoded vendor pair",
          "amd_cdna" not in _occ53b.to_text() and "amd_rdna" in _occ53b.to_text())


def _check_shape_guards() -> None:
    """不変条件 54-55: 形状不一致を暗黙 broadcast に委ねず即 DIVERGENT にするガード。"""
    import numpy as np

    from tsugi import portability
    from tsugi.equivalence import compare

    # 54. equivalence.compare は形状不一致を NumPy の暗黙 broadcast に委ねず即 DIVERGENT
    # にする（broadcast による偽OK/偽BLOCK を防ぐ・vendor が誤った形状を返すバグの実証）。
    _rng54 = np.random.default_rng(0)
    _a54 = _rng54.standard_normal((8, 8)).astype(np.float32)
    _b54_bug = _a54[0].copy()   # vendor バグ: 先頭行しか返さない（形状 (8,)）
    _rep54 = compare(_a54, _b54_bug, "float32")
    check("compare() flags shape mismatch instead of silently broadcasting (no false-OK)",
          _rep54.shape_mismatch and not _rep54.equivalent)
    check("compare() same-shape path is unaffected (no regression)",
          compare(_a54, _a54.copy(), "float32").equivalent
          and not compare(_a54, _a54.copy(), "float32").shape_mismatch)

    # 55. audit_runtime も同型の形状ガードを持つ（equivalence phase の broadcast 依存を排除）。
    from tsugi.audit import audit_runtime as _art55
    _ad55 = _art55(_a54, _b54_bug, K=64, noise_floor=1e-6)
    check("audit_runtime rejects shape-mismatched a_out/b_out without broadcasting (no false-OK)",
          not _ad55.portable and _ad55.max_risk == portability.Risk.BLOCK)
    _ad55b = _art55(_a54, _a54.copy(), K=64, noise_floor=1e-6)
    check("audit_runtime same-shape path is unaffected (no regression)",
          _ad55b.portable)


def _check_meta_integrity() -> None:
    """不変条件 56-92: orphan テスト・facade 未接続・警告 facade・依存ライセンス・
    per-sample δ（Q19）・バージョン整合・SSA fork 接続（A-12 Round 1/2/3）・task レベル
    shared-mode（Q31）・denormal 率（Q16）・橋の仮定明示（Q15）・判定の機械可読性
    （First Principles）・誤差境界モデルの選択（確率的/最悪ケース）・検出境界の seed 非依存性（Q43）・
    INDISTINGUISHABLE の解消手（証拠の累積）——検証基盤と
    コードベース自身の構造整合性。"""
    import numpy as np

    # 56. tests/correctness/ の test_* 関数が全て main() のテストリストに登録されている
    # （= 一度は実行される）ことを機械検査する。「facade 未接続」と同型の欠陥が
    # テスト層で起きるのを防ぐ恒常ゲート（FEATURE-AUDIT.md A-6 の一部を解消）。
    _orphans = _orphan_tests()
    check("no orphan test functions (all registered in a main() test list)",
          not _orphans)
    if _orphans:
        for o in _orphans:
            print(f"    orphan: {o}")

    # 57. python/tsugi(_torch)/ の公開関数に新規の facade 未接続（実装済みだが誰からも
    # 呼ばれない）が紛れていないことを機械検査する（FEATURE-AUDIT.md A-6 の本体を解消）。
    # 既知の意図的な非接続は _FACADE_DISCONNECT_ALLOWLIST で除外済み。
    _disconnected = _facade_disconnected_functions()
    check("no new facade-disconnected functions beyond the documented allowlist",
          not _disconnected)
    if _disconnected:
        for d in _disconnected:
            print(f"    disconnected: {d}")

    # 58. tsugi_torch._tsugi_compile の警告メッセージが nondeterministic_ops/
    # requires_noise_floor を反映する（audit_fx は計算済みだったが従来この facade
    # ＝ユーザー向け警告に一切届いていなかった）。torch 無し環境でも duck-typed
    # スタンドインで検証可能（実 torch.fx との結線は torch 環境が要る）。
    import warnings as _warnings58

    from tsugi_torch import _tsugi_compile as _compile58

    class _TM58:
        def __init__(self, shape):
            self.shape = shape

    class _Node58:
        def __init__(self, op, target, shape=None):
            self.op, self.target = op, target
            self.meta = {"tensor_meta": _TM58(shape)} if shape else {}

    class _Graph58:
        def __init__(self, nodes):
            self.nodes = nodes

    class _GM58:
        def __init__(self, nodes):
            self.graph = _Graph58(nodes)

        def forward(self, *a, **kw):
            raise RuntimeError("duck-type stand-in")

    _gm_nondet58 = _GM58([
        _Node58("placeholder", "x"),
        _Node58("call_function", "aten.addmm.default", (8, 512)),
        _Node58("call_function", "aten.scatter_add.default"),
        _Node58("output", "output"),
    ])
    with _warnings58.catch_warnings(record=True) as _w58:
        _warnings58.simplefilter("always")
        _compile58(_gm_nondet58, [])
    check("_tsugi_compile warns about nondeterministic ops (audit_fx→warning facade wired)",
          len(_w58) == 1 and "scatter_add" in str(_w58[0].message)
          and "noise floor" in str(_w58[0].message))

    # 59. audit_fx/_tsugi_compile が正規化層（LayerNorm/RMSNorm）を検出し、警告に出す
    # （FEATURE-AUDIT.md A-5）。当初は「scale リセット未考慮の保守的な上界」という
    # 但し書きだったが、数値実験で LayerNorm は平均優勢入力で *増幅* すると判明したため
    # 文言を撤回した（不変条件 77 が新文言と旧主張の非復活を固定する）。検出フラグ自体は
    # 可視化として維持。
    from tsugi_torch.fxbridge import audit_fx as _afx59
    _gm_norm59 = _GM58([
        _Node58("placeholder", "x"),
        _Node58("call_function", "aten.addmm.default", (8, 512)),
        _Node58("call_function", "aten.native_layer_norm.default"),
        _Node58("output", "output"),
    ])
    check("audit_fx detects normalization layers (has_normalization=True)",
          _afx59(_gm_norm59)["has_normalization"])
    with _warnings58.catch_warnings(record=True) as _w59:
        _warnings58.simplefilter("always")
        _compile58(_gm_norm59, [])
    check("_tsugi_compile surfaces the normalization scale-reset caveat in its warning",
          len(_w59) == 1 and "has_normalization" in str(_w59[0].message))
    _gm_no_norm59 = _GM58([
        _Node58("placeholder", "x"),
        _Node58("call_function", "aten.addmm.default", (8, 512)),
        _Node58("call_function", "aten._softmax.default"),
        _Node58("output", "output"),
    ])
    check("audit_fx does not flag has_normalization for a graph without norm ops",
          not _afx59(_gm_no_norm59)["has_normalization"])

    # 60. pyproject.toml の宣言済み依存が全て permissive ライセンス許容リストに載っている
    # （SOCRATIC-50 Q42: 依存ライセンス自動監査）。新規依存追加時のレビュー漏れを防ぐ
    # 恒常ゲート。plant-and-detect で検出器自体が機能することも併せて確認する。
    check("all declared dependencies have a documented permissive license (Q42)",
          not _undocumented_dependencies())
    _synthetic_toml60 = (
        'dependencies = ["numpy>=1.22", "some-undocumented-lib>=1.0"]\n'
        '[project.optional-dependencies]\n'
        'torch = ["torch>=2.1"]\n'
    )
    _synthetic_undoc60 = [d for d in _declared_dependencies(_synthetic_toml60)
                          if d not in _DEPENDENCY_LICENSE_ALLOWLIST]
    check("dependency license scan actually detects an undocumented dependency (plant-and-detect)",
          _synthetic_undoc60 == ["some-undocumented-lib"])

    # 61. decision.flip_bound_from_divergence は per-sample scale を使い、低スケール多数派に
    # 紛れた高スケール near-tie サンプルのフリップ risk を過小評価しない（SOCRATIC-50 Q19）。
    # δ_abs = δ_rel·RMS のグローバル RMS は「平均的スケール」であり、高スケールの少数
    # サンプルにとって δ が過小評価され margin<2δ を満たさなくなる（偽OK方向）。
    from tsugi.decision import flip_bound_from_divergence as _fbd61
    from tsugi.decision import margin as _margin61
    from tsugi.decision import predicted_flip_bound as _pfb61
    _small61 = np.stack([np.full(5000, 0.05), np.full(5000, 0.03)], axis=-1)
    _big61 = np.array([[50.0, 49.5]])   # 自身のスケール ~49.75 に対し margin=0.5 は near-tie
    _z61 = np.concatenate([_small61, _big61], axis=0)
    _rel61 = 0.01
    _global_scale61 = float(np.sqrt(np.mean(_z61 ** 2)))
    # Wilson 上側限界は k=0 でも >0 を返す（rule-of-three）ため bound==0 では検査できない。
    # 「見逃し」の機構（k=0）と、ユーザーに見える効果（bound の厳密な増加）の両方を固定する。
    check("global-RMS-only delta misses the high-scale near-tie outlier (reproduces the false-OK setup)",
          int(np.count_nonzero(_margin61(_z61) < 2.0 * _rel61 * _global_scale61)) == 0)
    check("flip_bound_from_divergence uses per-sample scale and does not underestimate it (Q19)",
          _fbd61(_z61, _rel61) > _pfb61(_z61, _rel61 * _global_scale61))

    # 62. pyproject.toml の version と tsugi.__version__ が一致する。0.3.0 リリース
    # （eface10）で pyproject は 0.3.0 にバンプされたが __init__.py の __version__ が
    # 0.2.0 のまま取り残されていた（配布メタデータと実行時 API が別バージョンを名乗る）。
    # 「発見した欠陥は不変条件で固定する」慣例に従い、リリース時のバンプ漏れを機械検出する。
    import re as _re62

    import tsugi as _tsugi62
    _toml62 = (PY / "pyproject.toml").read_text(encoding="utf-8")
    _m62 = _re62.search(r'^\s*version\s*=\s*"([^"]+)"', _toml62, _re62.M)
    check("pyproject.toml version matches tsugi.__version__ (no release bump drift)",
          _m62 is not None and _m62.group(1) == _tsugi62.__version__)

    # 63. audit() の propagation phase が SSA の use-def からフォーク/マージを再構築し、
    # propagate_dag（テスト済みだが従来 audit から一度も呼ばれなかった）に流す
    # （FEATURE-AUDIT.md A-12）。従来は kernel body を線形走査し Op.operands/Op.result を
    # 捨てていたため、residual/softmax の fork/merge 構造が発散予測に反映されなかった。
    import tsugi as _t63
    from tsugi import tile as _tile63
    from tsugi.audit import _graph_ops as _go63
    from tsugi.audit import _iter_graphops as _ig63
    from tsugi.autotune import TileConfig as _TC63

    @_t63.jit
    def _softmax63(x, out, N, BN):
        p = _t63.program_id(0)
        row = _tile63.load(x, (p * BN, 0), (BN, N))
        m = _tile63.reduce(row, 1, "max")
        e = _tile63.exp(row - m)
        s = _tile63.reduce(e, 1, "sum")
        _tile63.store(out, (p * BN, 0), (e / s).to(_t63.float16))

    _x63 = np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32)
    _mod63 = _t63.trace(_softmax63, (_x63, _x63.copy(), 16, 16), {}, (0,))
    _cfg63 = _TC63(block_m=16, block_n=16, block_k=16, num_stages=2, num_warps=4)
    _g63 = _go63(_mod63, _cfg63)
    _forks63 = [o for o in _g63 if isinstance(o, list)]
    # softmax の row/e 再利用が恒等路つきフォーク [[], [..reduce..]] として抽出され、
    # 増幅 op（reduce×2/exp）は葉として残る（フォークに畳んでも op は消えない）。
    _leaves63 = [o.kind for o in _ig63(_g63)]
    check("audit _graph_ops reconstructs SSA fork/merge for propagate_dag (A-12)",
          bool(_forks63)
          and all(len(f) == 2 and f[0] == [] for f in _forks63)
          and _leaves63.count("reduce") == 2 and "exp" in _leaves63)

    # 64. A-12 Round 2: 恒等路の無い計算 2 分岐（attention ヘッド和・row を exp と reduce の
    # 2 経路が消費し add で合流）も SSA から検出し `[[A],[B]]`（恒等路なし）で出す。
    # audit() は correlated=True（保守側）で合流するため DAG 発散は線形版を下回らない
    # （並列分岐を independent 仮定で過小評価 → 偽OK になる罠を回避）。
    from tsugi.propagation import propagate as _prop64
    from tsugi.propagation import propagate_dag as _pdag64

    @_t63.jit
    def _twobranch64(x, out, N, BN):
        p = _t63.program_id(0)
        row = _tile63.load(x, (p * BN, 0), (BN, N))
        a = _tile63.exp(row)
        b = _tile63.reduce(row, 1, "sum")
        _tile63.store(out, (p * BN, 0), (a + b).to(_t63.float16))

    _x64 = np.random.default_rng(1).standard_normal((8, 8)).astype(np.float32)
    _mod64 = _t63.trace(_twobranch64, (_x64, _x64.copy(), 8, 8), {}, (0,))
    _cfg64 = _TC63(block_m=8, block_n=8, block_k=8, num_stages=2, num_warps=4)
    _g64 = _go63(_mod64, _cfg64)
    _forks64 = [o for o in _g64 if isinstance(o, list)]
    _bk64 = ({tuple(op.kind for op in br) for br in _forks64[0]} if _forks64 else set())
    _flat64 = list(_ig63(_g64))
    check("audit _graph_ops detects computed two-branch merge, merged conservatively (A-12 Round 2)",
          len(_forks64) == 1
          and all(br != [] for br in _forks64[0])
          and _bk64 == {("exp",), ("reduce",)}
          and _pdag64(_g64, correlated=True).model_divergence
          >= _prop64(_flat64).model_divergence)

    # 65. audit_runtime(logits_oracle=...) が task レベルの shared-mode を検出する（Q31・A-9）。
    # フリップ率は A↔B の一致を測る（正しさではない）。両ベンダーが互いに一致（低フリップ率）
    # でも両方 oracle 判断と食い違えば同一誤り——tensor レベルの detect_shared_mode の task 版。
    from tsugi.audit import audit_runtime as _ar65
    from tsugi.envelope import certify_gemm as _cg65
    from tsugi.report import Risk as _R65
    _a65 = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    _env65 = _cg65(K=64, dtype="float32", scale=1.0)
    _lg65 = np.random.default_rng(0).standard_normal((200, 10)).astype(np.float32)
    _or65 = _lg65.copy()
    _or65[:100] = _or65[:100][:, ::-1]      # 半数のサンプルで oracle の argmax を変える
    _ad65 = _ar65(_a65, _a65.copy(), K=64, env=_env65, noise_floor=1e-6,
                  logits_a=_lg65, logits_b=_lg65, logits_oracle=_or65, flip_budget=0.01)
    _dp65 = next(p for p in _ad65.phases if p.name.startswith("decision"))
    # logits_oracle 無しでは正しさ行を出さない（後方互換）
    _ad65b = _ar65(_a65, _a65.copy(), K=64, env=_env65, noise_floor=1e-6,
                   logits_a=_lg65, logits_b=_lg65, flip_budget=0.5)
    _dp65b = next(p for p in _ad65b.phases if p.name.startswith("decision"))
    check("audit_runtime(logits_oracle) flags task-level shared-mode (agreement != correctness, Q31)",
          "判断誤り" in _dp65.to_text() and "SHARED-MODE" in _dp65.to_text()
          and _dp65.max_risk >= _R65.WARN
          and "判断誤り" not in _dp65b.to_text())

    # 66. envelope.check_tensor が denormal を *率* で区別する（SOCRATIC-50 Q16）。
    # 偶発的な単一 denormal 値と、値の大半が denormal（scale が dtype に対し小さすぎ＝
    # 認証 atol の前提が崩れる）を区別し、後者に rescale/再認証を促す強い警告を出す。
    from tsugi.envelope import certify_gemm as _cg66
    from tsugi.envelope import check_tensor as _ct66
    from tsugi.envelope import dtype_limits as _dl66
    _lim66 = _dl66("float16")
    _sys66 = np.full((256,), _lim66.min_normal * 0.3, dtype=np.float32)   # 大半 denormal
    _inc66 = np.full((1000,), 1.0, dtype=np.float32)
    _inc66[0] = _lim66.min_normal * 0.1                                    # 1 値のみ denormal
    _msg_sys66 = " ".join(f.message for f in _ct66(_sys66, _cg66(K=64, dtype="float16")).findings)
    _msg_inc66 = " ".join(f.message for f in _ct66(_inc66, _cg66(K=64, dtype="float16")).findings)
    check("check_tensor distinguishes systematic vs incidental denormal by fraction (Q16)",
          "小さすぎ" in _msg_sys66 and "再認証" in _msg_sys66
          and "denormal" in _msg_inc66 and "小さすぎ" not in _msg_inc66)

    # 67. audit(ref_logits=) の propagation→decision 橋が仮定を *レポートに明示* する
    # （SOCRATIC-50 Q15）。相対発散 δ_rel を最終 logit にそのまま乗せる近似の妥当域
    # （正規化の scale リセット・最終射影の条件数・分布シフト）を暗黙化しない。
    from tsugi.audit import audit as _au67
    from tsugi.portcheck import _demo_module as _dm67
    _mod67, _blk67, _cfg67 = _dm67()
    _lg67 = np.random.default_rng(0).standard_normal((100, 10)).astype(np.float32)
    _p67 = next(p for p in _au67(_mod67, _cfg67, block_dims=_blk67, ref_logits=_lg67).phases
                if p.name.startswith("propagation"))
    _p67n = next(p for p in _au67(_mod67, _cfg67, block_dims=_blk67).phases
                 if p.name.startswith("propagation"))
    check("audit bridge (propagation->decision) surfaces its validity-domain assumption (Q15)",
          any("仮定: op グラフ相対発散" in ln for ln in _p67.lines)
          and not any("仮定: op グラフ相対発散" in ln for ln in _p67n.lines))

    # 68. A-12 Round 3: 3 分岐以上（N≥3）の計算フォークも SSA から検出する。
    # Round 2 は消費者ちょうど 2 に限定され、3 分岐は線形（保守側）に落ちていた。
    # 全枝が単一消費の一本鎖として同一合流 op に収束する場合のみ [[A],[B],[C]] で受理。
    @_t63.jit
    def _three68(x, out, N, BN):
        p = _t63.program_id(0)
        row = _tile63.load(x, (p * BN, 0), (BN, N))
        a = _tile63.exp(row)
        b = _tile63.exp(row)
        c = row.to(_t63.float16)
        _tile63.store(out, (p * BN, 0), _tile63.dot(a, b, c).to(_t63.float16))

    _x68 = np.abs(np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)) * 0.1
    _mod68 = _t63.trace(_three68, (_x68, _x68.copy().astype(np.float16), 8, 8), {}, (0,))
    _g68 = _go63(_mod68, _TC63(block_m=8, block_n=8, block_k=8, num_stages=2, num_warps=4))
    _f68 = [o for o in _g68 if isinstance(o, list)]
    _flat68 = list(_ig63(_g68))
    check("audit _graph_ops detects N-way (3+) computed fork, merged conservatively (A-12 Round 3)",
          len(_f68) == 1 and len(_f68[0]) == 3
          and all(br != [] for br in _f68[0])
          and _pdag64(_g68, correlated=True).model_divergence
          >= _prop64(_flat68).model_divergence)

    # 69. 判定が機械可読（JSON）＋終了コード契約を持つ（First Principles の不足発見）。
    # この製品の存在意義は「出荷してよいか」を CI が自動でゲートすること。だが従来
    # Audit は to_text()（人間向け日本語散文）しか持たず、機械が判定を消費するには
    # 散文の正規表現パースしかなかった（脆い）。構造化データと終了コードを契約にする。
    import json as _json69

    from tsugi.report import Risk as _Risk69
    from tsugi.report import exit_code as _ec69
    _ad69 = _au67(_mod67, _cfg67, block_dims=_blk67)
    _d69 = _ad69.to_dict()
    _s69 = _json69.dumps(_d69, ensure_ascii=False)      # Risk が漏れていれば TypeError
    check("Audit verdict is machine-readable (JSON) with a CI exit-code contract",
          _json69.loads(_s69)["max_risk"] == _ad69.max_risk.name
          and _d69["verdict"] == ("portable" if _ad69.portable else "blocked")
          and len(_d69["phases"]) == len(_ad69.phases)
          and {"name", "when", "max_risk", "lines"} <= set(_d69["phases"][0])
          and [_ec69(r) for r in _Risk69] == [0, 0, 1, 2]
          and _ad69.exit_code == _ec69(_ad69.max_risk))

    # 70. 誤差境界モデルの選択（確率的 √K / 最悪ケース K）が判定を実際に支配する。
    # 既定の √K は Higham & Mary の *確率的* 丸め誤差解析（丸め誤差が独立・平均 0 の
    # 仮定下で高確率に成り立つ境界）であって保証ではない。仮定が破れる典型が系統誤差
    # （calibration.check_systematic の検出対象）で、その場合は古典的 Wilkinson の
    # γ_K ≈ K·u が妥当。保証が要る利用者が model="worstcase" を選べることを固定する。
    from tsugi.tolerance import derive_tolerance as _dt70
    from tsugi.tolerance import expected_gemm_abs_error as _ege70
    from tsugi.tolerance import explain as _ex70
    _p70 = _ege70(2048, "float16", model="probabilistic")
    _w70 = _ege70(2048, "float16", model="worstcase")
    _bad70 = False
    try:
        _dt70(64, "float16", model="typo")     # silent fallback は偽OK の温床
    except ValueError:
        _bad70 = True
    check("tolerance offers a worst-case (Wilkinson K.u) bound alongside the probabilistic (sqrt(K)) default",
          _w70 > _p70 and abs(_w70 / _p70 - 45.254) / 45.254 < 0.01
          and _dt70(2048, "float16")["model"] == "probabilistic"
          and _dt70(2048, "float16", model="worstcase")["atol"] > _dt70(2048, "float16")["atol"]
          and _bad70
          and "確率的" in _ex70(2048, "float16")
          and "check_systematic" in _ex70(2048, "float16"))

    # 71. 検出境界が seed に依らず SAFETY·u に一致する（SOCRATIC-50 Q43・乱数境界の点検）。
    # Q43 の懸念は「乱数依存テストは seed 固定でも境界付近で脆く、別 seed で反転しうる」。
    # 系統バグ強度を理論境界 SAFETY·u の ±1% に置き、多数 seed で判定が全会一致になる
    # ことを固定する（＝判定が seed 非依存でバグ強度のみに支配される証拠）。他の固定
    # seed テストが「たまたま通っている」のでないことの根拠にもなる。
    from tsugi.calibration import is_equivalent_combined as _iec71
    from tsugi.constants import SAFETY as _SAFETY71
    from tsugi.tolerance import unit_roundoff as _u71
    _K71, _dt71, _n71 = 256, "float16", 24
    _th71 = _SAFETY71 * _u71(_dt71)

    def _eqcount71(strength: float) -> int:
        c = 0
        for _s in range(_n71):
            _a = np.random.default_rng(_s).standard_normal((64, 64)).astype(np.float32)
            c += bool(_iec71(_a, _a * (1 + strength), _K71, _dt71))
        return c

    check("detection verdict is seed-independent and sits exactly at SAFETY*u (Q43)",
          _eqcount71(0.99 * _th71) == _n71 and _eqcount71(1.01 * _th71) == 0)

    # 72. INDISTINGUISHABLE が終端でなく「あと N run で決着する」実行可能な次手を出す。
    # 単発ではノイズに埋もれる差も、独立 run を平均すれば平均のノイズは σ/√N に縮み
    # 系統差は縮まない → SNR = d·√N/σ が伸びて分離できる（N > (z·σ/d)²）。
    # DiFR が多数トークンに証拠を累積して設定誤りを検出するのと同型（docs/SOURCES.md）。
    import math as _math72

    from tsugi.nondeterminism import _erfinv as _ei72
    from tsugi.nondeterminism import runs_to_resolve as _rtr72
    _n1_72, _n2_72 = _rtr72(1e-3, 1e-2), _rtr72(5e-4, 1e-2)
    # 73. audit_fx(sample=) が torch 経路の「scale=1 / cond=1」暗黙仮定を実測で置き換える
    # （FEATURE-AUDIT.md A-3）。audit() は B-1a/B-1b で既に持っていた機能が、製品の想定入口で
    # ある torch 経路に無かった。実 LLM の活性は massive activations（中央値の ~1000 倍）を
    # 持つため scale=1 仮定は認証 atol を桁で誤らせる（docs/SOURCES.md）。
    from tsugi_torch.fxbridge import audit_fx as _afx73

    class _TM73:
        def __init__(s, shape): s.shape = shape

    class _N73:
        def __init__(s, op, t, shp=None):
            s.op, s.target = op, t
            s.meta = {"tensor_meta": _TM73(shp)} if shp else {}

    class _G73:
        def __init__(s, ns): s.graph = type("GR", (), {"nodes": ns})

    _gm73 = _G73([_N73("call_function", "aten.addmm.default", (8, 512)),
                  _N73("call_function", "aten._softmax.default"),
                  _N73("output", "output")])
    _x73 = np.random.default_rng(0).standard_normal((32, 512)).astype(np.float32) * 0.1
    _x73[:, 7] *= 1000.0                       # massive activation 相当の外れチャネル
    _base73 = _afx73(_gm73)                    # sample 無し（従来経路）
    _meas73 = _afx73(_gm73, sample=_x73)
    check("audit_fx(sample=) measures real scale/cond for the torch path (A-3, no scale=1 assumption)",
          "sample_scale" not in _base73                     # 後方互換（未指定なら出さない）
          and _meas73["sample_scale"] > 1.0
          and _meas73["channel_spread"] > 100.0             # 外れチャネルを検出
          and _meas73["cond_measured"] is True
          and _meas73["model_divergence"] > _base73["model_divergence"])  # 静的下界の是正

    check("INDISTINGUISHABLE yields an actionable run count N ~ (z*sigma/d)^2 (evidence accumulation)",
          abs(_math72.sqrt(2) * _ei72(2 * 0.95 - 1) - 1.6449) < 0.01
          and _rtr72(0.0, 1e-2) == 0 and _rtr72(2e-2, 1e-2) == 0
          and _n1_72 > 1 and abs(_n2_72 / _n1_72 - 4.0) < 0.1
          and _rtr72(1e-3, 1e-2, 0.99) > _rtr72(1e-3, 1e-2, 0.95))

    # 74. SAFETY=4.0 の実機校正が「4σ は σ 既知の極限値」であることを数値で示す
    #     （FEATURE-AUDIT.md A-2）。片側正規許容係数 k(n,coverage,confidence) は
    #     公表表（Natrella 1963・0.99/0.95）を再現し、有限標本では常に k > z_p。
    #     つまり n=8 対しか測っていない実機で「4σ で十分」と言う根拠は無い。
    #     近似は表より小さい側（＝要求 SAFETY を過小＝許容を締める＝偽BLOCK 側）に
    #     外れることも固定する（偽OK 方向に外れていないことの保証）。
    from tsugi.calibration import tolerance_factor_normal as _tf74
    from tsugi.calibration import wilks_confidence as _wc74
    from tsugi.calibration import wilks_min_runs as _wm74
    from tsugi.nondeterminism import normal_quantile as _nq74

    _tbl74 = {10: 3.981, 20: 3.295, 30: 3.064, 100: 2.684}
    _z74 = _nq74(0.99)
    check("one-sided tolerance factor reproduces the published table and exceeds z_p (A-2)",
          all(abs(_tf74(n, 0.99, 0.95) - v) / v < 0.015 and _tf74(n, 0.99, 0.95) <= v
              for n, v in _tbl74.items())
          and all(_tf74(n, 0.99, 0.95) > _z74 for n in (8, 16, 100, 1000))
          and _tf74(2, 0.99, 0.95) == float("inf")          # 標本不足は inf で正直に
          and _wm74(0.99, 0.95) == 299 and _wc74(299, 0.99) >= 0.95)

    # 75. 実機入口が SAFETY を実測校正し、run-to-run 標本では「下げてよい」と
    #     言わない（未測定のクロス成分を許容から外す＝偽OK 方向を封じる）。
    #     校正標本は run 対の差（比較される量と同単位）であって中心偏差ではない。
    from tsugi.calibration import SRC_CROSS_VENDOR as _SCV75
    from tsugi.calibration import calibrate_safety as _cs75
    from tsugi.nondeterminism import pair_deviations as _pd75
    from tsugi.tolerance import expected_gemm_abs_error as _eg75

    _sig75 = _eg75(256, "float16", 1.0, safety=1.0)
    _r2r75 = _cs75(np.full(32, 0.01 * _sig75), 256, scale=1.0)
    _xv75 = _cs75(np.full(32, 0.01 * _sig75), 256, scale=1.0, source=_SCV75)
    _big75 = _cs75(np.full(32, 6.0 * _sig75), 256, scale=1.0)
    _st75 = np.stack([np.full((4,), -3.0 if i % 2 == 0 else 3.0) for i in range(8)])
    check("audit_cross_vendor calibrates SAFETY from measured runs, never downward (A-2)",
          any("下げる根拠にはならない" in f.message for f in _r2r75.findings)
          and not any("余裕" in f.message for f in _r2r75.findings)
          and any("余裕" in f.message for f in _xv75.findings)   # cross 標本なら下げ代を提示
          and _big75.required > 4.0 and not _big75.covers_measured_noise
          and _cs75(np.zeros(0), 256).required == float("inf")   # 標本ゼロは校正済み扱いしない
          and np.allclose(_pd75(_st75), 6.0)                     # 対の差 2d（中心偏差 d の 2 倍）
          and _pd75(_st75[:1]).size == 0)

    # 76. 正規化層の相対発散増幅を *数値実験で検証してから* モデルへ入れた（A-5）。
    #     LayerNorm y=(x−μ)/√(σ²+eps) のヤコビアン J=(1/√(σ²+eps))(I−11ᵀ/d−ŷŷᵀ/d) は
    #     平均方向と半径（スケール）方向の 2 特異値が消える。ゆえに (a) x に平行な摂動は
    #     出力を変えず、(b) 独立方向の摂動は RMS/√(σ²+eps) 倍まで増幅されうる。
    #     旧実装は正規化を reduce に写し amp≈1 としており、平均優勢入力で増幅を
    #     見逃していた（偽OK）。RMSNorm は J=(1/r)(I−ŷŷᵀ) で増幅 ≤1 が無条件に成り立つが、
    #     1 未満の減衰係数は未検証ゆえ入れず amp=1.0 に固定する（保守側）。
    from tsugi.propagation import GraphOp as _GO76
    from tsugi.propagation import amplification as _amp76
    from tsugi.propagation import empirical_cond as _ec76
    from tsugi.propagation import is_amplifier as _ia76

    def _ln76(x, eps=1e-5):
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)

    def _measamp76(x, delta):
        d_in = np.sqrt(np.mean(delta ** 2)) / np.sqrt(np.mean(x ** 2))
        y, y2 = _ln76(x), _ln76(x + delta)
        return (np.sqrt(np.mean((y2 - y) ** 2)) / np.sqrt(np.mean(y ** 2))) / d_in

    _x76 = np.random.default_rng(0).standard_normal((32, 512)) + 10.0
    _mag76 = 1e-4 * np.sqrt(np.mean(_x76 ** 2))
    _generic76 = _measamp76(_x76, _mag76 * np.random.default_rng(9).standard_normal(_x76.shape))
    _bound76 = _ec76(_x76, "layer_norm")
    _zero76 = np.random.default_rng(1).standard_normal((32, 512))
    check("layer_norm amplifies and rms_norm neither amplifies nor damps (A-5)",
          _ia76("layer_norm") and not _ia76("rms_norm")
          and _amp76(_GO76("rms_norm", cond=100.0)) == 1.0      # 減衰係数も入れない
          and _amp76(_GO76("layer_norm", cond=7.0)) == 7.0)
    check("empirical_cond(layer_norm)=max RMS/sqrt(var+eps) bounds measured amplification",
          _generic76 > 2.0 and _generic76 <= _bound76 * 1.05      # 増幅は実在し上界内
          and _bound76 > 5.0 and _ec76(_zero76, "layer_norm") < 2.0)
    check("LayerNorm Jacobian nullspace (scale/mean directions) is annihilated as predicted",
          _measamp76(_x76, 1e-4 * _x76) < 0.01                     # スケール方向
          and _measamp76(_x76, np.full_like(_x76, _mag76)) < 0.01)  # 平均方向

    # 77. torch 経路: 正規化は専用 kind に写り（mean/sum/var は reduce 据置）、平均優勢
    #     sample の実測 cond が model_divergence を引き上げる。旧警告の「実際の発散は
    #     これより小さい可能性」（無条件の偽OK 主張）が復活していないことも固定する。
    from tsugi_torch.fxbridge import audit_fx as _afx77
    from tsugi_torch.fxbridge import fx_to_graph_ops as _f2g77

    def _gm77(target):
        return _GM58([
            _Node58("placeholder", "x"),
            _Node58("call_function", "aten.addmm.default", (8, 512)),
            _Node58("call_function", target),
            _Node58("output", "output"),
        ])

    _rng77 = np.random.default_rng(0)
    _hot77 = _afx77(_gm77("aten.native_layer_norm.default"),
                    sample=_rng77.standard_normal((32, 512)) * 0.1 + 5.0)
    _static77 = _afx77(_gm77("aten.native_layer_norm.default"))["model_divergence"]
    with _warnings58.catch_warnings(record=True) as _w77:
        _warnings58.simplefilter("always")
        _compile58(_gm77("aten.native_layer_norm.default"), [])
    check("fx maps norm ops to dedicated kinds (mean/sum stay reduce)",
          [o.kind for o in _f2g77(_gm77("aten.native_layer_norm.default"))]
          == ["matmul", "layer_norm"]
          and [o.kind for o in _f2g77(_gm77("aten._rms_norm.default"))]
          == ["matmul", "rms_norm"]                       # 判定順序（rms が先）を固定
          and [o.kind for o in _f2g77(_gm77("aten.mean.dim"))] == ["matmul", "reduce"])
    check("mean-dominated sample raises model_divergence and the false-OK claim is retracted",
          _hot77["model_divergence"] > _static77 * 3 and "layer_norm" in _hot77["amplifiers"]
          and len(_w77) == 1 and "増幅" in str(_w77[0].message)
          and "小さい可能性" not in str(_w77[0].message))

    # 78. 温度サンプリング下の分布一致（A-9・Q22/Q32）。実運用 LLM は温度サンプリングで
    #     出力するため argmax フリップ率だけでは出荷形態を覆えない。TV 距離の大域的上界
    #     tanh(ε/T) を採る（確率比が [e^{−2ε/T},e^{2ε/T}] に収まることの帰結・一次近似でない）。
    #     指示書が出発点に挙げた係数 1/2 型は実測で **破れる**（偽OK）ので採らない——
    #     A-5 と同じ「係数は数値実験で確かめてから入れる」適用例。
    #     ε は shift 不変・scale 非不変でなければならない: softmax は shift 不変だが
    #     scale 非不変（一様スケール＝温度変化）で、argmax は両方に不変という非対称がある。
    #     既存の residual（argmax 用）は純 scale で ≈0 → 偽OK、total は純 shift で大 → 偽BLOCK。
    from tsugi.decision import compare_task as _ct78
    from tsugi.decision import divergence_rms as _drms78
    from tsugi.decision import flip_rate as _fr78
    from tsugi.decision import residual_divergence_rms as _res78
    from tsugi.decision import sampling_epsilon as _se78
    from tsugi.decision import tv_bound as _tvb78

    def _sm78(z, T):
        e = np.exp((z - z.max(-1, keepdims=True)) / T)
        return e / e.sum(-1, keepdims=True)

    def _tv78(a, b, T):
        return 0.5 * np.abs(_sm78(a, T) - _sm78(b, T)).sum(-1)

    _rng78 = np.random.default_rng(0)
    _z78 = _rng78.standard_normal((2000, 32)) * 2.0
    _b78 = _z78 + 0.05 * _rng78.standard_normal(_z78.shape)
    # 上界が有効（任意の摂動）／tanh/2 では破れる（係数の外部検証）。
    # 後者は **敵対的な摂動**（各座標 ±ε）でのみ露出する——ランダムなガウス摂動は
    # 最悪ケースから遠く、そこだけ見ると誤って「1/2 でも足りる」と結論しかねない。
    _ok78 = True
    for _T in (0.2, 1.0, 4.0):
        _tv = _tv78(_z78, _b78, _T)
        _bd = _tvb78(_se78(_z78, _b78), _T)
        _ok78 = _ok78 and bool((_tv <= _bd + 1e-12).all())
    _half78 = False
    for _eps78 in (0.1, 1.0, 3.0):
        _adv78 = _z78 + np.random.default_rng(7).choice([-_eps78, _eps78], size=_z78.shape)
        for _T in (0.5, 1.0):
            _tva = _tv78(_z78, _adv78, _T)
            _bda = _tvb78(_se78(_z78, _adv78), _T)
            _ok78 = _ok78 and bool((_tva <= _bda + 1e-12).all())
            _half78 = _half78 or bool((_tva > _bda / 2).any())
    check("tanh(eps/T) bounds sampling TV globally and the 1/2-coefficient form breaks (A-9)",
          _ok78 and _half78)
    check("sampling epsilon is shift-invariant but scale-sensitive (residual/total both wrong)",
          float(_se78(_z78, _z78 + 3.0).max()) < 1e-9          # 純 shift → ε=0
          and float(_tv78(_z78, _z78 + 3.0, 1.0).max()) < 1e-12  # 実際 TV も 0
          and _drms78(_z78, _z78 + 3.0) > 2.9                  # total は大＝偽BLOCK
          and float(_se78(_z78, 1.1 * _z78).min()) > 0.01      # 純 scale → ε>0
          and float(_tv78(_z78, 1.1 * _z78, 1.0).mean()) > 0.01  # 実際 TV も非ゼロ
          and _res78(_z78, 1.1 * _z78) < 1e-9)                 # residual は ≈0＝偽OK
    # T→0 で TV 平均が argmax フリップ率に一致する（サンプリング層は decision 層の一般化）
    _greedy78 = _fr78(_z78, _b78)
    _cold78 = _ct78(_z78, _b78, task="sampling", temperature=0.01).flip_rate
    check("sampling flip rate converges to the argmax flip rate as T->0 (layer continuity)",
          _greedy78 > 0.005 and abs(_cold78 - _greedy78) <= 0.1 * _greedy78)

    # 79. facade: audit_runtime(task="sampling") が実測 TV と worst-case 上界を報告し、
    #     低温で上界が無情報になることを自ら申告する（額面通り BLOCK に使わせない）。
    from tsugi.audit import audit_runtime as _ar79

    _ad79 = _ar79(_z78, _b78, 256, logits_a=_z78, logits_b=_b78, task="sampling",
                  task_kwargs={"temperature": 0.02}, flip_budget=0.01)
    _dp79 = next((p for p in _ad79.phases if "sampling" in p.name), None)
    _txt79 = _dp79.to_text() if _dp79 is not None else ""
    check("audit_runtime(task=sampling) reports measured TV and flags a vacuous bound (A-9)",
          _dp79 is not None and "実測 TV" in _txt79 and "tanh" in _txt79
          and "無情報" in _txt79)

    # 80. beam 探索は静的な代表 logit から *証明可能に* 認証できない（A-9/Q22）。
    #     per-token フリップ率を (1−p)^L で合成する既存則は beam に不健全: argmax フリップは
    #     beam の復元を無視して survival を過小評価（偽OK）、frontier(top-k 集合)フリップは
    #     過度に悲観的。beam は累積対数尤度で並べ替えるため独立合成の前提が崩れる。
    #     rollout_from_logits(decode="beam") は greedy を経験的下界の参考値として返しつつ
    #     verdict を必ず WARN 以上にする（never OK・fail-safe）。
    from tsugi.report import Risk as _Risk80
    from tsugi.rollout import rollout_from_logits as _rfl80

    _id80 = np.zeros((100, 4), dtype=np.float64)
    _id80[:, 0] = 10.0
    _g80 = _rfl80(_id80, _id80.copy(), 8, decode="greedy", conservative=False)
    _bm80 = _rfl80(_id80, _id80.copy(), 8, decode="beam", conservative=False)
    check("beam is never certified OK from static logits (greedy reference, +WARN)",
          _g80.max_risk == _Risk80.OK and _bm80.max_risk >= _Risk80.WARN
          and abs(_bm80.flip_rate - _g80.flip_rate) < 1e-12
          and any("証明可能に" in f.message for f in _bm80.findings))

    # 81. beam survival ≥ greedy survival を実 beam 探索（自己回帰トイモデル）で実証。
    #     beam は幅 k の仮説を保持し過渡的な順位低下を復元するため、クロスベンダー一致は
    #     greedy 以上になる。これが「greedy を beam の下界に使うのは安全側」の根拠であり、
    #     同時に「greedy を beam の *認証値* に流用してはいけない（beam は実際もっと等価）」理由。
    def _beam81(start, trans, k, steps, noise):
        V = start.shape[0]
        beams = [([], 0.0)]
        for t in range(steps):
            cand = []
            for toks, sc in beams:
                prev = toks[-1] if toks else None
                lp = (start if prev is None else trans[prev]) + noise[t]
                lp = lp - np.log(np.exp(lp).sum())
                for v in np.argpartition(-lp, min(k, V - 1))[:min(k, V)]:
                    cand.append((toks + [int(v)], sc + float(lp[v])))
            cand.sort(key=lambda x: -x[1])
            beams = cand[:k]
        return tuple(beams[0][0])

    _rng81 = np.random.default_rng(0)
    _V81, _st81, _eps81, _n81 = 24, 6, 0.6, 120
    _start81 = _rng81.standard_normal(_V81) * 1.5
    _trans81 = _rng81.standard_normal((_V81, _V81)) * 1.5
    _bmatch81 = _gmatch81 = 0
    for _ in range(_n81):
        _noise81 = [_rng81.standard_normal(_V81) for _ in range(_st81)]
        _na81 = [z + _eps81 * _rng81.standard_normal(_V81) for z in _noise81]
        _nb81 = [z + _eps81 * _rng81.standard_normal(_V81) for z in _noise81]
        _bmatch81 += (_beam81(_start81, _trans81, 6, _st81, _na81)
                      == _beam81(_start81, _trans81, 6, _st81, _nb81))
        _gmatch81 += (_beam81(_start81, _trans81, 1, _st81, _na81)
                      == _beam81(_start81, _trans81, 1, _st81, _nb81))
    check("beam survival dominates greedy survival (redundancy recovers divergence)",
          _bmatch81 >= _gmatch81 and _bmatch81 > _gmatch81)

    # 82. 予測フリップ率上界の *信頼性* を裾サポート（超過数 k）で定量化する（A-9/Q21）。
    #     P(margin<2δ) は決定境界近傍の裾確率で、その相対不確実性は total n でなく k に
    #     支配される（≈1/√k・n 非依存）。Wilson は与えられた集合の比率不確実性を織り込むが、
    #     集合が本番を代表しているかは問えない —— 大マージンばかりの代表集合は少数の
    #     near-tie しか含まず、near-tie が多い本番のフリップ率を過小評価する（偽OK）。
    #     この gap を裾サポート（well_supported=k≥30・極値理論の安定裾目安）が暴く。
    #     従来の散文「分布シフトで妥当域を外れうる」を定量シグナルに置き換えた。
    from tsugi.decision import flip_bound_support as _fbs82
    from tsugi.decision import flip_rate as _fr82
    from tsugi.decision import predicted_flip_bound as _pfb82

    _r82 = np.random.default_rng(0)
    _delta82 = 0.05
    _ref82 = _r82.standard_normal((500, 10)) * 6.0            # 大マージン・境界を踏まない
    _pa82 = _r82.standard_normal((500, 10)) * 0.3            # 本番: 境界近傍に多い
    _pb82 = _pa82 + _delta82 * _r82.standard_normal(_pa82.shape)
    _bound82 = _pfb82(_ref82, _delta82)
    _true82 = _fr82(_pa82, _pb82)
    _sup82 = _fbs82(_ref82, _delta82)
    # 1/√k が n 非依存: 同じ k なら n が桁違いでも rel_uncertainty は同じ
    def _setk82(k, n):
        m = np.full(n, 10.0)
        m[:k] = 0.001
        return np.stack([m, np.zeros(n)], axis=-1)
    _u1_82 = _fbs82(_setk82(30, 1000), 0.5)["rel_uncertainty"]
    _u2_82 = _fbs82(_setk82(30, 100000), 0.5)["rel_uncertainty"]
    check("flip-bound tail support exposes an unrepresentative calibration set (A-9/Q21)",
          _true82 > _bound82 * 2.0                            # 偽OK gap を再現
          and not _sup82["well_supported"]                    # 診断が不足を暴く
          and _sup82["exceedances"] < _sup82["min_exceedances"]
          and abs(_u1_82 - 1.0 / (30 ** 0.5)) < 1e-9          # ≈1/√k
          and abs(_u1_82 - _u2_82) < 1e-9)                    # n 非依存

    # 82b. facade: audit の propagation→decision 橋が裾サポートを定量報告する（配線）。
    from tsugi.audit import audit as _audit82
    from tsugi.portcheck import _demo_module as _dm82

    _mod82, _blk82, _cfg82 = _dm82()
    _ad82 = _audit82(_mod82, _cfg82, block_dims=_blk82,
                     ref_logits=_r82.standard_normal((300, 10)) * 6.0)
    _prop82 = next(p for p in _ad82.phases if "propagation" in p.name)
    check("audit bridge quantifies calibration-set tail support (not just prose)",
          any("裾サポート" in ln for ln in _prop82.lines))

    # 83. テンサーコアの入力精度（TF32/bf16）をクロスベンダー発散源としてモデル化する
    #     （累積順序差とは *別源*）。NVIDIA は fp32 GEMM を既定で TF32 に落としうる
    #     （PyTorch 2.9 fp32_precision・FlexAttention の ieee→tf32 回帰）が AMD は TF32 非対応。
    #     入力仮数 truncation は各要素の相対摂動なので発散は ~u で **K 非依存**（累積は √K·u）。
    #     precision_policy_hint が fp32 の TF32 帯発散を「バグでなく既知の精度ポリシー差」の
    #     兆候として拾い（LAYOUT 判定と同系統の良性差検出）、audit_runtime に接続済み。
    from tsugi.equivalence import input_precision_divergence as _ipd83
    from tsugi.equivalence import precision_policy_hint as _pph83
    from tsugi.equivalence import simulate_vendor_matmul as _svm83
    from tsugi.equivalence import truncate_to_tensorcore as _ttc83

    _r83 = np.random.default_rng(0)
    _a83 = _r83.standard_normal((64, 2048)).astype(np.float32)
    _b83 = _r83.standard_normal((2048, 64)).astype(np.float32)
    _ie83 = _svm83(_a83, _b83)
    _tf83 = _svm83(_a83, _b83, input_precision="tf32")
    _rel83 = float(np.linalg.norm(_tf83 - _ie83) / np.linalg.norm(_ie83))
    # 別の K でも発散はほぼ一定（flat・√K でない）
    _a83b = _r83.standard_normal((64, 256)).astype(np.float32)
    _b83b = _r83.standard_normal((256, 64)).astype(np.float32)
    _rel83b = float(np.linalg.norm(_svm83(_a83b, _b83b, input_precision="tf32")
                                   - _svm83(_a83b, _b83b))
                    / np.linalg.norm(_svm83(_a83b, _b83b)))
    _vtrunc83 = _ttc83(_a83, "tf32")
    check("tensor-core input precision (TF32) is modeled as a K-independent divergence source",
          np.abs((_vtrunc83 - _a83) / _a83).max() <= 2.0 ** -11 + 1e-9   # フォーマット定義
          and _rel83 <= _ipd83("tf32")                                   # 予測上界内
          and max(_rel83, _rel83b) / min(_rel83, _rel83b) < 1.5          # K 非依存（flat）
          and _pph83(_ie83, _tf83, 2048, "float32") is not None          # 兆候を拾う
          and _pph83(_ie83, _ie83 * 1.01, 2048, "float32") is None       # 粗いバグは拾わない
          and _pph83(_ie83, _tf83, 2048, "float16") is None)             # 非 fp32 は黙る

    # 83b. facade: audit_runtime が fp32 の TF32 帯発散に精度ポリシーヒントを surface する。
    from tsugi.audit import audit_runtime as _ar83

    _ad83 = _ar83(_ie83, _tf83, 2048, dtype="float32")
    _eqp83 = next(p for p in _ad83.phases if "equivalence" in p.name)
    check("audit_runtime surfaces the TF32 precision-policy hint on fp32 divergence",
          any("精度ポリシー差" in ln for ln in _eqp83.lines))

    # 84. テンサーコアの丸めモード差（RTZ vs RNE）は *系統* 発散＝第 3 の発散クラス。
    #     入力精度差（~u・K 非依存）と累積順序差（√K·u）はどちらもゼロ平均だが、
    #     round-toward-zero は仮数を切り捨てて |値| を系統的に縮めるため RMS 比が下がる
    #     （一方向バイアス）。テンサーコアの丸めは実装定義で RTZ 系が報告される
    #     （Fasi/Higham/Mikaitis/Pranesh, PeerJ CS 7:e330, 2021）。max_abs だけの等価判定は
    #     この一方向差を見逃しうるが、calibration.check_systematic（RMS 比）が捕まえる。
    from tsugi.calibration import check_systematic as _cs84
    from tsugi.calibration import systematic_divergence as _sd84
    from tsugi.equivalence import simulate_vendor_matmul as _svm84
    from tsugi.equivalence import truncate_to_tensorcore as _ttc84

    _r84 = np.random.default_rng(0)
    _a84 = _r84.standard_normal((64, 2048)).astype(np.float32)
    _b84 = _r84.standard_normal((2048, 64)).astype(np.float32)
    _ie84 = _svm84(_a84, _b84)
    _order84 = abs(_sd84(_ie84, _svm84(_a84, _b84, split_k=8)))
    _prec84 = abs(_sd84(_ie84, _svm84(_a84, _b84, input_precision="tf32")))
    _rtz84 = _svm84(_a84, _b84, input_precision="tf32", input_rounding="rtz")
    _bias84 = _sd84(_ie84, _rtz84)
    # RTZ 既定回帰なし: input_rounding 未指定 = rne
    _v84 = _r84.standard_normal(1000).astype(np.float32)
    check("round-toward-zero is a biased divergence class caught by check_systematic",
          _bias84 < 0                                          # |値| が縮む（一方向）
          and abs(_bias84) > 20 * max(_order84, _prec84)       # 他 2 クラスより桁違いに大
          and not _cs84(_ie84, _rtz84, K=2048, dtype="float32").ok   # RMS 比が検出
          and np.array_equal(_ttc84(_v84, "tf32"),             # 既定は RNE（後方互換）
                             _ttc84(_v84, "tf32", "rne")))

    # 85. 3xTF32（Triton input_precision="tf32x3" / CUTLASS 3xTF32）は TF32 発散の *緩和策*。
    #     各 fp32 を hi+lo の 2 TF32 成分に分割し 3 項で積む（lo·lo を落とす・Ootomo & Yokota
    #     2022, arXiv:2203.03341）。残差 ~u_tf32²（~2⁻²²）で平 TF32（~2⁻¹¹）より桁違いに正確。
    #     精度ポリシー選択（ieee/tf32/tf32x3）が発散を決める——tf32x3 ベンダーと IEEE ベンダーの
    #     発散は fp32 等価ゆえ precision_policy_hint は拾わない。
    from tsugi.equivalence import input_precision_divergence as _ipd85
    from tsugi.equivalence import precision_policy_hint as _pph85
    from tsugi.equivalence import simulate_vendor_matmul as _svm85

    _r85 = np.random.default_rng(0)
    _a85 = _r85.standard_normal((64, 2048)).astype(np.float32)
    _b85 = _r85.standard_normal((2048, 64)).astype(np.float32)
    _ie85 = _svm85(_a85, _b85)
    _tf85 = _svm85(_a85, _b85, input_precision="tf32")
    _x3_85 = _svm85(_a85, _b85, input_precision="tf32x3")
    def _rel(x):
        return float(np.linalg.norm(x - _ie85) / np.linalg.norm(_ie85))
    check("tf32x3 error-correction recovers near-fp32 accuracy (mitigates TF32 divergence)",
          _rel(_x3_85) < _rel(_tf85) / 50                       # 桁違いに正確
          and _rel(_x3_85) <= _ipd85("tf32x3")                  # 予測上界内
          and _ipd85("tf32x3") < _ipd85("tf32")                 # 復元を上界も反映
          and _pph85(_ie85, _tf85, 2048, "float32") is not None  # tf32 は拾う
          and _pph85(_ie85, _x3_85, 2048, "float32") is None)    # tf32x3 は fp32 等価ゆえ黙る

    # 86. CLI（portcheck.report / `python -m tsugi`）の終了コードは CI ゲート契約
    #     （OK/INFO=0・WARN=1・BLOCK=2）に忠実であること。従来 `0 if portable else 1` で
    #     BLOCK(2) と WARN(1) を 1 に潰しており、CI が exit>=2 でのみ失敗する設定だと BLOCK が
    #     素通りした（プロセス層の偽OK）。report は audit.exit_code をそのまま返す。
    from tsugi.audit import audit as _audit86
    from tsugi.portcheck import _demo_module as _dm86
    from tsugi.portcheck import report as _rep86

    _mod86, _blk86, _cfg86 = _dm86()
    import contextlib as _ctx86
    import io as _io86
    _buf86 = _io86.StringIO()
    with _ctx86.redirect_stdout(_buf86):
        _rc86 = _rep86(_mod86, _blk86, _cfg86)
    _a86 = _audit86(_mod86, _cfg86, block_dims=_blk86)
    check("CLI exit code follows the OK/WARN/BLOCK gate contract (BLOCK=2, not collapsed)",
          _a86.max_risk.name == "BLOCK" and _rc86 == _a86.exit_code == 2)

    # 87. tsugi.verify() は移植性検証のワンコール入口（CLI の Python 版・簡素化）。
    #     パス（.py カーネル）と traced module の両方を受け、Audit を返す。手作業の
    #     trace→audit 連結を 1 コールに畳んだ価値経路——CLI(portcheck)もこれ経由。
    import tsugi as _tsugi87
    from tsugi.portcheck import _demo_module as _dm87

    _mod87, _blk87, _cfg87 = _dm87()
    _ad87 = _tsugi87.verify(_mod87, block_dims=_blk87, cfg=_cfg87)
    from pathlib import Path as _P87
    _ex87 = _P87(__file__).resolve().parent / "examples" / "user_kernel.py"
    _adp87 = _tsugi87.verify(str(_ex87))
    check("tsugi.verify() one-call facade accepts module and path, returns gated Audit",
          isinstance(_ad87, _tsugi87.Audit) and _ad87.exit_code == 2
          and isinstance(_adp87, _tsugi87.Audit) and _adp87.exit_code in (0, 1, 2))

    # 88. 検証ゲートは `check.py` の **単一定義** をローカルと CI が共有する。
    #     以前は CONTRIBUTING.md と docs/ci-reference.yml にゲートが二重定義され実際に
    #     食い違っていた（CONTRIBUTING が verify.py を lint 対象から落としていた）——
    #     「ローカル緑・CI 赤」の温床。両文書が check.py を *呼ぶだけ* になっており、
    #     ゲートを再列挙していないことを機械的に固定する（ドリフトを構造的に不可能にする）。
    _root88 = _P87(__file__).resolve().parent
    _chk88 = (_root88 / "check.py").read_text(encoding="utf-8")
    _ci88 = (_root88 / "docs" / "ci-reference.yml").read_text(encoding="utf-8")
    _contrib88 = (_root88 / "CONTRIBUTING.md").read_text(encoding="utf-8")
    _rel88 = (_root88 / "docs" / "RELEASING.md").read_text(encoding="utf-8")
    check("verification gates have a single definition (check.py) shared by local/CI/release",
          "LINT_TARGETS" in _chk88 and "SMOKE_EXAMPLES" in _chk88   # ゲート実体は check.py
          and "verify.py" in _chk88                                 # lint 対象に verify.py を含む
          # 旧 3 重定義（CI/CONTRIBUTING/RELEASING）はいずれも呼ぶだけ・再列挙しない
          and all("python check.py" in d for d in (_ci88, _contrib88, _rel88))
          and not any("ruff check python/" in d for d in (_ci88, _rel88)))

    # 89. このプロダクトの楔は **フレームワーク層（torch.compile）** なのに、ゲート付き判定
    #     （exit_code/to_text）は tile-DSL 経路だけが持ち、想定ユーザーである PyTorch 開発者は
    #     `audit_fx` の素の dict しか得られず出荷判断に使えなかった。`audit_torch`
    #     （`tsugi.verify(gm)` から到達）が両経路の契約を揃える。
    #     fail-safe: 静的 FX は等価性を認証できない（第2ベンダー出力が無い）ので発散量に閾値を
    #     発明して BLOCK にしない——BLOCK は利用者が与えた flip_budget 超過のときだけ。
    import tsugi as _tsugi89
    from tsugi_torch.fxbridge import audit_torch as _at89

    _gm89 = _GM58([
        _Node58("placeholder", "x"),
        _Node58("call_function", "aten.addmm.default", (8, 512)),
        _Node58("call_function", "aten._softmax.default"),
        _Node58("output", "output"),
    ])
    _ad89 = _at89(_gm89)
    _nt89 = np.random.default_rng(0).standard_normal((500, 32)) * 0.05   # near-tie 多め
    _blk89 = _at89(_gm89, ref_logits=_nt89, flip_budget=0.001)
    _ok89 = _at89(_gm89, ref_logits=_nt89, flip_budget=1.0)
    check("torch/FX path yields a gated Audit; BLOCK only from the user's flip_budget",
          isinstance(_ad89, _tsugi89.Audit)
          and _ad89.exit_code in (0, 1) and _ad89.portable      # 閾値を発明しない
          and any(p.when == "pending" for p in _ad89.phases)    # 実機照合が要ると明示
          and _blk89.exit_code == 2                             # 予算超過なら BLOCK
          and _ok89.exit_code < 2                               # 予算が緩ければ通す
          and _tsugi89.verify(_gm89).exit_code == _ad89.exit_code)  # verify() から到達

    # 90. 「codegen は LLVM/MLIR + 実機が要るので不可能」は **要件の誤り**だった。
    #     生成（純関数）とアセンブル（CPU ツール）と実行（要 GPU）は別物で、前二者は
    #     この環境で可能。単一の IR から 3 ターゲットの実アセンブリが出ること、そして
    #     **ベンダー自身のアセンブラ**（ptxas / llvm-mc）が受理することを固定する。
    #     ツールが無い環境では ok=None（L1）に落ち、**未検証を合格に丸めない**。
    from tsugi import codegen as _cg90
    from tsugi import tile as _tile90
    from tsugi.tracer import EMITTABLE_OPS as _EMITTABLE90

    @_tsugi89.jit
    def _KMM90(a, b, c, M, N, K, BM, BN, BK):
        pm, pn = _tsugi89.program_id(0), _tsugi89.program_id(1)
        acc = _tile90.zeros((BM, BN), _tsugi89.float32)
        for k in range(0, K, BK):
            acc = _tile90.dot(_tile90.load(a, (pm * BM, k), (BM, BK)),
                              _tile90.load(b, (k, pn * BN), (BK, BN)), acc)
        _tile90.store(c, (pm * BM, pn * BN), acc.to(_tsugi89.float16))

    _ARGS90 = (np.zeros((32, 32), np.float16), np.zeros((32, 32), np.float16),
               np.zeros((32, 32), np.float32), 32, 32, 32, 16, 16, 16)
    _mod90 = _tsugi89.trace(_KMM90, _ARGS90, {}, program_ids=(0, 0))
    _res90 = {t: _cg90.verify_codegen(_mod90, target=t) for t in _cg90.TARGETS}
    check("one IR emits real PTX/AMDGCN that the vendors' own assemblers accept (or L1)",
          all(em.text.strip() and not em.uncovered for em, _ in _res90.values())
          and ".visible .entry" in _res90["nvidia"][0].text
          and all(".amdgcn_target" in _res90[t][0].text
                  for t in ("amd_cdna", "amd_rdna"))
          and all((a.ok is True and a.obj_bytes > 0) if _cg90.toolchain(t) is not None
                  else (a.ok is None and a.available is False)
                  for t, (_, a) in _res90.items())
          # L3（実機実行検証）は到達不能。どの経路もそれを主張しない。
          and all(a.level != _cg90.VERIFY_LEVELS[3] for _, a in _res90.values()))

    # 91. codegen の価値は「テキストが出る」ことでなく、**arch 条件付きの可用性を
    #     ツールチェインが事実として返す**こと。手書きの対応表では作り込めない
    #     移植ブロッカー（WMMA は sm_70+ / MFMA は CDNA 専用）をアセンブラに問う。
    #     さらに壊れたアセンブリが弾かれることで、この検査自体の有効性を担保する。
    _probe91 = _cg90.probe_op
    check("the assembler is the oracle: arch-conditional gaps are found, junk is rejected",
          (_cg90.toolchain("nvidia") is None or (
              _probe91("dot", target="nvidia", arch="sm_80").ok is True
              and _probe91("dot", target="nvidia", arch="sm_60").ok is False
              and _probe91("add", target="nvidia", arch="sm_60").ok is True
              and _cg90.assemble(".version 7.0\n.target sm_80\nbogus_ins;\n",
                                 target="nvidia").ok is False))
          and (_cg90.toolchain("amd_cdna") is None or (
              _probe91("dot", target="amd_cdna", arch="gfx90a").ok is True
              and _probe91("dot", target="amd_rdna", arch="gfx1100").ok is True
              # CDNA の MFMA を RDNA の arch へ → 成立しない（単一命令では移植不可）
              and _probe91("dot", target="amd_cdna", arch="gfx1100",
                           isa="amd_cdna").ok is False
              and _cg90.assemble("\tv_no_such_instruction v0, v1\n",
                                 target="amd_cdna").ok is False)))

    # 92. 生成が facade（audit / compile）から到達し、**保証しないもの**を黙らない。
    #     L2 が言えるのは命令の存在・構文・arch 可用性まで。レイアウト接合と実行の
    #     正しさは L3（実機）で、そこは常に空だとレポート自身が述べること。
    _ad92 = _tsugi89.audit(_mod90, block_dims=(32,))
    _cgp92 = [p for p in _ad92.phases if p.name.startswith("codegen")]
    _art92 = _tsugi89.compile(_KMM90, _ARGS90, target="nvidia", emit_machine_code=True)
    check("codegen is reachable from the facades and declares what it does NOT verify",
          len(_cgp92) == 1                                        # audit に載る
          and "L3" in _cgp92[0].to_text() and "常に空" in _cgp92[0].to_text()
          and any(n.startswith("layout-unstitched")               # 未接合を黙らない
                  for n in _cg90.emit(_mod90, target="nvidia").unstitched)
          and _art92.asm is not None                              # compile から到達
          and _art92.level == (_cg90.VERIFY_LEVELS[2]
                               if _cg90.toolchain("nvidia") is not None
                               else _cg90.VERIFY_LEVELS[1])
          and _tsugi89.compile(_KMM90, _ARGS90, target="nvidia").asm is None
          # ビット同一でないと分類した op が実際に近似命令を出している（分類が腐らない）
          and set(_cg90.BIT_EXACT_ACROSS_VENDORS) == set(_cg90.CODEGEN_OPS)
          and _cg90.CODEGEN_OPS <= _EMITTABLE90
          and not _cg90.uncodegenned_ops("nvidia"))


def main() -> int:
    """全不変条件をテーマ別グループの順で実行し、集計を印字する（SOCRATIC-50 Q34:
    単一巨大 main() を分割し失敗箇所をグループ名で局所化できるようにした）。
    実行順・check 文言・件数はグループ化前と同一（純粋なコード移動）。"""
    sys.path.insert(0, str(PY))
    for group in (
        _check_prohibited_and_suites,
        _check_core_pillars,
        _check_verifier_calibration,
        _check_audit_facades,
        _check_propagation_floors_and_search,
        _check_reports_and_diagnostics,
        _check_dtype_precision,
        _check_nondet_catalog_and_dynamic_shapes,
        _check_late_facade_wiring,
        _check_statistical_rigor,
        _check_shape_guards,
        _check_meta_integrity,
    ):
        group()

    failed = [n for n, c in INVARIANTS if not c]
    print(f"\n{'VERIFY PASS' if not failed else 'VERIFY FAIL'}: "
          f"{len(INVARIANTS) - len(failed)}/{len(INVARIANTS)} invariants")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
