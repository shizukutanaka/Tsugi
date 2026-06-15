"""Tsugi audit — 検証層を 1 つの判定に束ねる統合ファサード。

8 つの視点（portability/equivalence/occupancy/tolerance/feasibility/propagation/
envelope/decision）＋メタ層（calibration）＋基盤（nondeterminism）が出揃った。
個別に呼ぶのでなく、traced IR ＋タイル構成から **静的に実行できる層をまとめて回し、
1 つの Audit レポートにする**。さらに *実機データが要る層*（実行時エンベロープ・
非決定性ノイズ・タスクフリップ）を「実行時チェックリスト」として明示し、検証の
ライフサイクル（静的 → 動的 → メタ → 基盤 → 翻訳）を一望できるようにする。

設計: 各視点は自分の所見を返すだけ。ここは束ねて深刻度を集約する単一責任。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import ir
from .report import Risk

TARGETS = ("nvidia", "amd_cdna", "amd_rdna")


@dataclass
class AuditPhase:
    name: str          # 層名
    when: str          # "static"（今ここで判定）/ "runtime"（実機データが要る）
    max_risk: Risk
    lines: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        tag = "" if self.when == "static" else " [要実機データ]"
        head = f"[{self.max_risk.name:5s}] {self.name}{tag}"
        body = "\n".join("    " + ln for ln in self.lines)
        return head + ("\n" + body if body else "")


@dataclass
class Audit:
    phases: list[AuditPhase] = field(default_factory=list)

    @property
    def static_phases(self) -> list[AuditPhase]:
        return [p for p in self.phases if p.when == "static"]

    @property
    def max_risk(self) -> Risk:
        """判定は静的層のみから（実行時層は実機まで未確定）。"""
        return max((p.max_risk for p in self.static_phases), default=Risk.OK)

    @property
    def portable(self) -> bool:
        return self.max_risk < Risk.BLOCK

    def to_text(self) -> str:
        lines = ["=== Tsugi audit（検証層の統合判定）==="]
        for p in self.phases:
            lines.append(p.to_text())
        verdict = "移植可（要注意点あり）" if self.portable else "移植ブロッカーあり"
        lines.append(f"\n判定（静的層）: {verdict} [max_risk={self.max_risk.name}]")
        return "\n".join(lines)


def _gemm_depth(module: ir.Module, cfg) -> int:
    """累積深さ K ≈ dot 反復数 × BK（マジックナンバーでなく構成由来）。"""
    n_dots = sum(1 for k in module.kernels for op in k.body if op.kind == "dot")
    bk = cfg.block_k if cfg is not None else 32
    return n_dots * bk


def audit(module: ir.Module, cfg=None, *, targets=TARGETS,
          block_dims=None) -> Audit:
    """traced IR ＋構成から静的検証層をまとめて回し、1 つの判定に束ねる。"""
    from .calibration import detectability_floor
    from .envelope import certify_gemm
    from .feasibility import cross_vendor_feasibility, first_vendor_only
    from .occupancy import occupancy_gap
    from .portability import analyze
    from .tolerance import explain

    a = Audit()

    # --- 静的: 移植性（per target） ---
    worst = Risk.OK
    port = AuditPhase("portability 移植リスク", "static", Risk.OK)
    for t in targets:
        rep = analyze(module, t, block_dims=block_dims, cfg=cfg)
        worst = max(worst, rep.max_risk)
        port.lines.append(f"{t}: max_risk={rep.max_risk.name}")
        for f in rep.findings:
            if f.risk >= Risk.WARN:
                port.lines.append(f"  [{f.risk.name}] {f.op} {f.message}")
    port.max_risk = worst
    a.phases.append(port)

    # --- 静的: 起動可能性（占有率より上流のゲート） ---
    if cfg is not None:
        feas = AuditPhase("feasibility 起動可能性", "static", Risk.OK)
        fr = cross_vendor_feasibility(cfg)
        for v, f in fr.items():
            feas.lines.append(f"{v}: {'LAUNCHABLE' if f.launchable else 'NOT-LAUNCHABLE'}")
        only = first_vendor_only(cfg, "nvidia", "amd_cdna")
        if only:
            feas.max_risk = Risk.BLOCK
            feas.lines.append("単一ソース約束の破綻（片方でしか起動しない）:")
            for o in only:
                feas.lines.append(f"  起動不能: {o}")
        a.phases.append(feas)

        # --- 静的: 占有率ギャップ（速度の片寄り） ---
        gap = occupancy_gap(cfg, "nvidia", "amd_cdna")
        occ = AuditPhase("occupancy 占有率", "static",
                         Risk.WARN if gap >= 0.25 else Risk.INFO)
        occ.lines.append(f"NVIDIA↔AMD CDNA 占有率差 = {gap:.0%}"
                         + ("（性能が片方だけ崩れる）" if gap >= 0.25 else ""))
        a.phases.append(occ)

    # --- 静的: 数値等価性の目安（導出許容 + 認証エンベロープ + 検出限界） ---
    K = _gemm_depth(module, cfg)
    if K > 0:
        num = AuditPhase("numerics 数値等価性の目安", "static", Risk.INFO)
        num.lines.append("導出許容: " + explain(K, "float16"))
        num.lines.append("認証エンベロープ: " + certify_gemm(K, "float16", 1.0).to_text())
        floor = detectability_floor(K, "float16")
        num.lines.append(
            f"検出限界(偽OKの盲点): max_abs は相対 {floor['rel'] * 100:.1f}% 未満の"
            "系統誤差を見逃す → calibration.check_systematic で相補検査")
        a.phases.append(num)

    # --- 実行時（実機データが要る層をチェックリストとして明示） ---
    rt = AuditPhase("runtime 実行時チェックリスト", "runtime", Risk.INFO)
    rt.lines += [
        "envelope.check_tensor(x, env): 本番入力が認証前提内か（overflow/denormal/scale）",
        "nondeterminism.compare_stable(runA, runB, K): run-to-run ノイズを実測し分布比較",
        "  クロス差 ≤ ノイズなら INDISTINGUISHABLE（等価判定は未定義）",
        "decision.compare_decisions(logitsA, logitsB): 判断フリップ率（タスク影響）",
        "  数値発散はマージン分布を介してフリップに翻訳・タスク予算で判定",
    ]
    a.phases.append(rt)
    return a
