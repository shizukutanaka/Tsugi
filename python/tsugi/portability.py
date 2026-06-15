"""Tsugi portability — クロスベンダー移植リスクの静的解析（ソクラテス問答で発見した新視点）。

新視点: Tsugi の最強の楔は codegen でなく「移植が正しいと証明する検証層」。
Triton は NVIDIA/AMD カーネルを生成するが、両者の数値等価性は保証しない。
本モジュールは traced IR から *実行前に* 移植リスクを告げる（GPU 不要・CPU で動く）。

リサーチ由来: 堀の本質はライブラリ＋QA。AMD の弱点は性能でなく QA 文化（SemiAnalysis）。
ゆえにクロスベンダー QA そのものが差別化になる。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from . import ir

# 各ベンダーの実行モデル定数（移植時に壊れる主因）
VENDOR_WARP = {"nvidia": 32, "amd_cdna": 64, "amd_rdna": 32}

# 各ベンダーが行列コアで直接サポートする MMA 形状（M,N,K）の代表例
# 範囲外は分解 or 性能劣化 or 非対応 → 移植リスク
SUPPORTED_MMA = {
    "nvidia": {(16, 16, 16), (8, 8, 4), (16, 8, 16)},      # WMMA/HMMA 系
    "amd_cdna": {(16, 16, 16), (32, 32, 8), (16, 16, 4)},  # MFMA 系
    "amd_rdna": {(16, 16, 16)},                            # RDNA3 WMMA
}

# bf16 を行列コアで扱えるか（リファレンスは f32 近似だが実機差がある）
BF16_MATRIX = {"nvidia": True, "amd_cdna": True, "amd_rdna": False}


class Risk(IntEnum):
    OK = 0
    INFO = 1
    WARN = 2
    BLOCK = 3


@dataclass
class Finding:
    risk: Risk
    op: str
    message: str


@dataclass
class PortabilityReport:
    target: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def max_risk(self) -> Risk:
        return max((f.risk for f in self.findings), default=Risk.OK)

    @property
    def portable(self) -> bool:
        return self.max_risk < Risk.BLOCK

    def to_text(self) -> str:
        lines = [f"portability report → {self.target} (max_risk={self.max_risk.name})"]
        for f in self.findings:
            lines.append(f"  [{f.risk.name:5s}] {f.op:6s} {f.message}")
        if not self.findings:
            lines.append("  (no risks detected)")
        return "\n".join(lines)


def analyze(module: ir.Module, target: str,
            block_dims: tuple[int, ...] | None = None,
            cfg=None) -> PortabilityReport:
    """traced IR を target 向けに静的解析し、移植リスクを列挙する。

    cfg: TileConfig を渡すと占有率に基づく警告も追加する（任意）。
    """
    if target not in VENDOR_WARP:
        raise ValueError(f"unknown target: {target}")
    rep = PortabilityReport(target=target)

    # warp/wavefront 想定の暗黙依存（block dims が warp の倍数か）
    if block_dims:
        warp = VENDOR_WARP[target]
        for d in block_dims:
            if d % warp != 0:
                rep.findings.append(Finding(
                    Risk.WARN, "block",
                    f"block dim {d} は {target} の warp/wavefront={warp} の倍数でない "
                    f"→ 占有率低下・他ベンダーと挙動差"))

    # 起動可能性（cfg があれば）— 占有率より上流の categorical ゲート。
    # per-block のハード上限を越える構成は「遅い」のでなく「起動しない」= BLOCK。
    if cfg is not None:
        from .feasibility import check as _feasible
        try:
            feas = _feasible(cfg, target)
            for b in feas.blockers:
                rep.findings.append(Finding(
                    Risk.BLOCK, "launch",
                    f"{b.resource} required={b.required} が {target} の per-block 上限 "
                    f"{b.limit} を超過 → カーネルが起動しない（占有率の問題でなく起動不能）"))
        except ValueError:
            feas = None

    # 占有率（起動可能な場合のみ意味を持つ）
    if cfg is not None and (feas is None or feas.launchable):
        from .occupancy import estimate
        try:
            occ = estimate(cfg, target)
            if occ.occupancy < 0.25:
                rep.findings.append(Finding(
                    Risk.WARN, "occ",
                    f"占有率 {occ.occupancy:.0%}（{occ.limited_by} 制約）"
                    "→ このベンダーで性能が崩れる可能性"))
        except ValueError:
            pass

    for kernel in module.kernels:
        n_dots = sum(1 for op in kernel.body if op.kind == "dot")
        for op in kernel.body:
            if op.kind == "dot":
                _check_dot(op, target, rep)
            if op.kind == "cast" and op.attrs.get("to") == "bfloat16":
                if not BF16_MATRIX.get(target, False):
                    rep.findings.append(Finding(
                        Risk.WARN, "cast",
                        f"bf16 が {target} の行列コアで非対応 → f32 経路へ降格・性能差"))
            if op.kind == "atomic":
                rep.findings.append(Finding(
                    Risk.WARN, "atomic",
                    "atomics はベンダー間で順序・性能差が大きい（特に float atomics）"))
            if op.attrs.get("fast_math"):
                rep.findings.append(Finding(
                    Risk.WARN, "fastmath",
                    "fast-math 有効 → 丸め・NaN 挙動がベンダー間で発散（数値非等価の主因）"))
        # 累積を伴う matmul（K ループで dot 反復）は累積順序差で数値発散しうる
        if n_dots >= 2:
            rep.findings.append(Finding(
                Risk.INFO, "dot",
                f"K 方向に {n_dots} 回累積 → fp16 累積順序がベンダー間で異なり数値発散の可能性"
                "（equivalence で要照合）"))
    return rep


def _check_dot(op: ir.Op, target: str, rep: PortabilityReport) -> None:
    # result type tensor<MxNxf32> から M,N を読み、K は推定不可なので形状不明は INFO
    rtype = op.result.type if op.result else ""
    shape = _parse_shape(rtype)
    if shape is None:
        rep.findings.append(Finding(Risk.INFO, "dot",
                                    "matmul 形状を IR から確定できず（実機で要確認）"))
        return
    m, n = shape
    supported = {(sm, sn) for (sm, sn, _sk) in SUPPORTED_MMA[target]}
    if (m, n) not in supported and m != n:
        rep.findings.append(Finding(
            Risk.INFO, "dot",
            f"tile {m}x{n} は {target} の直接 MMA 形状でない → タイル分解で対応（性能差注意）"))


def _parse_shape(t: str) -> tuple[int, int] | None:
    # "tensor<32x32xf32>" → (32, 32)
    try:
        inner = t[t.index("<") + 1: t.index(">")]
        dims = inner.split("x")
        return int(dims[0]), int(dims[1])
    except Exception:  # noqa: BLE001
        return None


def cross_vendor_diff(module: ir.Module,
                      targets: tuple[str, ...] = ("nvidia", "amd_cdna")) -> list[str]:
    """複数ベンダー間で移植リスクが *異なる* 箇所を抽出（等価性の要注意点）。"""
    reps = {t: analyze(module, t) for t in targets}
    diffs: list[str] = []
    keys = {(f.op, f.message) for t in targets for f in reps[t].findings}
    for op, msg in sorted(keys):
        present = [t for t in targets
                   if any(f.op == op and f.message == msg for f in reps[t].findings)]
        if len(present) != len(targets):
            diffs.append(f"{op}: '{msg}' は {present} のみ → ベンダー間で挙動差の疑い")
    return diffs
