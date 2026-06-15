"""Tsugi propagation — クロスベンダー発散の合成的検証（ソクラテス問答・第4ラウンド）。

盲点: equivalence / tolerance / feasibility / portability はすべて *単一カーネル* を
判定する。だがユーザーは torch.compile(model) で *モデル = op グラフ* をコンパイルする。
各カーネルが許容内でも、ベンダー間の数値発散はグラフを *伝播* し、深さに比例して累積し、
ill-conditioned な op（相殺を伴う reduction・小値除算・大値 exp）で *増幅* される。
→ per-kernel 等価 ⇏ per-model 等価。検証単位はカーネルでなくグラフであるべき。

モデル（誠実な近似）: 各 op の出力相対発散 δ_out は
  δ_out = amp * δ_in + local
  - local: 入力が完全一致でも両ベンダーで生じる発散（matmul の累積順序差など）
  - amp  : 入力発散をどれだけ増幅するか（well-conditioned で 1・条件数で増大）
これを op 列に沿って合成する。厳密な最悪ケース境界でなく、深さと条件数の効果を
可視化するための一次近似（tolerance.py と同じ safety 係数で粗さを吸収）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .tolerance import unit_roundoff

# amp が 1 を超える（入力発散を増幅する）代表的な op。既定は well-conditioned(amp=1)。
# cond を明示すると条件数依存の増幅を反映する（相殺・小値除算・大値 exp）。
_AMPLIFYING = {"reduce", "add", "exp", "reciprocal", "rsqrt", "div", "softmax"}


@dataclass
class GraphOp:
    """グラフ中の 1 op。matmul は K（累積深さ）、増幅 op は cond（条件数）を持つ。"""

    kind: str
    K: int = 1
    dtype: str = "float16"
    cond: float = 1.0   # 条件数（>1 で ill-conditioned・発散を増幅）
    safety: float = 4.0


def local_divergence(op: GraphOp) -> float:
    """入力が一致でも両ベンダーで正当に生じる相対発散（accumulation 等）。"""
    u = unit_roundoff(op.dtype)
    if op.kind == "matmul":
        # 累積順序差: tolerance.expected_gemm_abs_error と整合（相対・scale 抜き）
        return op.safety * math.sqrt(max(1, op.K)) * u
    if op.kind in ("reduce", "softmax"):
        # reduction の丸めは条件数で増幅されうる
        return op.safety * u * op.cond
    # elementwise/cast/scale: 1 回の丸め
    return op.safety * u


def amplification(op: GraphOp) -> float:
    """入力発散をどれだけ増幅して下流へ渡すか（well-conditioned で 1）。"""
    if op.kind in _AMPLIFYING:
        return max(1.0, op.cond)
    # matmul/scale/cast は正規化済みネットなら相対発散をほぼ保つ（amp≈1）
    return 1.0


@dataclass
class OpTrace:
    kind: str
    local: float
    amp: float
    cumulative: float   # この op を通過した後の累積相対発散


@dataclass
class PropagationReport:
    ops: list[OpTrace] = field(default_factory=list)

    @property
    def model_divergence(self) -> float:
        """グラフ全体を通過した後の予測相対発散（= モデルレベル許容の目安）。"""
        return self.ops[-1].cumulative if self.ops else 0.0

    @property
    def naive_sum(self) -> float:
        """per-kernel 思考の素朴な見積り（local の単純和・増幅を無視）。"""
        return sum(o.local for o in self.ops)

    @property
    def dominant(self) -> OpTrace | None:
        """累積発散を最も押し上げた op（cumulative の増分が最大）。"""
        if not self.ops:
            return None
        gains, prev = [], 0.0
        for o in self.ops:
            gains.append((o.cumulative - prev, o))
            prev = o.cumulative
        return max(gains, key=lambda g: g[0])[1]

    def to_text(self) -> str:
        lines = [f"propagation report ({len(self.ops)} ops)"]
        prev = 0.0
        for o in self.ops:
            gain = o.cumulative - prev
            prev = o.cumulative
            lines.append(
                f"  {o.kind:10s} local={o.local:.2e} amp={o.amp:.1f} "
                f"→ cum={o.cumulative:.2e} (+{gain:.2e})")
        lines.append(f"  model_divergence = {self.model_divergence:.2e}")
        lines.append(f"  naive per-kernel sum = {self.naive_sum:.2e} "
                     f"(×{self.model_divergence / (self.naive_sum + 1e-30):.1f} 過小評価)")
        d = self.dominant
        if d is not None:
            lines.append(f"  dominant amplifier = {d.kind} (amp={d.amp:.1f})")
        return "\n".join(lines)


def propagate(ops: list[GraphOp], input_div: float = 0.0) -> PropagationReport:
    """op グラフ（線形列）に沿ってベンダー間相対発散を合成する。

    δ_out = amp * δ_in + local を順に適用。返り値の model_divergence が
    「モデルレベルで正当に生じうる発散」= per-model 許容の目安。
    """
    rep = PropagationReport()
    delta = input_div
    for op in ops:
        loc = local_divergence(op)
        amp = amplification(op)
        delta = amp * delta + loc
        rep.ops.append(OpTrace(kind=op.kind, local=loc, amp=amp, cumulative=delta))
    return rep


def model_tolerance(ops: list[GraphOp]) -> float:
    """グラフ全体に対する許容相対誤差（合成済み）。

    per-kernel 検証はこの値でなく各 op の local を見る。両者の差が
    「per-kernel 等価 ⇏ per-model 等価」の定量化。
    """
    return propagate(ops).model_divergence
