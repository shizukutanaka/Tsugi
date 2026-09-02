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

from .arrays import asarray
from .constants import SAFETY
from .tolerance import unit_roundoff

# *相対*誤差を増幅する op（実測で確認）。相対発散の枠組みでは reciprocal/div/add は
# 相対条件数 ~1（増幅しない）。真の相対増幅は (1) 符号付き reduction の相殺、
# (2) exp（相対条件数=|x|）、(3) LayerNorm（平均優勢入力 μ/RMS→1 で amp≈RMS/σ・
# ヤコビアン J=(g/σ)(I−11ᵀ/d−ŷŷᵀ/d) の最大特異値 g/σ から。実 LayerNorm への
# 数値実験で検証済み——shift=10 で実測 amp≈10 を RMS/σ≈10.7 が上界する）。
# RMSNorm は含めない: J=(g/r)(I−ŷŷᵀ) で相対増幅は無条件に ≤1（文献も unconditional
# forward stability を報告・docs/SOURCES.md）。ただし 1 未満の減衰係数は入れない
# （未検証係数の禁止・amp=1.0 固定が保守側）。cond を明示するとその大きさを反映する。
_AMPLIFYING = {"reduce", "softmax", "exp", "layer_norm"}


def is_amplifier(kind: str) -> bool:
    """この op が *相対*誤差を増幅しうるか（reduce 相殺・exp・平均優勢 LayerNorm）。"""
    return kind in _AMPLIFYING


@dataclass
class GraphOp:
    """グラフ中の 1 op。matmul は K（累積深さ）、増幅 op は cond（条件数）を持つ。"""

    kind: str
    K: int = 1
    dtype: str = "float16"
    cond: float = 1.0   # 条件数（>1 で ill-conditioned・発散を増幅）
    safety: float = SAFETY
    residual: bool = False   # True なら y=x+f(x) の残差ブロック（発散が希釈される）


def local_divergence(op: GraphOp) -> float:
    """入力が一致でも両ベンダーで正当に生じる相対発散（accumulation 等）。"""
    u = unit_roundoff(op.dtype)
    if op.kind == "matmul":
        # 累積順序差: tolerance.expected_gemm_abs_error と整合（相対・scale 抜き）
        return op.safety * math.sqrt(max(1, op.K)) * u
    if op.kind in ("reduce", "softmax", "layer_norm", "rms_norm"):
        # reduction の丸めは条件数で増幅されうる（正規化層も内部に mean/var 縮約を持つ。
        # rms_norm は cond=1 のままなので実質 safety·u＝elementwise と同じ）
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

    通常 op（出力が入力を置換）: δ_out = amp · δ_in + local（線形に累積）。
    残差 op（y = x + f(x)）: skip 接続が x をそのまま運ぶので δ_in は *再増幅されず*、
    ブロックの寄与 (amp·local) だけが random-walk で加わる →
        δ_out = sqrt(δ_in² + (amp·local)²)
    これが深い残差ネットが安定な理由（一次近似）。同じ深さでも残差は線形累積より
    緩やかに（~√L）増え、発散が *希釈* される。pre-norm transformer の numpy 実測で
    残差 < 平坦チェーンを確認済み（test_propagation）。

    返り値の model_divergence が「モデルレベルで正当に生じうる発散」= per-model 許容の目安。
    """
    rep = PropagationReport()
    delta = input_div
    for op in ops:
        loc = local_divergence(op)
        amp = amplification(op)
        if op.residual:
            delta = math.sqrt(delta ** 2 + (amp * loc) ** 2)
        else:
            delta = amp * delta + loc
        rep.ops.append(OpTrace(kind=op.kind, local=loc, amp=amp, cumulative=delta))
    return rep


def merge_divergence(divs, *, correlated: bool = False) -> float:
    """並列ブランチの末端発散を合流（merge）点で合成する。

    transformer の合流（multi-head attention のヘッド和・残差加算・gated 経路・concat）は
    複数ブランチの発散を 1 本にまとめる。合成則は相関仮定に依る:
      - `correlated=False`（既定・random-walk）: δ = √(Σ δ_i²)。各ブランチの丸めが独立
        （別カーネル・別累積）なら発散は二乗平均で *希釈* される（残差が安定な理由と同根）。
      - `correlated=True`（worst-case・保守）: δ = Σ δ_i。系統モードを共有する（同じ ill-cond
        な入力を全ブランチが処理する等）なら線形に加わる。検証器の非対称コスト下では
        相関が不明なら保守側（True）を選ぶ。
    """
    ds = [max(0.0, float(d)) for d in divs]
    if not ds:
        return 0.0
    return sum(ds) if correlated else math.sqrt(sum(d * d for d in ds))


def propagate_dag(nodes, input_div: float = 0.0, *,
                  correlated: bool = False) -> PropagationReport:
    """直列 op と並列フォークが混在する series-parallel グラフに沿って発散を伝播する。

    `propagate`（線形列）の一般化。`nodes` の各要素は:
      - `GraphOp`            : 直列 op（δ をそのまま通す。residual フラグも従来通り効く）。
      - `list[list[GraphOp]]`: フォーク。現在の δ から各ブランチを *独立に* propagate し、
        末端で `merge_divergence` により合流させる。空ブランチ `[]` は恒等（skip）路として
        δ をそのまま運ぶ。

    これで attention（並列ヘッド→和）・残差（恒等＋f→加算）・concat 等の DAG を表現できる。
    各フォークはブランチが現在の発散 δ_in を *再処理* するものとして扱う（f が発散入力を
    見る実態に即し、非対称コスト下で保守側）。**対象は series-parallel まで**——交差辺を
    もつ一般 DAG（重み共有・cross-attention の往復等）は線形/SP 近似に留まる。
    """
    rep = PropagationReport()
    delta = input_div
    for node in nodes:
        if isinstance(node, GraphOp):
            loc = local_divergence(node)
            amp = amplification(node)
            if node.residual:
                delta = math.sqrt(delta ** 2 + (amp * loc) ** 2)
            else:
                delta = amp * delta + loc
            rep.ops.append(OpTrace(kind=node.kind, local=loc, amp=amp, cumulative=delta))
        else:   # フォーク: 各ブランチを現在の δ から伝播し merge で合流
            branch_terms = []
            for branch in node:
                sub = propagate(list(branch), input_div=delta)
                rep.ops.extend(sub.ops)                    # トレース可視化のため残す
                branch_terms.append(sub.model_divergence if branch else delta)
            delta = merge_divergence(branch_terms, correlated=correlated)
            rep.ops.append(OpTrace(kind=f"merge×{len(node)}", local=0.0, amp=1.0,
                                   cumulative=delta))
    return rep


def empirical_cond(sample, kind: str, axis: int = -1, reduce_kind: str = "sum",
                   eps: float = 1e-5) -> float:
    """代表サンプルからデータ依存の *相対* 条件数を実測する（静的 cond=1 の置換）。

    静的には cond 不明（符号や値域に依存）。実機/代表データがあれば測れる:
      - reduce(sum): 和の条件数 κ = Σ|x| / |Σx|（相殺で増大・正の和なら ~1）。worst-case
        だが検証器は非対称コストゆえ保守側でよい。reduce(max) は ~1。
      - exp: 相対条件数 = max|x|。
      - layer_norm: 行ごとの RMS/√(Var+eps) の **max**。LayerNorm y=(x−μ)/√(σ²+eps) の
        ヤコビアン最大特異値は 1/√(σ²+eps) で、相対 RMS 増幅は RMS(x)/√(σ²+eps) が
        上界（実 LayerNorm への数値実験で検証済み）。零平均なら ≈1・平均優勢なら ≫1。
        median でなく max を使う理由: 外れ行（massive activation 型・零平均多数派の中に
        平均優勢行が 1 本）を median は隠し偽OK になる。reduce の median と違い max が
        暴走しない根拠は eps ガード——比は RMS/√eps で有界（相殺のような発散が無い）。
        eps=1e-5 は発明した係数ではなく torch.nn.LayerNorm の既定値＝実装が実際に割る数
        （docs/SOURCES.md）。それより小さい custom eps の近定数行では上界を超えうる
        （病的ケース・正直な限界として記す）。
      - rms_norm: 1.0（増幅は無条件に ≤1 を実測検証済み。ただし 1 未満の減衰は
        未検証係数になるため入れず、1.0 に固定する）。
      - その他（div/reciprocal/add 等）: 相対的には ~1。
    audit_runtime からこれを与えれば propagation の増幅が実データで発火する。
    注意（既存の reduce/exp と共通の制限）: sample は *ネットワーク入力* であり、深部の
    正規化層が実際に見る活性ではない（bias 等で平均がシフトしうる・分布シフト未追跡）。
    """
    import numpy as np

    x = asarray(sample, dtype=np.float64)
    if kind in ("reduce", "softmax"):
        if reduce_kind == "max":
            return 1.0
        num = np.sum(np.abs(x), axis=axis)
        den = np.abs(np.sum(x, axis=axis))
        ratio = num / np.maximum(den, 1e-30)
        return float(np.median(ratio))
    if kind == "layer_norm":
        if not x.size:
            return 1.0
        rms = np.sqrt(np.mean(x ** 2, axis=axis))
        sd = np.sqrt(np.maximum(x.var(axis=axis), 0.0) + eps)
        return float(np.max(rms / sd))
    if kind == "exp":
        return float(np.abs(x).max()) if x.size else 1.0
    return 1.0


def model_tolerance(ops: list[GraphOp]) -> float:
    """グラフ全体に対する許容相対誤差（合成済み）。

    per-kernel 検証はこの値でなく各 op の local を見る。両者の差が
    「per-kernel 等価 ⇏ per-model 等価」の定量化。
    """
    return propagate(ops).model_divergence
