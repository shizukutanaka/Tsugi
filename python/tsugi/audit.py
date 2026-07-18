"""Tsugi audit — 検証層を 1 つの判定に束ねる統合ファサード。

13 視点（portability/equivalence/occupancy/tolerance/feasibility/propagation/
envelope/decision/rollout/worstcase/decision拡張/attribution/blame）＋メタ層
（calibration/oracle_check）＋基盤（nondeterminism）が出揃った。
個別に呼ぶのでなく、traced IR ＋タイル構成から **静的に実行できる層をまとめて回し、
1 つの Audit レポートにする**。さらに *実機データが要る層*（実行時エンベロープ・
非決定性ノイズ・タスクフリップ）を「実行時チェックリスト」として明示し、検証の
ライフサイクル（静的 → 動的 → メタ → 基盤 → 翻訳）を一望できるようにする。

oracle を渡すと correctness 層が動き、shared-mode 検出に加え blame（新視点13）が
「どちらのベンダーを優先修正するか」を verdict に算入する（診断チェーンを製品経路で閉じる）。

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
    when: str          # "decided"（今 verdict に算入）/ "pending"（実機データ待ち）
    max_risk: Risk
    lines: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        tag = "" if self.when == "decided" else " [要実機データ]"
        head = f"[{self.max_risk.name:5s}] {self.name}{tag}"
        body = "\n".join("    " + ln for ln in self.lines)
        return head + ("\n" + body if body else "")


@dataclass
class Audit:
    phases: list[AuditPhase] = field(default_factory=list)
    certificate: object = None   # provenance.Certificate（verdict を環境に束ねる・陳腐化検出用）

    @property
    def decided_phases(self) -> list[AuditPhase]:
        """verdict に算入する層（実データで決定済み）。pending（実機待ち）は除く。"""
        return [p for p in self.phases if p.when == "decided"]

    def is_stale(self, **env: str) -> bool:
        """この verdict が計算されたスタックと現在の環境が違えば True（再検証要）。

        証明書が無ければ False（未スタンプ＝陳腐化判定不能）。
        """
        if self.certificate is None:
            return False
        from .provenance import is_stale
        return is_stale(self.certificate, **env)

    # 後方互換エイリアス（旧名）。
    @property
    def static_phases(self) -> list[AuditPhase]:
        return self.decided_phases

    @property
    def max_risk(self) -> Risk:
        """判定は decided 層のみから（pending=実機データ待ちは未確定ゆえ除外）。"""
        return max((p.max_risk for p in self.decided_phases), default=Risk.OK)

    @property
    def portable(self) -> bool:
        return self.max_risk < Risk.BLOCK

    def stamp(self, **env: str) -> "Audit":
        """verdict を環境フィンガープリントに束ねる（provenance）。再検証の起点。"""
        from .provenance import certify
        self.certificate = certify("portable" if self.portable else "blocked", **env)
        return self

    def to_text(self) -> str:
        lines = ["=== Tsugi audit（検証層の統合判定）==="]
        for p in self.phases:
            lines.append(p.to_text())
        verdict = "移植可（要注意点あり）" if self.portable else "移植ブロッカーあり"
        lines.append(f"\n判定（静的層）: {verdict} [max_risk={self.max_risk.name}]")
        if self.certificate is not None:
            lines.append("  " + self.certificate.to_text())
        return "\n".join(lines)


def _gemm_depth(module: ir.Module, cfg) -> int:
    """累積深さ K ≈ dot 反復数 × BK（マジックナンバーでなく構成由来）。"""
    n_dots = sum(1 for k in module.kernels for op in k.body if op.kind == "dot")
    bk = cfg.block_k if cfg is not None else 32
    return n_dots * bk


# IR op → propagation の論理 op への写像。memory op は数値発散に寄与しないので除く。
_AMPLIFY_KINDS = {"reduce", "exp", "rsqrt", "softmax", "div", "reciprocal"}
_SKIP_KINDS = {"load", "store", "zeros"}


def _classify_ops(ops, bk: int):
    """program 順の op 列を propagation の論理 GraphOp 列へ写す。

    連続 dot は 1 つの行列積（K=反復×BK）に集約し、memory op（_SKIP_KINDS）は数値
    発散に寄与しないので除く。reduce/exp 等は増幅 op、cast/add 等は elementwise
    (local のみ)。条件数 cond は静的には不明ゆえ既定 1（sample/実機で上書きされる前提）。
    """
    from .propagation import GraphOp

    out = []
    run_dots = 0
    for op in ops:
        if op.kind in _SKIP_KINDS:         # memory op: K ループを途切れさせない
            continue
        if op.kind == "dot":
            run_dots += 1
            continue
        if run_dots:                       # 実 compute op で dot 連を 1 matmul に集約
            out.append(GraphOp("matmul", K=run_dots * bk))
            run_dots = 0
        if op.kind in _AMPLIFY_KINDS:
            out.append(GraphOp(op.kind))
        else:                              # cast/to/add/scale 等は elementwise
            out.append(GraphOp("scale"))
    if run_dots:
        out.append(GraphOp("matmul", K=run_dots * bk))
    return out


def _identity_fork_merge(body, i: int, consumers: dict) -> int | None:
    """body[i] の結果が「恒等路つき単純フォーク」の起点なら合流 op の index を返す。

    SSA の use-def（Op.operands/Op.result）から検出する Case-A 限定の形:
      - result の消費者がちょうど 2 op（idx_a < idx_b）
      - 合流点 idx_b は result を *直接* operand に持つ（恒等 skip 路）
      - i+1..idx_b-1 の全 op がフォーク値だけから計算される一本鎖で、鎖の値が
        idx_b のもう一方の operand に到達し、中間値は合流点より先で消費されない
    residual（y = x + f(x)）と softmax の row 再利用（exp(row - reduce(row))）が
    この形に該当する。これ以外（多分岐・恒等路なし・外部値の混入・交差辺）は None を
    返し従来の線形扱いに落とす（不確実なら保守側に倒す fail-safe 慣例）。
    """
    op = body[i]
    if op.result is None:
        return None
    uses = consumers.get(op.result.name, [])
    if len(uses) != 2 or uses[0] == uses[1]:
        return None
    idx_b = uses[1]
    if idx_b - i < 2:                      # 計算路に最低 1 op（恒等×2 は対象外）
        return None
    chain = {op.result.name}
    for j in range(i + 1, idx_b):
        o = body[j]
        if not o.operands or any(v.name not in chain for v in o.operands):
            return None                    # load/zeros・外部値の混入 → 一本鎖でない
        if o.result is not None:
            if any(c > idx_b for c in consumers.get(o.result.name, [])):
                return None                # 中間値が合流の先で消費 → series-parallel でない
            chain.add(o.result.name)
    merge_operands = {v.name for v in body[idx_b].operands}
    if not merge_operands & (chain - {op.result.name}):
        return None                        # 鎖の値が合流点に届いていない
    return idx_b


def _computed_fork_merge(body, i: int, consumers: dict):
    """body[i] の結果が「2 本の計算路が合流する」フォークの起点なら
    (合流 index, branchA の op index 列, branchB の op index 列) を返す（A-12 Round 2）。

    恒等路の *無い* Case-B: フォーク値 r を 2 つの op が消費し（a0/b0）、それぞれが
    単一消費の一本鎖として伸び、共通の合流 op M で `M.operands == {tailA, tailB}`
    （両鎖の末端ちょうど 2 つ）として再合流する形。literal multi-head attention の
    ヘッド和・並列 2 経路の concat/加算がこれに該当し、`propagate_dag` の
    `[[branchA], [branchB]]` フォーク（恒等路なし）で表せる。

    偽OK を出さないため *完全に検証できた* 構造だけを受理する: 一本鎖でない・分岐が
    3 本以上・中間値が鎖外/合流の先で消費・領域(i,M)が 2 鎖で丁度覆われない、の
    いずれかなら None を返し従来の線形扱いに落とす（不確実なら保守側の fail-safe）。
    """
    op = body[i]
    if op.result is None:
        return None
    r = op.result.name
    uses = consumers.get(r, [])
    if len(uses) != 2 or uses[0] == uses[1]:
        return None

    def _trace(start):
        """start から単一消費の一本鎖を辿り (index 列, 合流 index, 到達名集合) を返す。"""
        names = {r}
        chain: list[int] = []
        cur = start
        while True:
            o = body[cur]
            if o.result is None:
                return None
            if not o.operands or any(v.name not in names for v in o.operands):
                return None                # 外部値の混入 → 一本鎖でない
            chain.append(cur)
            names.add(o.result.name)
            nxts = consumers.get(o.result.name, [])
            if len(nxts) != 1:
                return None                # 分岐/未使用 → 一本鎖でない
            m = nxts[0]
            if all(v.name in names for v in body[m].operands):
                cur = m                    # まだ枝の内側 → 鎖を延長
                continue
            return chain, m, names         # m は枝外 operand を持つ = 合流境界

    ra, rb = _trace(uses[0]), _trace(uses[1])
    if ra is None or rb is None:
        return None
    chain_a, m_a, _ = ra
    chain_b, m_b, _ = rb
    if m_a != m_b:
        return None                        # 別々の合流 → series-parallel でない
    m = m_a
    if set(chain_a) & set(chain_b):
        return None                        # 枝が交差 → 一般 DAG
    tails = {body[chain_a[-1]].result.name, body[chain_b[-1]].result.name}
    if len(tails) != 2:
        return None
    if {v.name for v in body[m].operands} != tails:
        return None                        # 合流が両末端ちょうどでない（r 直結や外部混入）
    if set(range(i + 1, m)) != set(chain_a) | set(chain_b):
        return None                        # 領域(i,m) が 2 鎖で丁度覆われない
    return m, chain_a, chain_b


def _detect_fork(body, i: int, consumers: dict, bk: int):
    """body[i] 起点のフォークを検出し (合流 index, propagate_dag フォークノード) を返す。

    恒等路つき（Case-A・residual/softmax）を先に、無ければ計算 2 分岐（Case-B・
    attention ヘッド和）を試す。どちらも該当しなければ None（線形扱い）。
    """
    m = _identity_fork_merge(body, i, consumers)
    if m is not None:
        branch = _classify_ops(body[i + 1:m], bk)
        if branch:
            return m, [[], branch]         # 恒等 skip 路 ＋ 計算路
    res = _computed_fork_merge(body, i, consumers)
    if res is not None:
        m, chain_a, chain_b = res
        b_a = _classify_ops([body[j] for j in chain_a], bk)
        b_b = _classify_ops([body[j] for j in chain_b], bk)
        if b_a and b_b:
            return m, [b_a, b_b]           # 計算 2 分岐（恒等路なし）
    return None


def _graph_ops(module: ir.Module, cfg):
    """traced IR を propagation 用の op グラフへ写す（SSA fork/merge 対応・A-12）。

    torch.compile(model) の op グラフを模す。従来は kernel body を線形走査するだけで
    Op.operands/Op.result の SSA 参照を捨てていたが、フォーク→マージ構造を use-def から
    検出し `propagate_dag` のフォークノードとして出す:
      - 恒等路つき（Case-A・residual y=x+f(x)・softmax の row 再利用）→ `[[], branch]`
      - 計算 2 分岐（Case-B・attention ヘッド和）→ `[[branchA], [branchB]]`
    検出できない形は従来通り線形（保守側）。返り値は GraphOp と
    フォーク（list[list[GraphOp]]）の混在列で、そのまま propagate_dag に渡せる。
    """
    bk = cfg.block_k if cfg is not None else 32
    nodes: list = []
    for k in module.kernels:
        body = k.body
        consumers: dict[str, list[int]] = {}
        for j, op in enumerate(body):
            for v in op.operands:
                consumers.setdefault(v.name, []).append(j)
        pending: list = []                 # 線形扱いの op を貯めて一括分類
        i = 0
        while i < len(body):
            detected = _detect_fork(body, i, consumers, bk)
            if detected is not None:
                merge_idx, fork_node = detected
                pending.append(body[i])          # フォーク元の op 自体は直列に分類
                nodes.extend(_classify_ops(pending, bk))
                pending = []
                nodes.append(fork_node)          # [[], branch] または [[A], [B]]
                pending.append(body[merge_idx])  # 合流 op は通常 op として δ に乗る
                i = merge_idx + 1
                continue
            pending.append(body[i])
            i += 1
        nodes.extend(_classify_ops(pending, bk))
    return nodes


def _iter_graphops(nodes):
    """_graph_ops の出力（GraphOp | フォーク list[list[GraphOp]]）の葉 GraphOp を列挙する。"""
    for node in nodes:
        if isinstance(node, list):
            for branch in node:
                yield from branch
        else:
            yield node


def audit(module: ir.Module, cfg=None, *, targets=TARGETS,
          block_dims=None, ref_logits=None, sample=None, provenance=None) -> Audit:
    """traced IR ＋構成から静的検証層をまとめて回し、1 つの判定に束ねる。

    ref_logits を渡すと、propagation のモデル発散を decision に橋渡しして、第2ベンダーを
    走らせる前にタスク判断フリップ率の上界を予測する（静的 → タスク影響）。

    sample を渡すと、認証エンベロープの scale を代表テンソルから実測する
    （envelope.certify_from_sample・SOCRATIC Q13/Q14）。無指定なら scale=1.0 を仮定するが、
    実テンソルの RMS は 1 でない（LLM 未正規化活性は数十、量子化後は小さい等）ため、
    scale=1 認証は本番 check_tensor でスケール超過を誤 BLOCK/見逃す原因になりうる。
    sample 無指定時は「scale=1 仮定」であることを numerics phase に明記し、暗黙化を防ぐ。
    """
    from .calibration import detectability_floor
    from .envelope import certify_from_sample, certify_gemm
    from .feasibility import cross_vendor_feasibility, first_vendor_only
    from .occupancy import cross_vendor_occupancy
    from .portability import analyze
    from .propagation import propagate_dag
    from .tolerance import explain

    a = Audit()

    # --- 静的: 移植性（per target） ---
    worst = Risk.OK
    port = AuditPhase("portability 移植リスク", "decided", Risk.OK)
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
        feas = AuditPhase("feasibility 起動可能性", "decided", Risk.OK)
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
        # 従来は nvidia/amd_cdna の 2 者間ギャップに固定されており、targets に amd_rdna が
        # 含まれていても一切報告されなかった（cross_vendor_occupancy は実装済みだが未使用・
        # portability/feasibility phase は targets 全体を見るのにここだけ 2 者固定だった）。
        # targets 全体を計測し、最大ギャップとどのペアで生じたかを報告する。
        occ_map = cross_vendor_occupancy(cfg, vendors=targets)
        pairs = [(u, v) for i, u in enumerate(targets) for v in targets[i + 1:]]
        gaps = [(u, v, abs(occ_map[u].occupancy - occ_map[v].occupancy)) for u, v in pairs]
        worst_gap = max((g for _, _, g in gaps), default=0.0)
        occ = AuditPhase("occupancy 占有率", "decided",
                         Risk.WARN if worst_gap >= 0.25 else Risk.INFO)
        for v in targets:
            occ.lines.append(f"{v}: occupancy={occ_map[v].occupancy:.0%}")
        if gaps:
            wu, wv, wg = max(gaps, key=lambda t: t[2])
            occ.lines.append(f"最大ギャップ = {wu}↔{wv} {wg:.0%}"
                             + ("（性能が片方だけ崩れる）" if wg >= 0.25 else ""))
        a.phases.append(occ)

    # --- 静的: 数値等価性の目安（導出許容 + 認証エンベロープ + 検出限界） ---
    K = _gemm_depth(module, cfg)
    if K > 0:
        num = AuditPhase("numerics 数値等価性の目安", "decided", Risk.INFO)
        num.lines.append("導出許容: " + explain(K, "float16"))
        if sample is not None:
            env = certify_from_sample(sample, K, "float16")
            num.lines.append(f"認証エンベロープ(sample 実測 scale={env.scale_max:.3g}): "
                             + env.to_text())
        else:
            env = certify_gemm(K, "float16", 1.0)
            num.lines.append("認証エンベロープ(scale=1.0 仮定・sample 未指定): " + env.to_text())
        floor = detectability_floor(K, "float16")
        num.lines.append(
            f"検出限界(偽OKの盲点): max_abs は相対 {floor['rel'] * 100:.1f}% 未満の"
            "系統誤差を見逃す → calibration.check_systematic で相補検査")
        a.phases.append(num)

    # --- 静的: 合成的等価性（per-kernel 等価 ⇏ per-model 等価） ---
    gops = _graph_ops(module, cfg)
    if gops:
        from .propagation import empirical_cond, is_amplifier

        # sample があれば増幅 op（reduce/softmax/exp）の cond を実データから実測する
        # （Q7/Q8/Q11: 静的 cond=1 は well-conditioned 仮定の下界・empirical_cond で置換）。
        cond_measured = False
        if sample is not None:
            for o in _iter_graphops(gops):
                if is_amplifier(o.kind) and o.cond == 1.0:
                    o.cond = empirical_cond(sample, o.kind)
                    cond_measured = True

        # correlated=True: 合流点の分岐発散を線形和 Σδ で合成する（保守側）。
        # クロスベンダー発散は系統的（相関）でありうる（calibration.check_systematic）ため、
        # 相関が不明な検証器は非対称コスト下で保守側を選ぶ——independent 仮定
        # （correlated=False の √Σδ²）は並列分岐を過小評価し偽OK の温床になりうる。
        # これにより DAG 発散は線形版 propagate を下回らない（fail-safe・過小評価しない）。
        pr = propagate_dag(gops, correlated=True)
        ratio = pr.model_divergence / (pr.naive_sum + 1e-30)
        # 発散が深さ/増幅でナイーブ和を大きく超えるならモデルレベルで要注意。
        prop = AuditPhase("propagation 合成的等価性", "decided",
                          Risk.WARN if ratio > 2.0 else Risk.INFO)
        prop.lines.append(
            f"モデル発散(予測)={pr.model_divergence:.2e}  "
            f"naive per-kernel 和={pr.naive_sum:.2e}  (×{ratio:.1f})")
        if pr.dominant is not None:
            prop.lines.append(f"支配的増幅 op = {pr.dominant.kind}（amp={pr.dominant.amp:.1f}）")
        if sum(1 for _ in _iter_graphops(gops)) == 1:
            prop.lines.append("単一 op グラフ: 伝播増幅なし。多 op モデルでは深さ・"
                              "条件数で累積（cond は実機/モデル依存・既定 1）")
        # 正直さ: データ依存増幅 op（reduce/exp）に静的 cond=1 を当てるのは *下界*。
        amps = sorted({o.kind for o in _iter_graphops(gops) if is_amplifier(o.kind)})
        if amps and cond_measured:
            prop.lines.append(
                f"データ依存増幅 op {amps} の cond を sample から実測済み（empirical_cond）: "
                "静的下界の過小評価を解消")
        elif amps and all(o.cond == 1.0 for o in _iter_graphops(gops)):
            prop.max_risk = Risk.WARN
            prop.lines.append(
                f"データ依存増幅 op {amps} が存在: 静的 cond=1 は *下界*（過小評価）。"
                "真の増幅は empirical_cond / audit_runtime で実データから定量化せよ"
                "（audit(sample=...) を渡せば自動実測）")
        # propagation → decision の橋: 静的発散を代表 logit でタスクフリップ率に翻訳
        if ref_logits is not None:
            from .decision import flip_bound_from_divergence
            bound = flip_bound_from_divergence(ref_logits, pr.model_divergence)
            prop.lines.append(
                f"タスク影響(予測): 判断フリップ率 ≤ {bound * 100:.2f}%"
                "（第2ベンダー実行前・代表 logit 分布から）")
            # Q15: 橋の仮定を明示する（暗黙化しない）。propagation は *op グラフの相対発散*
            # δ_rel を返し、橋は δ_abs = δ_rel·RMS(logits) として最終 logit に *そのまま*
            # 乗せる。だが δ_rel は GEMM 連の相対発散であり、最終 logit 層の相対発散は
            # 正規化（scale リセット）や最終射影の条件数で変わりうる（logit scale ≠ GEMM
            # 出力 scale）。分布シフト時（本番 logit が代表集合と違う）も予測は外れる。
            prop.lines.append(
                "  ↑ 仮定: op グラフ相対発散が最終 logit にそのまま乗る（正規化の scale "
                "リセット・最終射影の条件数・分布シフトで妥当域を外れうる・要再評価）")
        a.phases.append(prop)

    # --- 実行時（実機データが要る層をチェックリストとして明示） ---
    rt = AuditPhase("runtime 実行時チェックリスト", "pending", Risk.INFO)
    rt.lines += [
        "envelope.check_tensor(x, env): 本番入力が認証前提内か（overflow/denormal/scale）",
        "nondeterminism.compare_stable(runA, runB, K): run-to-run ノイズを実測し分布比較",
        "  クロス差 ≤ ノイズなら INDISTINGUISHABLE（等価判定は未定義）",
        "decision.compare_decisions(logitsA, logitsB): 判断フリップ率（タスク影響）",
        "  数値発散はマージン分布を介してフリップに翻訳・タスク予算で判定",
        "→ 実データがあれば audit_runtime(...) でこれらを実行し 1 判定に束ねる",
    ]
    a.phases.append(rt)
    a.stamp(**(provenance or {}))
    return a


def audit_runtime(a_out, b_out, K: int, *, dtype: str = "float16", env=None,
                  noise_floor: float = 0.0, logits_a=None, logits_b=None,
                  logits_oracle=None,
                  flip_budget: float = 0.001, oracle=None, provenance=None,
                  gen_length: int = 0, task: str = "classification",
                  task_kwargs: dict | None = None,
                  layers_a=None, layers_b=None, layers_oracle=None, x0=None,
                  layer_names=None,
                  fn_a=None, fn_b=None, worst_samples=None, worst_radius: float = 1.0,
                  worst_steps: int = 400, worst_seed: int = 0, worst_bounds=None,
                  worst_tol: float | None = None) -> Audit:
    """実行時チェックリストの *実行版*。実機/実データのクロスベンダー出力を束ねて判定する。

    静的 audit() の鏡像。与えられたデータに応じて適用可能な層だけ回す:
      - env があれば envelope.check_tensor（本番入力が認証前提内か）
      - calibration.check_systematic（max_abs の盲点に隠れる系統バイアス）
      - equivalence + nondeterminism（noise_floor を織り込んだ 3 状態帰属）
      - logits があれば decision 層（タスク判断フリップ）:
        task="classification"（既定）は compare_decisions（argmax・多クラス専用）。
        task="regression"/"binary"/"ranking" は compare_task へ委譲（非分類タスクの
        出荷判断・decision.compare_task はテスト済みだが従来 audit_runtime に未接続だった）。
        logits_oracle（真値の判断・classification のみ）を渡すと各ベンダーの判断誤り率も
        併記し、A↔B が一致（低フリップ率）でも両方 oracle 判断と食い違う task レベルの
        shared-mode を WARN する（Q31: フリップ率＝一致 ≠ 正しさ）。
      - oracle があれば correctness 層: oracle_check（oracle 自体の信頼性）＋
        detect_shared_mode（a≈b でも両方 oracle と不一致＝共有モード障害）
      - layers_a/layers_b（＋任意 x0）があれば attribution.diagnose で層別診断:
        「どの層で発散が始まるか（onset）」「どの層が支配的か（spike）」を特定し、
        layers_oracle も渡せば spike 層でどちらのベンダーが正しいかも責帰する。
        diagnose() は attribution+blame の集大成関数だが従来 audit_runtime に未接続だった
        （BLOCK になった時に「どこが悪いか」を追加で特定する診断チェーンを製品経路で閉じる）。
      - fn_a/fn_b（＋ worst_samples）があれば worstcase.analyze_worst_case で能動探索:
        代表サンプルの *典型* 発散だけでなく、認証エンベロープ内で発散を *最大化* する
        入力を微分フリー探索で能動的に探す（平均ケース等価 ⇏ 最悪ケース等価・envelope の
        受動検査と対をなす能動検査）。worst_tol 未指定なら既に計算済みの eq.atol を流用する
        （worst-case 探索に別基準を課したい場合は明示的に上書きできる）。
        analyze_worst_case は実装済みだが従来 audit_runtime に未接続だった
        （唯一の能動探索層が製品経路から欠落していた）。
    すべて実データで *決定済み* なので静的 verdict に算入する（when="decided"）。

    oracle 無しでは a≈b（portability）しか言えず correctness は未確定 —— shared-mode は
    原理的に検出不能（SPEC-verification §4.1）。oracle があって初めて正しさを問える。
    """
    import numpy as np

    from .calibration import check_systematic
    from .decision import compare_decisions, compare_task
    from .envelope import check_outlier_features, check_softmax_input, check_tensor
    from .equivalence import DV_LAYOUT, classify_divergence, compare_gemm
    from .nondeterminism import attribute

    ad = Audit()
    af = np.asarray(a_out, dtype=np.float64)
    bf = np.asarray(b_out, dtype=np.float64)

    # 形状不一致は即 BLOCK で打ち切る（誤ったテンソルを渡した・カーネルが間違った
    # 形状を返した等の構造的バグの証拠）。以降の全 phase は af/bf が同形状である
    # ことを前提に素朴な要素ごとの演算（引き算・比較）を行うため、ここで拒否しないと
    # NumPy の暗黙 broadcast に頼ることになり、broadcast が偶然成立する形状の組では
    # 偽 DIVERGENT にも偽 OK にもなりうる（equivalence.compare の同型修正と対）。
    if af.shape != bf.shape:
        sp = AuditPhase("equivalence 数値等価性", "decided", Risk.BLOCK)
        sp.lines.append(f"[SHAPE MISMATCH] a.shape={af.shape} vs b.shape={bf.shape} "
                        "→ 比較不能な構造的発散（broadcast による偽比較を拒否）")
        ad.phases.append(sp)
        ad.stamp(**(provenance or {}))
        return ad

    # envelope: 本番入力（両ベンダー出力）が認証前提内か。
    # check_softmax_input（fp16 exp-overflow）・check_outlier_features（単一 scale 仮定の
    # 破綻）は実装・テスト済みだったが従来この phase から呼ばれていなかった
    # （FEATURE-AUDIT.md A-1 と同型の facade 未接続。envelope.check_tensor だけが
    # 呼ばれ、同モジュールの他の検査が製品経路に届いていなかった）。
    if env is not None:
        ep = AuditPhase("envelope 実行時エンベロープ", "decided", Risk.OK)
        for name, x in (("A", af), ("B", bf)):
            r = check_tensor(x, env)
            ep.max_risk = max(ep.max_risk, r.max_risk)
            ep.lines.append(f"{name}: {'IN' if r.in_envelope else 'OUT'}-envelope "
                            f"(max_risk={r.max_risk.name})")
            outlier = check_outlier_features(x)
            if outlier.findings:   # .ok は BLOCK 未満なら True なので WARN を見逃す・findings で判定
                ep.max_risk = max(ep.max_risk, outlier.max_risk)
                ep.lines.append(f"{name}: {outlier.to_text()}")
        if logits_a is not None and logits_b is not None:
            for name, lg in (("A", logits_a), ("B", logits_b)):
                sm = check_softmax_input(np.asarray(lg), env)
                if sm.findings:
                    ep.max_risk = max(ep.max_risk, sm.max_risk)
                    ep.lines.append(f"{name} logits: {sm.to_text()}")
        ad.phases.append(ep)

    # equivalence（ノイズ床を織り込んだ 3 状態）+ systematic（相補・偽OK 対策）
    eq = compare_gemm(af, bf, K, dtype)
    cross = float(np.abs(af - bf).max())
    verdict = attribute(cross, noise_floor, eq.atol)
    eqp = AuditPhase("equivalence 数値等価性", "decided",
                     Risk.BLOCK if verdict == "DIVERGENT" else Risk.OK)
    eqp.lines.append(f"{verdict}: cross={cross:.2e} atol={eq.atol:.2e} noise={noise_floor:.2e}")
    if verdict == "INDISTINGUISHABLE":
        eqp.max_risk = Risk.WARN
        eqp.lines.append("クロス差 ≤ ノイズ → 等価判定は未定義（要 run 増/決定化）")
    elif verdict == "DIVERGENT" and af.shape == bf.shape:
        # レイアウト不一致（転置・再タイル等・値は正しい）か真の数値発散かを区別する
        # （classify_divergence は実装・テスト済みだが従来 audit_runtime に未接続だった）。
        # LAYOUT なら修正先は codegen の整列問題であり、数値精度のチューニングではない。
        dv = classify_divergence(af, bf, K, dtype)
        if dv == DV_LAYOUT:
            eqp.lines.append("LAYOUT: 値の多重集合は一致 → レイアウト不一致（転置/再タイル）"
                             "の疑い・数値精度バグでなく codegen の整列問題を調査せよ")
    sysrep = check_systematic(af, bf, K, dtype)
    if not sysrep.ok:
        eqp.max_risk = max(eqp.max_risk, sysrep.max_risk)
        eqp.lines.append(f"系統バイアス {sysrep.bias * 100:+.3f}% "
                         f"（max_abs 検出限界 {sysrep.floor_rel * 100:.1f}% の下に隠れる）")
    ad.phases.append(eqp)

    # decision: タスク判断のフリップ（最終単位）。既定は分類（argmax）・非分類は compare_task へ委譲。
    if logits_a is not None and logits_b is not None:
        if task == "classification":
            dr = compare_decisions(np.asarray(logits_a), np.asarray(logits_b),
                                   flip_budget=flip_budget)
            dp = AuditPhase("decision タスクレベル等価", "decided", dr.max_risk)
            dp.lines.append(f"判断フリップ率 {dr.flip_rate * 100:.2f}% "
                            f"(予算 {flip_budget * 100:.2f}%・上界 ≤{dr.predicted_bound * 100:.2f}%)")
            # Q31: フリップ率は A↔B の *一致* を測る（正しさではない）。oracle の判断
            # （argmax）が与えられれば各ベンダーの *正しさ* も併記する。両ベンダーが
            # 互いに一致（低フリップ率）していても、両方が oracle と食い違えば同一誤り
            # （task レベルの shared-mode）——一致だけを見ると見逃す偽OK。
            if logits_oracle is not None:
                from .decision import flip_rate as _flip_rate
                _lo = np.asarray(logits_oracle)
                err_a = _flip_rate(np.asarray(logits_a), _lo)
                err_b = _flip_rate(np.asarray(logits_b), _lo)
                dp.lines.append(
                    f"oracle 照合(正しさ): A 判断誤り {err_a * 100:.2f}% / "
                    f"B {err_b * 100:.2f}%（フリップ率＝一致 ≠ 正しさ）")
                # 両ベンダーとも oracle 判断からの誤りが予算超 → 一致しても両方誤り。
                if min(err_a, err_b) > flip_budget:
                    dp.max_risk = max(dp.max_risk, Risk.WARN)
                    dp.lines.append(
                        "task-level SHARED-MODE: A↔B は一致するが両方 oracle 判断と"
                        f"食い違う（min 誤り {min(err_a, err_b) * 100:.2f}% > 予算 "
                        f"{flip_budget * 100:.2f}%）——一致は正しさを意味しない")
        else:
            tr = compare_task(np.asarray(logits_a), np.asarray(logits_b), task=task,
                              flip_budget=flip_budget, **(task_kwargs or {}))
            dp = AuditPhase(f"decision タスクレベル等価({task})", "decided", tr.max_risk)
            dp.lines.append(f"{task} フリップ率 {tr.flip_rate * 100:.2f}% "
                            f"(予算 {flip_budget * 100:.2f}%・n={tr.n})")
        ad.phases.append(dp)

        # rollout: per-token フリップを生成長へ合成（自己回帰では複利的に増幅・新視点9）。
        # 自己回帰トークン選択（argmax）の概念ゆえ分類タスクに限る（regression/binary/ranking は対象外）。
        if gen_length > 0 and task == "classification":
            from .rollout import analyze_rollout, flip_rate_upper_bound
            # fail-safe: 点推定 dr.flip_rate でなく上側信頼限界を使う。0 フリップ観測でも
            # p=0 と過信せず、複利増幅した survival を過大評価しない（rollout_from_logits と整合）。
            p_safe = flip_rate_upper_bound(round(dr.flip_rate * dr.n), dr.n)
            rr = analyze_rollout(p_safe, gen_length)
            rp = AuditPhase("rollout 自己回帰的等価", "decided", rr.max_risk)
            rp.lines.append(f"L={gen_length}: survival={rr.survival * 100:.2f}%・"
                            f"safe_len={rr.safe_length}・p≤{p_safe * 100:.3f}%/tok(上側限界)"
                            f"（per-token 許容 ⇏ per-sequence 許容）")
            ad.phases.append(rp)

    # correctness 層（oracle がある時のみ）: 一致≠正しさ。oracle 信頼性＋共有モード障害。
    if oracle is not None:
        from .blame import compare_accuracy
        from .calibration import SM_DIVERGENT, SM_SHARED, detect_shared_mode
        from .oracle_check import verify_oracle
        oref = np.asarray(oracle, dtype=np.float64)
        cp = AuditPhase("correctness oracle 照合", "decided", Risk.OK)
        oracle_healthy = verify_oracle().ok
        if not oracle_healthy:
            cp.max_risk = Risk.BLOCK
            cp.lines.append("oracle 自体がメタモルフィック検証に失敗 → 真値として使えない")
        sm = detect_shared_mode(af, bf, oref, K, dtype)
        if sm == SM_SHARED:
            cp.max_risk = Risk.BLOCK
            cp.lines.append("SHARED_MODE: a≈b だが両方 oracle と不一致＝両ベンダー同一バグ"
                            "（cross-vendor 一致では検出不能・oracle 照合で発覚）")
        elif sm == SM_DIVERGENT:
            cp.lines.append("a≢b（cross-vendor が捕捉済み）")
        else:
            cp.lines.append("a≈b≈oracle: portability かつ correctness")
        # blame: oracle が健全な時だけ責帰を算入（不健全な oracle で blame すると誤指摘になる）
        if oracle_healthy:
            bl = compare_accuracy(af, bf, oref, tol=eq.atol)
            cp.max_risk = max(cp.max_risk, bl.max_risk)
            if bl.max_risk == Risk.OK:
                cp.lines.append(f"責帰: 両ベンダーとも oracle 距離 ≤ atol "
                                f"(A={bl.dist_a:.2e}/B={bl.dist_b:.2e}) — 責帰不要")
            elif bl.closer == "TIED":
                cp.lines.append(f"責帰: A({bl.dist_a:.2e})↔B({bl.dist_b:.2e}) 同程度 "
                                f"(ratio={bl.ratio:.1f}) — 方向不明・両実装/oracle を疑う")
            else:
                blamed = "B" if bl.closer == "A" else "A"
                cp.lines.append(f"責帰: vendor {bl.closer} が oracle に近い "
                                f"(A={bl.dist_a:.2e}/B={bl.dist_b:.2e}・ratio={bl.ratio:.1f}x) "
                                f"→ vendor {blamed} の実装を優先修正")
        else:
            cp.lines.append("責帰: oracle が不健全 — blame はスキップ（誤指摘を防ぐ）")
        ad.phases.append(cp)

    # attribution: layers_a/layers_b があれば層別に発散を追い onset/spike を特定する
    # （どこで発散が始まるか・どの層が支配的か。診断チェーンを製品経路で閉じる）。
    if layers_a is not None and layers_b is not None:
        from .attribution import diagnose
        dg = diagnose(layers_a, layers_b, layers_oracle, x0 if x0 is not None else a_out,
                     tol=eq.atol, names=layer_names)
        ap = AuditPhase("attribution 層別診断", "decided", dg.max_risk)
        ap.lines.append(dg.to_text())
        ad.phases.append(ap)

    # worstcase: fn_a/fn_b/worst_samples があれば認証エンベロープ内で発散を最大化する
    # 反例を能動探索する（平均ケース検証の盲点を露出・envelope の受動検査と対）。
    if fn_a is not None and fn_b is not None and worst_samples is not None:
        from .worstcase import analyze_worst_case
        wc = analyze_worst_case(fn_a, fn_b, worst_samples,
                                tol=worst_tol if worst_tol is not None else eq.atol,
                                radius=worst_radius, steps=worst_steps, seed=worst_seed,
                                bounds=worst_bounds)
        wp = AuditPhase("worstcase 能動探索", "decided", wc.max_risk)
        wp.lines.append(wc.to_text())
        ad.phases.append(wp)

    ad.stamp(**(provenance or {}))
    return ad


def audit_cross_vendor(run_a, run_b, K: int, *, dtype: str = "float16", env=None,
                       n_runs: int = 16, logits_a=None, logits_b=None,
                       flip_budget: float = 0.001, run_batch=None,
                       batch_tiles=(32, 64, 128, 256, 512), robust: bool = False,
                       provenance=None) -> Audit:
    """実機向けの入口: 各ベンダーの非決定性床を実測してから audit_runtime する。

    run_a/run_b: seed を受け取り出力テンソルを返す呼び出し可能（= 実 GPU カーネル）。
    run_batch: 任意。tile(=バッチ依存分割)を受け batch-invariance 床を測る呼び出し可能
      （2025 研究: バッチ変動が支配的非決定源）。与えれば実効床に max で織り込む。
    robust: True で run-to-run / batch-invariance 床に外れ値頑健な spread_robust
      （10-90 パーセンタイル幅）を使う（SOCRATIC Q49）。既定 False（max-min）は測定
      グリッチ 1 個で床が桁違いに膨張しうる —— nondeterminism.compare_stable の
      robust オプションと同じ理由でここにも露出する（実機入口が Q49 修正から漏れていた）。
    決定論を仮定せず、各ベンダーを n_runs 回走らせてノイズフロアを測り（視点7）、
    その noise を等価判定の床に織り込む。実機では run_* を実カーネルにするだけ。
    """
    from .nondeterminism import measure_batch_variance, measure_noise_floor

    key = "spread_robust" if robust else "spread"
    nf_a = measure_noise_floor(run_a, n_runs)
    nf_b = measure_noise_floor(run_b, n_runs)
    noise = max(nf_a[key], nf_b[key])
    if run_batch is not None:                       # batch-invariance 床を実効床に合流
        noise = max(noise, measure_batch_variance(run_batch, batch_tiles)[key])
    return audit_runtime(run_a(0), run_b(0), K, dtype=dtype, env=env,
                         noise_floor=noise, logits_a=logits_a, logits_b=logits_b,
                         flip_budget=flip_budget, provenance=provenance)
