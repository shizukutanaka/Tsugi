"""FX グラフ → Tsugi 検証層の橋（torch.compile で「検証だけ先に届ける」）。

GPU codegen 完成前でも、torch.compile(model, backend="tsugi") の FX グラフに対し
静的検証（propagation/増幅 op の可視化）を走らせて警告を出せる —— これが楔の早期価値。

torch を import せずに動く（FX ノードは duck-typed で読む）。写像は stand-in グラフに加え
**実 `torch.fx.symbolic_trace` の出力でも検証済み**（`tests/correctness/test_fxlower.py`・
torch が無い環境ではその部分だけ正直に skip する）。

実 FX で検証して初めて判った欠陥（第 60 回）: `call_module` の target は "0"/"1" という
経路名で op の種類を表さない。`resolved_target` で解決しないと `_kind_of` が全ノードで
None を返し、**実モデルが必ず「0 numeric ops・発散 0」＝無害判定になる**。stand-in は
aten 名を使うため露見しなかった。duck-typed 検証の限界を示す実例として記録する。
"""
from __future__ import annotations

from typing import Any

from tsugi.nondeterminism import classify_nondeterminism
from tsugi.propagation import GraphOp, is_amplifier, propagate

_CALL_OPS = ("call_function", "call_method", "call_module")


def resolved_target(node: Any, gm: Any = None) -> str:
    """ノードの op 種別を表す名前。`call_module` はサブモジュールのクラス名へ解決する。

    **これが無いと実 `torch.fx.symbolic_trace` に対して全滅する**: Sequential の
    子モジュールは target が "0"/"1"/"2" という経路名で、op の種類を一切表さない。
    解決しないと `_kind_of` が全ノードで None を返し、`audit_fx` は
    「0 numeric ops・発散 0」を報告する——**想定ユーザーのモデルが必ず無害判定になる
    という最悪の偽OK**。stand-in グラフは aten 名を使っていたため露見しなかった
    （実 torch を入れて初めて判明・第 60 回）。
    """
    t = str(getattr(node, "target", ""))
    if getattr(node, "op", None) == "call_module" and gm is not None:
        try:
            return type(gm.get_submodule(t)).__name__
        except (AttributeError, KeyError, TypeError):
            return t
    return t


def _matches(t: str, *pats: str) -> bool:
    """アンダースコアの有無を無視して照合する。

    aten 名（`native_layer_norm`）とモジュールのクラス名（`LayerNorm`）で区切りが
    違うため、素朴な部分文字列一致では実 FX のモジュール名を取りこぼす。
    """
    low = t.lower()
    flat = low.replace("_", "")
    return any(p in low or p.replace("_", "") in flat for p in pats)


def _kind_of(target_name: str) -> str | None:
    """aten/torch op 名を Tsugi の論理 op 種別へ写す（None=数値発散に無関係）。"""
    t = target_name.lower()
    if _matches(t, "addmm", "mm", "matmul", "bmm", "linear", "einsum", "conv"):
        return "matmul"
    if _matches(t, "softmax"):
        return "softmax"
    if _matches(t, "rsqrt"):
        return "rsqrt"
    if _matches(t, "exp"):
        return "exp"
    # 正規化層は専用 kind（従来は "reduce" に写していたが、reduce の cond 統計
    # Σ|x|/|Σx| は正規化に不適切で *両方向* に誤る——零平均 sample では相殺で爆発
    # （偽BLOCK）・平均優勢 sample では ≈1 なのに実 LayerNorm は RMS/σ 倍に増幅
    # （偽OK）。rms_norm 判定が先（rms_norm は "_norm" を含むため順序が本質）。
    if _matches(t, "rms_norm"):
        return "rms_norm"
    if _matches(t, "layer_norm", "_norm") or t.endswith("norm"):
        # group/batch/instance norm・linalg_vector_norm 等も平均減算や縮約を含むため
        # 保守的に layer_norm 扱い（cond は過大側に外れうるが偽BLOCK 方向・実験なしに
        # 特別扱いしない）。
        return "layer_norm"
    if _matches(t, "mean", "sum", "var"):
        return "reduce"
    if _matches(t, "add", "sub", "mul", "div", "tanh", "gelu", "relu",
                "sigmoid", "silu", "cast", "to_copy", "scale", "neg"):
        return "scale"
    return None   # placeholder/output/get_attr/view 等は除外


def _node_K(node: Any, default: int = 512, gm: Any = None) -> int:
    """縮約長 K の目安。優先順: Linear の `in_features` → shape meta → 既定。

    `torch.fx.symbolic_trace` は tensor_meta を付けないので、従来は **常に既定 512**
    に落ちていた（第 62 回の実測: K=64 の層に 512 が当たり √8≈2.8× の過大）。
    `call_module` なら重みの形から K が **厳密に** 判るので、そちらを最優先にする。
    """
    if getattr(node, "op", None) == "call_module" and gm is not None:
        try:
            sub = gm.get_submodule(str(node.target))
            k = getattr(sub, "in_features", None)
            if k is not None:
                return int(k)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    meta = getattr(node, "meta", None) or {}
    tm = meta.get("tensor_meta") if isinstance(meta, dict) else None
    shape = getattr(tm, "shape", None) if tm is not None else None
    if shape:
        try:
            return int(shape[-1])
        except (TypeError, ValueError, IndexError):
            return default
    return default


def _node_is_symbolic(node: Any) -> bool:
    """shape meta に symbolic 次元（torch.SymInt）が含まれるかを検査する。

    torch.compile(dynamic=True) や export 経路では、次元が具体的な int でなく
    torch.SymInt になる。SymInt は int() 変換で TypeError/ValueError を送出するため、
    その失敗で symbolic を判定する。

    shape guard 効果: symbolic shape があると torch.compile は実行時形状ごとに
    ガードを立て、ガード違反で再コンパイルを行う。形状別特化カーネルはタイル幅・
    縮約順序・アキュムレータ幅が変わり得るため、等価性は 1 形状のみ認証では不十分。
    実際の運用形状をカバーする per-shape 検証が必要。
    """
    meta = getattr(node, "meta", None) or {}
    tm = meta.get("tensor_meta") if isinstance(meta, dict) else None
    shape = getattr(tm, "shape", None) if tm is not None else None
    if not shape:
        return False
    for dim in shape:
        try:
            int(dim)
        except (TypeError, ValueError):
            return True
    return False


def fx_to_graph_ops(gm: Any) -> list[GraphOp]:
    """FX GraphModule を propagation 用の論理 op 列へ写す（duck-typed・torch 不要）。"""
    ops: list[GraphOp] = []
    for node in gm.graph.nodes:
        if getattr(node, "op", None) not in _CALL_OPS:
            continue
        kind = _kind_of(resolved_target(node, gm))
        if kind is None:
            continue
        ops.append(GraphOp(kind, K=_node_K(node, gm=gm)) if kind == "matmul"
                   else GraphOp(kind))
    return ops


def _is_normalization(target_name: str) -> bool:
    """op 名が LayerNorm/RMSNorm 系（正規化）かを判定する。

    正規化層は scale-invariant（LN(c·x)≈LN(x)）だが、それは「相対発散を増幅しない」
    ことを意味しない（A-5 の数値実験で判明・当初の想定が反転した事例）:
      - RMSNorm: 相対増幅は無条件に ≤1（J=(g/r)(I−ŷŷᵀ)・実測検証済み）→ amp=1 固定。
      - LayerNorm: 平均優勢入力（μ/RMS→1）では amp≈RMS/σ に **増幅** する
        （J=(g/σ)(…) の最大特異値 g/σ・shift=10 の実測 amp≈10）。sample があれば
        empirical_cond("layer_norm") が行ごとの RMS/√(σ²+eps) の max を実測する。
    旧実装はどちらも "reduce" に写し、Σ|x|/|Σx| という不適切な統計で両方向に誤っていた
    （零平均で偽BLOCK・平均優勢で偽OK）。この述語は has_normalization の可視化用に残す。
    """
    t = target_name.lower()
    return _matches(t, "layer_norm", "rms_norm", "_norm") or t.endswith("norm")


def fx_call_target_names(gm: Any) -> list[str]:
    """FX グラフの呼び出しノードの raw target 名を列挙する（非決定 op 照合用）。

    _kind_of は scatter_add/index_add 等を論理 op に畳まないため、生 target 名を別途
    取り出して nondeterminism カタログに照合する（atomicAdd 由来の非決定検出）。
    """
    names: list[str] = []
    for node in gm.graph.nodes:
        if getattr(node, "op", None) in _CALL_OPS:
            names.append(resolved_target(node, gm))
    return names


def audit_fx(gm: Any, ref_logits=None, sample=None) -> dict:
    """FX グラフに静的検証（propagation）を走らせ、要点を dict で返す。

    codegen 前でも「このモデルはクロスベンダーでどれだけ発散しうるか・どの増幅 op が
    あるか」を告げる。cond は静的不明ゆえ既定 1（=下界・実機/実データで定量化すべき）。

    ref_logits（代表的な出力 logit 分布）を渡すと、モデル発散を *タスク影響* に翻訳し
    判断フリップ率の上界 `task_flip_bound` を返す（静的グラフ → ユーザーに見える差）。

    sample（代表的な *入力* テンソル）を渡すと「scale=1 / cond=1」の暗黙仮定を実測で
    置き換える（FEATURE-AUDIT.md A-3）。`audit()` は B-1a/B-1b で既にこれを持っていたが
    torch 経路には無く、実 LLM では致命的だった——massive activations は中央値の
    ~1000 倍に達し（docs/SOURCES.md）、scale=1 仮定は認証 atol を桁で誤らせる。
    返り値に `sample_scale`（実 RMS）・`channel_spread`（外れチャネル検出）・
    `cond_measured` を追加し、増幅 op の cond を実測値で置き換えて発散を再計算する。
    """
    ops = fx_to_graph_ops(gm)
    rep = propagate(ops)
    amps = sorted({o.kind for o in ops if is_amplifier(o.kind)})
    # atomicAdd 由来の非決定 op を静的に検出（PyTorch 公式カタログ照合）。
    # これらがあれば静的許容では不十分で、実機 noise floor 実測が必須。
    nondet = classify_nondeterminism(fx_call_target_names(gm))
    # dynamic shape 検出: torch.compile(dynamic=True) や export 経路では shape が
    # torch.SymInt になる。形状ごとにカーネルが特化されるため、
    # 1 形状で認証した等価性は他の形状に転用できない（per-shape 再検証が必要）。
    has_dynamic_shapes = any(
        _node_is_symbolic(node)
        for node in gm.graph.nodes
        if getattr(node, "op", None) in _CALL_OPS
    )
    # 正規化層（LayerNorm/RMSNorm）の有無を検出する。propagate() は専用 kind で扱う:
    # RMSNorm は amp=1（無条件安定・実測検証済み）、LayerNorm は平均優勢入力で
    # amp≈RMS/σ に増幅しうる（sample があれば empirical_cond で実測・A-5）。
    has_normalization = any(
        _is_normalization(resolved_target(node, gm))
        for node in gm.graph.nodes
        if getattr(node, "op", None) in _CALL_OPS
    )
    out = {
        "n_ops": len(ops),
        "model_divergence": rep.model_divergence,
        "naive_sum": rep.naive_sum,
        "amplifiers": amps,
        "dominant": rep.dominant.kind if rep.dominant is not None else None,
        "nondeterministic_ops": list(nondet.nondet_ops),
        "requires_noise_floor": nondet.requires_noise_floor,
        "has_dynamic_shapes": has_dynamic_shapes,
        "has_normalization": has_normalization,
        "task_flip_bound": None,
    }
    if sample is not None:
        # 代表入力があれば「scale=1 / cond=1」の暗黙仮定を実測で置き換える
        # （`audit()` が B-1a/B-1b で既に持っていた機能。torch 経路には無かった＝A-3）。
        import numpy as _np

        from tsugi.envelope import channel_scale_spread
        from tsugi.propagation import empirical_cond
        x = _np.asarray(sample, dtype=_np.float64)
        if x.size:
            out["sample_scale"] = float(_np.sqrt(_np.mean(x ** 2)))
            # massive activations（一部チャネルが中央値の ~1000 倍）の検出。
            # 実 LLM 活性はこの型の外れ値を持ち、単一 scale 仮定が破れる
            # （docs/SOURCES.md「outlier feature / massive activations」節）。
            out["channel_spread"] = channel_scale_spread(x)
            # データ依存 cond を実測して伝播をやり直す（静的 cond=1 は *下界*）。
            measured = False
            for o in ops:
                if is_amplifier(o.kind) and o.cond == 1.0:
                    o.cond = empirical_cond(x, o.kind)
                    measured = True
            if measured:
                rep = propagate(ops)
                out["model_divergence"] = rep.model_divergence
                out["naive_sum"] = rep.naive_sum
                out["dominant"] = rep.dominant.kind if rep.dominant is not None else None
            out["cond_measured"] = measured
    if ref_logits is not None:
        import numpy as _np
        from tsugi.decision import flip_bound_from_divergence
        out["task_flip_bound"] = flip_bound_from_divergence(ref_logits, rep.model_divergence)
        # 実 logit の RMS scale を測定し certify_from_sample の代替 scale として公開する。
        # audit_fx を呼んだ後に certify_from_sample(x, K, dtype) へ渡す目安になる。
        _rf = _np.asarray(ref_logits, dtype=_np.float64)
        out["ref_scale"] = float(_np.sqrt(_np.mean(_rf ** 2))) if _rf.size else 1.0
    return out


def audit_torch(gm: Any, *, ref_logits=None, sample=None,
                flip_budget: float = 0.001) -> "Any":
    """FX グラフを **ゲート付き判定**（`Audit`）として返す——PyTorch 経路の製品入口。

    `audit_fx` は素の dict を返すため、CI ゲート契約（`exit_code`）も人間可読レポート
    （`to_text`）も持たなかった。**このプロダクトの楔は torch.compile（フレームワーク層）**
    なのに、ゲート付きの判定は tile-DSL 経路だけが持っていた——想定ユーザー（PyTorch 開発者）が
    出荷判断に使えない状態だった。ここで両経路の契約を揃える（`tsugi.verify(gm)` から到達）。

    fail-safe の要（静的 FX で BLOCK を捏造しない）:
    静的グラフだけでは *等価性を認証できない*（第 2 ベンダーの実出力が無い）。よって
    発散量の絶対値に閾値を発明して BLOCK にはしない（未検証係数の禁止）。BLOCK にするのは
    **利用者が与えた予算を超えたときだけ**——`ref_logits` があれば `task_flip_bound`
    （判断フリップ率の上界）を `flip_budget` と比較する。それ以外は WARN/INFO に留め、
    実機クロスベンダー照合が要ることを pending phase で明示する。
    """
    from tsugi.audit import Audit, AuditPhase
    from tsugi.report import Risk

    rep = audit_fx(gm, ref_logits=ref_logits, sample=sample)
    ad = Audit()

    p = AuditPhase("torch/FX 静的監査", "decided", Risk.INFO)
    # 「予測」と書いていたが実測の 100〜1000 倍だった（第 62 回）。天井と呼ぶ。
    p.lines.append(f"{rep['n_ops']} numeric ops・増幅 op={rep['amplifiers']}・"
                   f"モデル発散(許容の天井)≈{rep['model_divergence']:.2e}"
                   + (f"（dominant={rep['dominant']}）" if rep.get("dominant") else ""))
    if rep.get("cond_measured"):
        p.lines.append(f"  sample 実測: scale={rep['sample_scale']:.3g}"
                       "（増幅 op の cond をデータから測定済み・静的下界を解消）")
    else:
        p.lines.append("  cond=1 は *下界*（sample 未指定）——実データで cond を実測すべき")
    if rep.get("requires_noise_floor"):
        p.max_risk = max(p.max_risk, Risk.WARN)
        p.lines.append(f"  [WARN] 非決定 op {rep['nondeterministic_ops']} → run-to-run "
                       "ノイズ床の実測が必須（静的許容では不十分）")
    if rep.get("has_dynamic_shapes"):
        p.max_risk = max(p.max_risk, Risk.WARN)
        p.lines.append("  [WARN] dynamic shape 検出 → 形状ごとにカーネルが特化される。"
                       "1 形状の認証は他形状に転用できない（per-shape 再検証が必要）")
    if rep.get("has_normalization"):
        p.lines.append("  正規化層あり: RMSNorm は scale 中立・LayerNorm は平均優勢入力で"
                       " amp≈RMS/σ に増幅（sample 指定時は実測 cond に反映済み）")
    ad.phases.append(p)

    # --- 実測: CPU 上で 2 ベンダーを模倣する（第 62 回） ---
    # 第 61 回までこの経路の唯一の数値は静的伝播の `model_divergence` だった。実測と
    # 突き合わせたところ **最悪クラスの 200 倍・典型の 1000 倍以上** 大きく、これは
    # 「許容の天井」であって「予測」ではなかった。天井から導いた flip 上界は真だが
    # 無情報で、楔ユーザーには毎回「フリップ率 ≤ 40〜80%」と出る——偽BLOCK が常態化
    # すると、偽OK と同じく判定が信号を失う。降下した IR には既に実行可能な意味
    # （`interp.evaluate`）があるので、`dot` をベンダー模倣に差し替えて **実測** する。
    sim = None
    sim_note = ""
    if sample is not None:
        try:
            from .simulate import simulate_cross_vendor
            sim = simulate_cross_vendor(gm, sample)
            if sim is None:
                sim_note = ("降下が partial か重みを束縛できないため模倣できない"
                            "（0 で埋めず、天井のみで報告する）")
        except Exception as exc:            # noqa: BLE001 - 模倣の失敗で監査を落とさない
            sim_note = f"模倣に失敗: {type(exc).__name__}: {exc}"[:160]
    else:
        sim_note = "sample（代表入力）未指定 → 模倣は走らない"

    ceiling = rep.get("model_divergence")
    if sim is not None:
        sp = AuditPhase("simulation CPU 2 ベンダー模倣", "decided", Risk.INFO)
        sp.lines += sim.to_lines()
        w = sim.worst
        if w is not None and w.rel_divergence > 0 and ceiling:
            ratio = ceiling / w.rel_divergence
            sp.lines.append(f"  静的天井 δ={ceiling:.2e} は最悪クラス実測"
                            f"（{w.name}: δ={w.rel_divergence:.2e}）の ×{ratio:.0f}")
            if ratio >= 10.0:
                sp.lines.append("  → 天井は *無情報*: 許容の上限であって予測ではない。"
                                "判定は下の実測フリップで行う")
        ad.phases.append(sp)
    else:
        sp = AuditPhase("simulation CPU 2 ベンダー模倣", "pending", Risk.INFO)
        sp.lines.append(sim_note)
        sp.lines.append("代表入力 sample= を渡すと、同じ IR を 2 ベンダーの matmul 意味論で"
                        "走らせ、天井でなく実測の発散とフリップ率が出る。")
        ad.phases.append(sp)

    # タスク影響: 利用者の予算だけが BLOCK の根拠（閾値を発明しない）
    sim_worst = sim.worst if sim is not None else None
    if sim_worst is not None and sim_worst.n:
        from tsugi.rollout import samples_for_flip_budget

        # BLOCK は **観測されたフリップ** が予算を超えたときだけ。上界が超えるのは
        # 「標本が足りない」であって「壊れている」ではない——それを BLOCK にすると
        # n=256 のモデルは全部 BLOCK になり、また信号を失う。両者を別の深刻度で言う。
        over = sim_worst.flip_rate > flip_budget
        underpowered = (not over) and sim_worst.flip_rate_ub > flip_budget
        dp = AuditPhase("decision タスク影響(実測)", "decided",
                        Risk.BLOCK if over else (Risk.WARN if underpowered else Risk.INFO))
        dp.lines.append(
            f"実測フリップ率 {sim_worst.flip_rate * 100:.3f}%（上界 "
            f"{sim_worst.flip_rate_ub * 100:.3f}%・最悪クラス {sim_worst.name}・"
            f"n={sim_worst.n}）／予算 {flip_budget * 100:.3f}%")
        if rep.get("task_flip_bound") is not None:
            dp.lines.append(
                f"  参考: 静的天井由来の上界 {rep['task_flip_bound'] * 100:.2f}% は"
                "*許容の天井* であって予測ではない（判定には使わない）")
        if over:
            dp.lines.append("  予算超過（実測）→ ベンダー間でユーザーに見える判断が変わる")
        elif underpowered:
            need = samples_for_flip_budget(flip_budget)
            dp.lines.append(
                f"  [WARN] 観測は予算内だが n={sim_worst.n} では上界が予算を超える"
                f"——この予算を統計的に主張するには 0 フリップで n≥{need} 要る"
                "（標本不足であって不合格ではない）")
        dp.lines.append("  模倣は既知の発散クラスのみ＝実機発散の *下界*。"
                        "実機照合（audit_cross_vendor）で更新すること。")
        ad.phases.append(dp)
    elif rep.get("task_flip_bound") is not None:
        bound = rep["task_flip_bound"]
        over = bound > flip_budget
        dp = AuditPhase("decision タスク影響(天井)", "decided",
                        Risk.BLOCK if over else Risk.INFO)
        dp.lines.append(f"判断フリップ率 ≤ {bound * 100:.2f}%（予算 {flip_budget * 100:.2f}%）"
                        "——静的伝播の *許容の天井* からの上界。予測ではない")
        dp.lines.append("  実測は sample= を渡すと得られる（天井は実測の 100〜1000 倍に"
                        "なりうるので、この行だけで出荷判断をしないこと）")
        if over:
            dp.lines.append("  予算超過 → このモデルはベンダー間でユーザーに見える判断が変わりうる")
        ad.phases.append(dp)

    # --- codegen: 楔ユーザーを機械語まで届かせる（第 60 回） ---
    # ここが無い間、「単一ソースで両ベンダー」という看板の約束は tile-DSL を書く人に
    # しか届いていなかった。FX を IR へ降下し、tile-DSL 経路と同じ codegen 検証にかける。
    try:
        from tsugi.audit import _codegen_phase

        from .fxlower import fx_to_ir
        lm = fx_to_ir(gm)
        cgp = _codegen_phase(lm.module, ("nvidia", "amd_cdna", "amd_rdna"))
        cgp.name = "codegen 生成物（FX → IR → 実機械語）"
        cgp.lines = lm.report.to_lines() + cgp.lines
        if lm.report.partial:
            # 表せない op があるなら生成物はモデル全体ではない。「生成できた」を
            # 「モデルを生成できた」と読ませないため判定に載せる（偽OK 防止）。
            cgp.max_risk = max(cgp.max_risk, Risk.WARN)
        ad.phases.append(cgp)
    except Exception as exc:            # noqa: BLE001 - 降下失敗で監査全体を落とさない
        p2 = AuditPhase("codegen 生成物（FX → IR → 実機械語）", "decided", Risk.WARN)
        p2.lines.append(f"降下に失敗: {type(exc).__name__}: {exc}"[:200])
        ad.phases.append(p2)

    rt = AuditPhase("runtime クロスベンダー照合", "pending", Risk.INFO)
    rt.lines += [
        "静的 FX だけでは *等価性は認証できない*（第2ベンダーの実出力が無い）。",
        "実機/実データが揃ったら audit_runtime(out_a, out_b, K, logits_a=…, logits_b=…) へ。",
        "実機ノイズ床とクロス発散は audit_cross_vendor(run_a, run_b, K) で実測する。",
    ]
    ad.phases.append(rt)
    # 楔ユーザーの入口も同じ規律で被覆範囲を述べる（片方だけ開示すると片肺になる）。
    from tsugi.audit import _coverage_phase
    ad.phases.append(_coverage_phase(ad.phases))
    ad.stamp()
    return ad
