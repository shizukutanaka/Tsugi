"""FX→Tsugi 検証橋のテスト（SOCRATIC Q23/Q26）。

torch を使わず duck-typed の stand-in FX グラフで aten op→論理 op の写像と audit_fx を検証。
実 torch.fx との結線は torch 環境が要る（本環境では未検証・主張と実装の一致）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi_torch.fxbridge import audit_fx, fx_to_graph_ops  # noqa: E402


class _TM:
    def __init__(self, shape):
        self.shape = shape


class _Node:
    def __init__(self, op, target, shape=None):
        self.op = op
        self.target = target
        self.meta = {"tensor_meta": _TM(shape)} if shape else {}


class _Graph:
    def __init__(self, nodes):
        self.nodes = nodes


class _GM:
    """torch.fx.GraphModule の最小 duck-type（.graph.nodes）。"""
    def __init__(self, nodes):
        self.graph = _Graph(nodes)


def _transformer_block():
    # 代表的な attention+MLP ブロックの aten 列を模す
    return _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),       # qkv proj → matmul
        _Node("call_function", "aten.bmm.default", (8, 64)),          # attn scores → matmul
        _Node("call_function", "aten._softmax.default"),              # softmax（増幅）
        _Node("call_function", "aten.bmm.default", (8, 64)),          # attn·V → matmul
        _Node("call_function", "aten.mean.dim"),                      # layernorm 統計 → reduce
        _Node("call_function", "aten.mul.Tensor"),                    # scale（elementwise）
        _Node("call_function", "aten.addmm.default", (8, 2048)),      # MLP → matmul
        _Node("call_function", "aten.gelu.default"),                  # activation → scale
        _Node("get_attr", "weight"),                                  # 除外される
        _Node("output", "output"),                                    # 除外される
    ])


def test_fx_maps_aten_to_logical_ops():
    ops = fx_to_graph_ops(_transformer_block())
    kinds = [o.kind for o in ops]
    assert kinds == ["matmul", "matmul", "softmax", "matmul", "reduce", "scale",
                     "matmul", "scale"]
    # placeholder/get_attr/output は数値 op でないので除外
    assert all(k in ("matmul", "softmax", "reduce", "scale") for k in kinds)


def test_fx_matmul_K_from_shape_meta():
    ops = fx_to_graph_ops(_transformer_block())
    matmuls = [o for o in ops if o.kind == "matmul"]
    assert matmuls[0].K == 512        # addmm 出力末尾次元
    assert matmuls[-1].K == 2048      # MLP 出力末尾次元


def test_audit_fx_surfaces_amplifiers():
    rep = audit_fx(_transformer_block())
    assert rep["n_ops"] == 8
    assert "softmax" in rep["amplifiers"] and "reduce" in rep["amplifiers"]
    assert rep["model_divergence"] > 0.0


def test_audit_fx_empty_graph():
    rep = audit_fx(_GM([_Node("placeholder", "x"), _Node("output", "output")]))
    assert rep["n_ops"] == 0
    assert rep["model_divergence"] == 0.0


def test_audit_fx_translates_to_task_flip_bound():
    # ref_logits を渡すと静的グラフ発散 → タスク判断フリップ率上界に翻訳（静的→タスク）
    import numpy as np
    rep_none = audit_fx(_transformer_block())
    assert rep_none["task_flip_bound"] is None       # logit 無しなら None
    logits = np.random.default_rng(0).standard_normal((1000, 256)).astype(np.float32)
    rep = audit_fx(_transformer_block(), ref_logits=logits)
    assert rep["task_flip_bound"] is not None
    assert 0.0 <= rep["task_flip_bound"] <= 1.0      # 確率（上界）


class _SymInt:
    """torch.SymInt の最小 duck-type — int() で失敗する symbolic 次元。

    torch.compile(dynamic=True) や torch.export で生成される symbolic 次元を模す。
    int() が TypeError を送出することで _node_is_symbolic が dynamic を判定できる。
    """
    def __repr__(self) -> str:
        return "s0"

    def __int__(self) -> int:
        raise TypeError("symbolic SymInt — cannot convert to int")


def test_audit_fx_ref_scale_from_logits():
    """ref_logits を渡すと ref_scale（RMS）が audit 出力に含まれる（Q14: scale 推定）。

    certify_from_sample(x, K, dtype) に渡す scale ヒントになる。
    scale=1 仮定でモデルを認証すると、実 logit scale（例: 数十）との乖離で
    check_tensor が scale 超過 BLOCK を誤発火する。ref_scale はその乖離を事前に示す。
    """
    import numpy as np

    gm = _transformer_block()

    # ref_logits なしでは ref_scale は含まれない
    rep_no_logits = audit_fx(gm)
    assert "ref_scale" not in rep_no_logits, "logits 無しで ref_scale が含まれている"

    # scale ≈ 10 の logit（LLM の未正規化出力に近い）
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((500, 128)).astype(np.float32) * 10.0
    rep = audit_fx(gm, ref_logits=logits)
    assert "ref_scale" in rep, "logits 有りなのに ref_scale が含まれていない"
    actual_rms = float(np.sqrt(np.mean(logits ** 2)))
    assert abs(rep["ref_scale"] - actual_rms) / actual_rms < 0.01, \
        f"ref_scale={rep['ref_scale']:.2f} が実 RMS={actual_rms:.2f} と乖離"
    assert rep["task_flip_bound"] is not None, "ref_logits 有りなのに task_flip_bound=None"


def test_audit_fx_warns_dynamic_shapes():
    """dynamic=True コンパイルの symbolic shape を検出し has_dynamic_shapes=True を返す。

    torch.compile shape guard の研究知見（2025）:
    - shape guard は形状ごとにカーネルを特化する（タイル幅・縮約順序・アキュムレータ幅）
    - 特化カーネルの数値特性は形状依存 → 1 形状で認証した等価性は他形状に転用不可
    - has_dynamic_shapes=True は「per-shape 再検証が必要」のシグナル
    """
    dynamic_gm = _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (_SymInt(), _SymInt())),
        _Node("call_function", "aten._softmax.default"),
        _Node("output", "output"),
    ])
    rep = audit_fx(dynamic_gm)
    assert rep["has_dynamic_shapes"], "symbolic shape を dynamic と判定できていない"

    # 静的形状グラフ（int 次元）は dynamic でない
    rep_static = audit_fx(_transformer_block())
    assert not rep_static["has_dynamic_shapes"], "静的形状グラフが dynamic と誤判定された"

    # 形状なしノード（meta 無し）は dynamic 扱いしない
    no_meta_gm = _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default"),   # shape=None → meta なし
        _Node("output", "output"),
    ])
    assert not audit_fx(no_meta_gm)["has_dynamic_shapes"], \
        "shape meta なし → dynamic と誤判定（既定の int 扱いが期待値）"


def test_audit_fx_detects_normalization_layers():
    """LayerNorm/RMSNorm 系の op を検出し has_normalization=True を返す（FEATURE-AUDIT.md A-5）。

    scale-invariant（LN(c·x)≈LN(x)）は「相対発散を増幅しない」を意味しない——A-5 の
    数値実験で当初の想定が反転した: RMSNorm は無条件安定（amp≤1・実測 1.0021）だが、
    LayerNorm は平均優勢入力で amp≈RMS/σ に *増幅* する（shift=10 で実測 10.10）。
    has_normalization は両者を含む可視化フラグで、kind 自体は propagation 側で
    layer_norm / rms_norm に分かれる（増幅則が違うため）。
    """
    norm_gm = _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),
        _Node("call_function", "aten.native_layer_norm.default"),
        _Node("output", "output"),
    ])
    assert audit_fx(norm_gm)["has_normalization"]

    rms_gm = _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),
        _Node("call_function", "aten._rms_norm.default"),
        _Node("output", "output"),
    ])
    assert audit_fx(rms_gm)["has_normalization"]

    # 正規化層が無いグラフは False のまま（既存の _transformer_block は norm を含まない）
    rep_no_norm = audit_fx(_transformer_block())
    assert not rep_no_norm["has_normalization"]


def test_audit_fx_flags_nondeterministic_atomic_ops():
    # scatter_add 等 atomicAdd 由来の非決定 op を検出し noise floor 実測必須を宣言
    # （PyTorch 公式: https://pytorch.org/docs/stable/notes/randomness.html）
    gm = _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),
        _Node("call_function", "aten.scatter_add.default"),    # 非決定（forward atomicAdd）
        _Node("call_function", "aten._softmax.default"),
        _Node("output", "output"),
    ])
    rep = audit_fx(gm)
    assert rep["requires_noise_floor"], "scatter_add を含むのに noise floor 不要扱い"
    assert any("scatter_add" in n for n in rep["nondeterministic_ops"])

    # 決定論的グラフ（matmul/softmax のみ）は noise floor 不要
    det = audit_fx(_transformer_block())
    assert not det["requires_noise_floor"]
    assert det["nondeterministic_ops"] == []


def test_audit_fx_sample_replaces_scale1_cond1_assumptions():
    """audit_fx(sample=) が torch 経路の「scale=1 / cond=1」暗黙仮定を実測で置き換える。

    FEATURE-AUDIT.md A-3: `audit()` は B-1a/B-1b で代表サンプルからの scale 実測・
    empirical_cond を持っていたが、**製品の想定入口である torch 経路には無かった**。
    実 LLM の活性は massive activations（一部チャネルが中央値の ~1000 倍・
    docs/SOURCES.md）を持つため、scale=1 仮定は認証 atol を桁で誤らせる。
    """
    gm = _transformer_block()

    base = audit_fx(gm)                       # 従来経路（sample 無し）
    assert "sample_scale" not in base, "sample 未指定で実測フィールドが出ている（後方互換の破壊）"

    # massive activation を模した外れチャネル（中央値の ~1000 倍）を含む代表入力
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 512)).astype(np.float32) * 0.1
    x[:, 7] *= 1000.0
    rep = audit_fx(gm, sample=x)

    # 実 RMS が測られる（scale=1 仮定を脱する）
    assert rep["sample_scale"] > 1.0, rep["sample_scale"]
    # 外れチャネルが検出される（単一 scale 仮定の破綻を可視化）
    assert rep["channel_spread"] > 100.0, rep["channel_spread"]
    # 増幅 op の cond が実測され、発散が静的下界から更新される
    assert rep["cond_measured"] is True
    assert rep["model_divergence"] != base["model_divergence"]
    # 静的 cond=1 は *下界* ゆえ実測後は増える（過小評価の是正・fail-safe 方向）
    assert rep["model_divergence"] > base["model_divergence"]


def test_fx_maps_norm_ops_to_dedicated_kinds():
    """正規化層は reduce でなく専用 kind に写る（A-5）。

    旧写像は LayerNorm/RMSNorm を "reduce" に写し、reduce の cond 統計 Σ|x|/|Σx|
    （符号相殺）を当てていた。これは正規化に不適切で *両方向* に誤る——零平均 sample
    では相殺で爆発（偽BLOCK）、平均優勢 sample では ≈1 なのに実 LayerNorm は
    RMS/σ 倍に増幅（偽OK）。mean/sum/var は従来通り reduce のままであることも固定する
    （正規化の *統計* op と正規化そのものは別物）。
    """
    def _norm_gm(target):
        return _GM([
            _Node("placeholder", "x"),
            _Node("call_function", "aten.addmm.default", (8, 512)),
            _Node("call_function", target),
            _Node("output", "output"),
        ])

    assert [o.kind for o in fx_to_graph_ops(_norm_gm("aten.native_layer_norm.default"))] \
        == ["matmul", "layer_norm"]
    # rms_norm は "_norm" を含むので判定順序が本質（rms が先でないと layer_norm に落ちる）
    assert [o.kind for o in fx_to_graph_ops(_norm_gm("aten._rms_norm.default"))] \
        == ["matmul", "rms_norm"]
    # group/batch norm も平均減算を含むので保守的に layer_norm 扱い
    assert [o.kind for o in fx_to_graph_ops(_norm_gm("aten.native_group_norm.default"))] \
        == ["matmul", "layer_norm"]
    # 正規化の統計 op は reduce のまま（回帰なし）
    assert [o.kind for o in fx_to_graph_ops(_norm_gm("aten.mean.dim"))] == ["matmul", "reduce"]
    assert [o.kind for o in fx_to_graph_ops(_norm_gm("aten.sum.default"))] == ["matmul", "reduce"]
    # has_normalization は 3 種の正規化すべてで True（可視化は据置）
    for t in ("aten.native_layer_norm.default", "aten._rms_norm.default",
              "aten.native_group_norm.default"):
        assert audit_fx(_norm_gm(t))["has_normalization"]


def test_audit_fx_layer_norm_cond_fires_on_mean_dominated_sample():
    """平均優勢 sample では LayerNorm の実測 cond が model_divergence を引き上げる（A-5）。

    両方向の修正を 1 つのテストで実証する: 平均優勢（μ/RMS→1）では増幅が発火して
    予測が上がり（旧 cond≈1 は偽OK だった）、零平均では発火せず静的値に近いまま
    （旧 reduce 統計 Σ|x|/|Σx| は零平均で ~27 に爆発し偽BLOCK だった）。
    """
    gm = _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),
        _Node("call_function", "aten.native_layer_norm.default"),
        _Node("output", "output"),
    ])
    static = audit_fx(gm)["model_divergence"]
    rng = np.random.default_rng(0)

    mean_dominated = rng.standard_normal((32, 512)) * 0.1 + 5.0
    hot = audit_fx(gm, sample=mean_dominated)
    assert hot["cond_measured"] is True
    assert "layer_norm" in hot["amplifiers"]
    assert hot["model_divergence"] > static * 3, \
        f"平均優勢入力で増幅が発火しない（偽OK）: {hot['model_divergence']:.2e} vs {static:.2e}"

    zero_mean = rng.standard_normal((32, 512))
    cool = audit_fx(gm, sample=zero_mean)
    assert cool["model_divergence"] < static * 2, \
        f"零平均入力で幻の増幅が出た（偽BLOCK）: {cool['model_divergence']:.2e} vs {static:.2e}"


def main() -> int:
    ok = True
    tests = [
        test_fx_maps_aten_to_logical_ops,
        test_fx_matmul_K_from_shape_meta,
        test_audit_fx_surfaces_amplifiers,
        test_audit_fx_empty_graph,
        test_audit_fx_translates_to_task_flip_bound,
        test_audit_fx_ref_scale_from_logits,
        test_audit_fx_warns_dynamic_shapes,
        test_audit_fx_detects_normalization_layers,
        test_fx_maps_norm_ops_to_dedicated_kinds,
        test_audit_fx_layer_norm_cond_fires_on_mean_dominated_sample,
        test_audit_fx_flags_nondeterministic_atomic_ops,
        test_audit_fx_sample_replaces_scale1_cond1_assumptions,
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
