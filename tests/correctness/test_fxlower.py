"""FX → Tsugi IR 降下のテスト——**想定ユーザーの経路が機械語まで届くか**。

2 段構え:
  1. stand-in FX グラフ（torch 不要・常に実行）— 構造と正直さの契約
  2. **実 torch.fx**（torch があれば実行・無ければ正直に skip）— 実 FX との結線と、
     降下した IR を NumPy で評価して **torch eager と意味論を照合**する

(2) が本スイートの核心。分解（softmax/layer_norm/gelu…）が正しいかは、
アセンブラにも LLVM にも問えない——**実行して突き合わせる以外に方法がない**。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi import codegen as cg  # noqa: E402
from tsugi.interp import evaluate  # noqa: E402
from tsugi_torch.fxbridge import audit_fx, fx_to_graph_ops  # noqa: E402
from tsugi_torch.fxlower import fx_to_ir  # noqa: E402

try:
    import torch
    import torch.fx
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


# --- stand-in FX グラフ（torch 非依存・duck-typed） -------------------------

@dataclass(eq=False)
class N:
    op: str
    target: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


@dataclass
class G:
    nodes: list


class GM:
    def __init__(self, ns):
        self.graph = G(ns)


def _standin(*targets: str) -> GM:
    return GM([N("placeholder", "x"),
               *[N("call_function", t) for t in targets],
               N("output", "output")])


def test_lowering_produces_ir_that_both_vendors_assemble():
    """FX → IR → 実機械語。楔ユーザーが tile-DSL 経路と同じものを受け取る。"""
    lm = fx_to_ir(_standin("aten.addmm.default", "aten.native_layer_norm.default",
                           "aten._softmax.default"))
    assert lm.module.op_kinds(), "IR が空"
    assert not lm.report.partial, lm.report.unsupported
    for t in cg.TARGETS:
        em, asm = cg.verify_codegen(lm.module, target=t)
        if cg.toolchain(t) is None:
            assert asm.ok is None
            continue
        assert asm.ok is True, f"{t}: {asm.stderr[:300]}"
        assert cg.verify_encoding(lm.module, target=t).ok is True


def test_unrepresentable_ops_make_the_module_partial_not_silently_dropped():
    """表せない op を黙って落とすと偽OK になる。partial と言えること。"""
    lm = fx_to_ir(_standin("aten.addmm.default", "aten._log_softmax.default",
                           "aten.special_bessel_j0.default"))
    assert lm.report.partial
    assert any("log_softmax" in u for u in lm.report.unsupported)
    assert any("bessel" in u for u in lm.report.unsupported)
    txt = "\n".join(lm.report.to_lines())
    assert "partial" in txt and "このモデルを計算しない" in txt
    assert "未対応の演算" in txt          # 構造 op と区別して理由を言う


def test_exact_gelu_is_refused_rather_than_approximated():
    """erf 版 gelu を tanh 近似で黙って置換しない（実測 max|Δ|≈4.6e-4）。

    等価性を検証する道具が、検証対象を別の関数に差し替えるのは偽OK そのもの。
    tanh 版と明示されているときだけ分解する。
    """
    exact = fx_to_ir(_standin("aten.gelu.default"))
    assert exact.report.partial, "erf 版 gelu が黙って通ってしまった"
    assert any("erf" in u for u in exact.report.unsupported)
    tanh = fx_to_ir(GM([N("placeholder", "x"),
                        N("call_function", "aten.gelu.default",
                          kwargs={"approximate": "tanh"}),
                        N("output", "output")]))
    assert not tanh.report.partial
    assert any("tanh" in d for d in tanh.report.decomposed)


def test_shape_only_ops_are_recorded_as_ignored():
    """view/permute はデータ移動を伴うが本 IR は形状を持たない。黙って通さない。"""
    lm = fx_to_ir(_standin("aten.view.default", "aten.permute.default",
                           "aten.mul.Tensor"))
    assert lm.report.shape_only
    assert "形状のみ" in "\n".join(lm.report.to_lines())


def test_unspecified_eps_is_declared_as_an_assumption():
    """eps を静的に決められないときは仮定したと言う（黙って数値を選ばない）。"""
    lm = fx_to_ir(_standin("aten.native_layer_norm.default"))
    assert lm.report.assumptions
    assert any("eps" in a for a in lm.report.assumptions)
    # 明示されていれば仮定は立たない
    given = fx_to_ir(GM([N("placeholder", "x"),
                         N("call_function", "aten.native_layer_norm.default",
                           kwargs={"eps": 1e-6}),
                         N("output", "output")]))
    assert not given.report.assumptions


# --- 実 torch.fx（あれば） --------------------------------------------------

def test_call_module_targets_resolve_or_the_whole_audit_is_a_false_ok():
    """`call_module` の target は "0"/"1" という経路名で op の種類を表さない。

    解決しないと `_kind_of` が全ノードで None を返し、`audit_fx` は
    「0 numeric ops・発散 0」を報告する——**想定ユーザーのモデルが必ず無害判定になる**。
    実際にそうなっていた（stand-in は aten 名を使うため露見しなかった）。回帰固定。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 実 FX に対する結線は未検証（正直に skip）")
        return
    m = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.Softmax(-1))
    gm = torch.fx.symbolic_trace(m)
    assert all(n.op != "call_function" for n in gm.graph.nodes), \
        "この回帰テストは call_module 経路を突く前提"
    rep = audit_fx(gm)
    assert rep["n_ops"] >= 3, f"実 FX で {rep['n_ops']} ops しか見えていない（偽OK）"
    assert rep["model_divergence"] > 0.0
    assert rep["has_normalization"] is True
    assert "softmax" in rep["amplifiers"] and "layer_norm" in rep["amplifiers"]
    assert len(fx_to_graph_ops(gm)) >= 3


def test_lowered_ir_means_the_same_thing_as_the_model():
    """降下した IR を評価し **torch eager と突き合わせる**（意味論の照合）。

    分解が正しいかはアセンブラにも LLVM にも問えない。実行して比べるしかない。
    この照合で実際に 2 件の欠陥が出た: layer_norm の eps 欠落（max|Δ|≈1.9e-5）と、
    erf 版 gelu の無断置換（max|Δ|≈4.6e-4）。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 意味論照合は未実施（正直に skip）")
        return
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=torch.float64)
    cases = {
        "Softmax": nn.Softmax(-1),
        "LayerNorm": nn.LayerNorm(16, elementwise_affine=False),
        "RMSNorm": nn.RMSNorm(16, eps=1e-6, elementwise_affine=False),
        "GELU(tanh)": nn.GELU(approximate="tanh"),
        "SiLU": nn.SiLU(),
        "Tanh": nn.Tanh(),
        "Sigmoid": nn.Sigmoid(),
        "ReLU": nn.ReLU(),
    }
    for name, mod in cases.items():
        mod = mod.double()
        ref = mod(x).detach().numpy()
        lm = fx_to_ir(torch.fx.symbolic_trace(mod))
        assert not lm.report.partial, f"{name}: {lm.report.unsupported}"
        got = evaluate(lm.module, [x.numpy()])[-1]
        err = float(np.max(np.abs(got - ref)))
        assert err < 1e-9, f"{name}: 意味論が一致しない max|Δ|={err:.3e}"


def _bind(m, lm, x):
    """束縛記述子から実テンソルを引く（生成カーネルの引数の意味に従う）。"""
    params = dict(m.named_parameters())
    return {d: (x.numpy() if d.startswith("input:")
                else params[d.split(":", 1)[1]].detach().numpy())
            for d in lm.report.bindings}


def test_matmul_paths_with_real_weights_match_eager():
    """**最重要 op（matmul）を重み込みで意味論照合する**。

    降下した IR は `load` に束縛記述子（`input:x` / `param:0.weight`）を持つので、
    重みがグラフ外にあるモデルでも評価できる。束縛を明示して初めて 2 件の欠陥が出た:
      - `linear(x,W,b) = x·Wᵀ + b` の転置落ち（`x·W` になっていた）
      - `call_function` 経路で bias が落ちていた（max|Δ|≈3.8e-01）
    どちらもアセンブルは通る。意味論でしか捕まらない。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 重み込みの意味論照合は未実施（正直に skip）")
        return
    torch.manual_seed(0)
    x = torch.randn(3, 6, dtype=torch.float64)
    cases = {
        "Linear": nn.Linear(6, 4),
        "Linear(no bias)": nn.Linear(6, 4, bias=False),
        "MLP": nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 4)),
        "MLP+LN+SiLU": nn.Sequential(
            nn.Linear(6, 8), nn.LayerNorm(8, elementwise_affine=False),
            nn.SiLU(), nn.Linear(8, 4)),
        "MLP+GELU+Softmax": nn.Sequential(
            nn.Linear(6, 8), nn.GELU(approximate="tanh"),
            nn.Linear(8, 4), nn.Softmax(-1)),
    }
    for name, mod in cases.items():
        mod = mod.double()
        lm = fx_to_ir(torch.fx.symbolic_trace(mod))
        assert not lm.report.partial, f"{name}: {lm.report.unsupported}"
        assert lm.report.bindings, f"{name}: 束縛が記録されていない"
        got = evaluate(lm.module, _bind(mod, lm, x))[-1]
        ref = mod(x).detach().numpy()
        err = float(np.max(np.abs(got - ref)))
        assert err < 1e-9, f"{name}: 意味論が一致しない max|Δ|={err:.3e}"


def test_missing_binding_raises_rather_than_silently_reusing_a_tensor():
    """束縛が足りなければ **落ちる**（黙って別のテンソルを使わない）。"""
    lm = fx_to_ir(_standin("aten.native_layer_norm.default"))
    try:
        evaluate(lm.module, {"input:nonexistent": np.zeros((2, 2))})
    except KeyError as e:
        assert "束縛" in str(e)
    else:
        raise AssertionError("束縛が無いのに評価が通ってしまった")


def test_real_model_reaches_machine_code_through_the_product_facade():
    """`tsugi.verify(gm)` が実 nn.Module から実機械語の検証まで一気に届く。"""
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: facade の実 FX 経路は未検証（正直に skip）")
        return
    import tsugi
    m = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16),
                      nn.GELU(approximate="tanh"), nn.Softmax(-1))
    ad = tsugi.verify(torch.fx.symbolic_trace(m))
    names = [p.name for p in ad.phases]
    assert any(n.startswith("codegen") for n in names), names
    txt = ad.to_text()
    if cg.toolchain("amd_cdna") is not None:
        assert "L2-アセンブル検証済み" in txt
    assert "L3" in txt and "常に空" in txt          # 実行は未検証と言い続ける
    assert ad.exit_code in (0, 1, 2)


def test_a_realistic_transformer_block_is_handled_honestly():
    """トイの Sequential でなく **実際のアテンションブロック**で確かめる。

    `chunk`/`getitem` は本 IR に対応物が無い。値をそのまま通せば q/k/v が同一値の
    別名になり、生成物はモデルを**まったく計算しない**——だから shape_only ではなく
    「表せない」に倒す（fail-safe）。partial の文言も「大部分は動く」と読まれないよう
    「このモデルを計算しない」と書く。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 実モデルでの降下は未検証（正直に skip）")
        return

    class Block(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.d = d
            self.qkv = nn.Linear(d, 3 * d)
            self.o = nn.Linear(d, d)
            self.n1 = nn.LayerNorm(d)
            self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d),
                                    nn.GELU(approximate="tanh"),
                                    nn.Linear(4 * d, d))

        def forward(self, x):
            q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
            a = torch.softmax(q @ k.transpose(-2, -1) / (self.d ** 0.5), -1)
            x = x + self.o(a @ v)
            return x + self.ff(self.n2(x))

    gm = torch.fx.symbolic_trace(Block())
    rep = audit_fx(gm)
    assert rep["n_ops"] >= 10, rep["n_ops"]
    assert rep["has_normalization"] and "softmax" in rep["amplifiers"]
    lm = fx_to_ir(gm)
    assert len(lm.report.covered) >= 10, lm.report.covered
    # chunk/getitem は「構造」として表せないと言う（黙って別名にしない）
    assert lm.report.partial
    assert any("構造" in u for u in lm.report.unsupported), lm.report.unsupported
    txt = "\n".join(lm.report.to_lines())
    assert "このモデルを計算しない" in txt
    # 降下できた命令列自体は 3 ターゲットとも成立する
    for t in cg.TARGETS:
        if cg.toolchain(t) is None:
            continue
        assert cg.verify_codegen(lm.module, target=t)[1].ok is True


def test_the_documented_product_entry_point_works_with_real_torch():
    """`torch.compile(model, backend="tsugi")` — README が掲げる実際の入口。

    これまで実 torch で一度も走らせていなかった。想定ユーザーが最初に打つコマンドが
    動くことと、警告が **今の実態**（codegen は L2 まで・実行は eager 素通し）を
    述べることを固定する。"no codegen yet" は自分の製品についての虚偽になった。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 実 torch.compile 経路は未検証（正直に skip）")
        return
    import warnings

    import tsugi_torch
    tsugi_torch.register()
    m = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.Softmax(-1))
    x = torch.randn(4, 16)
    compiled = torch.compile(m, backend="tsugi")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = compiled(x)
    assert torch.allclose(out, m(x)), "eager 素通しの出力が変わっている"
    msgs = [str(x.message) for x in w if "[tsugi]" in str(x.message)]
    assert msgs, "検証警告が出ていない"
    msg = msgs[0]
    assert "no codegen yet" not in msg, "自分の製品について古い虚偽が残っている"
    assert "codegen:" in msg and "IR へ降下" in msg
    assert "実行は未検証" in msg           # 実行の未検証は言い続ける
    # 静的監査が実グラフを見えていること（0 ops の偽OK でない）
    assert "numeric ops" in msg and not msg.count("0 numeric ops")


def _sim_model():
    """束縛可能な（重みがグラフ外にある call_module のみの）実モデル。"""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64),
                      nn.GELU(approximate="tanh"), nn.Linear(64, 32))
    return m, torch.fx.symbolic_trace(m), torch.randn(256, 64)


def test_static_divergence_is_a_ceiling_not_a_prediction():
    """静的 `model_divergence` は **許容の天井** であって予測ではない（第 62 回）。

    第 61 回までこの経路の唯一の数値は静的伝播だった。同じモデルを CPU で 2 ベンダー
    として走らせて実測すると、静的値は最悪クラスの実測より桁違いに大きい。原因は
    構造的で、伝播モデルは格納 dtype `u(fp16)` を発散単位にするが、両ベンダーが f32 で
    累積するなら跨ベンダー差は `u(f32)` スケール（2¹³ 倍小さい）。この乖離を固定して、
    将来この値を「予測」として提示する退行を止める。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 天井と実測の突き合わせは未検証（正直に skip）")
        return
    from tsugi_torch.fxbridge import audit_fx
    from tsugi_torch.simulate import simulate_cross_vendor

    _, gm, x = _sim_model()
    rep = audit_fx(gm, sample=x.numpy())
    sim = simulate_cross_vendor(gm, x.numpy())
    assert sim is not None, "束縛可能なグラフで模倣が None になっている"
    worst = sim.worst
    assert worst is not None and worst.rel_divergence > 0.0
    ratio = rep["model_divergence"] / worst.rel_divergence
    assert ratio > 10.0, (
        f"天井 {rep['model_divergence']:.2e} が実測 {worst.rel_divergence:.2e} に "
        f"接近している（×{ratio:.1f}）——伝播モデルが変わったなら文書の主張も見直すこと")
    # 模倣は既知クラスのみ＝下界であることを黙らない
    assert any("下界" in ln for ln in sim.to_lines())


def test_gate_blocks_on_measured_flips_not_on_the_ceiling():
    """BLOCK の根拠は実測フリップであって天井ではない。

    天井由来の上界は真だが無情報（このモデルで 24%）。それで BLOCK を出し続けると
    偽BLOCK が常態化し、偽OK と同じく判定が信号を失う。実測が予算内なら BLOCK に
    しないこと、標本不足は BLOCK でなく WARN として **要求標本数つき** で言うことを固定。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 実測ゲートは未検証（正直に skip）")
        return
    from tsugi.report import Risk
    from tsugi_torch.fxbridge import audit_torch

    m, gm, x = _sim_model()
    ref = m(x).detach().numpy()
    ad = audit_torch(gm, ref_logits=ref, sample=x.numpy())
    names = [p.name for p in ad.phases]
    assert any(n.startswith("simulation") for n in names), "模倣フェーズが無い"
    dec = next(p for p in ad.phases if p.name.startswith("decision"))
    assert "(実測)" in dec.name, "判定が天井のままになっている"
    assert dec.max_risk is not Risk.BLOCK, "実測が予算内なのに BLOCK している"
    text = "\n".join(dec.lines)
    assert "天井" in text and "判定には使わない" in text
    assert "n≥" in text, "標本不足のとき必要標本数を示していない"

    # sample が無ければ天井で判定する（それしか無いので）。ただし天井と明示する。
    ad2 = audit_torch(gm, ref_logits=ref)
    dec2 = next(p for p in ad2.phases if p.name.startswith("decision"))
    assert "(天井)" in dec2.name and "予測ではない" in "\n".join(dec2.lines)


def test_simulation_refuses_graphs_it_cannot_bind():
    """表せない op があるグラフ、重みを引けないグラフでは **None**（0 で埋めない）。"""
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 模倣の拒否条件は未検証（正直に skip）")
        return
    from tsugi_torch.simulate import simulate_cross_vendor

    class Chunked(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x):
            a, b = self.fc(x).chunk(2, dim=-1)
            return torch.cat([b, a], dim=-1)

    gm = torch.fx.symbolic_trace(Chunked())
    assert simulate_cross_vendor(gm, torch.randn(4, 8).numpy()) is None


def test_the_representative_input_is_an_activation_not_a_weight():
    """`torch.compile` 経路の「実測」が **重み行列** を測っていた（第 62 回・修正済）。

    dynamo は重みを引数へ持ち上げるので `example_inputs[0]` は `nn.Parameter` である。
    それを代表入力として `audit_fx(sample=…)` に渡していたため、A-3 で導入した
    「sample 実測: scale=…」は活性でなく重みの統計を報告していた。実測と称して別の
    ものを測るのは、静的仮定を残すより悪い——利用者はそれを活性の実測だと読む。

    同じ根から模倣の束縛も壊れる: 持ち上げられた重みは `input:` 記述子になり
    `named_parameters()` は空になるので、代表入力 1 本を全記述子へ配ると重みの位置に
    活性が入る。**同じテンソルを複数の記述子に当てない**ことを固定する。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: dynamo の引数持ち上げは未検証（正直に skip）")
        return
    import torch._dynamo as dynamo

    from tsugi_torch import _activation_input, _sim_inputs
    from tsugi_torch.fxlower import fx_to_ir
    from tsugi_torch.simulate import refusal_reason, simulate_cross_vendor

    seen = []

    def spy(gm, ex):
        seen.append((gm, ex))
        return gm.forward

    dynamo.reset()
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.Softmax(-1))
    torch.compile(m, backend=spy)(torch.randn(300, 64))
    gm, ex = seen[0]

    # 前提の確認: 実際に重みが引数へ持ち上がっている（この事実が崩れたら本テストは無意味）
    assert any(isinstance(t, torch.nn.Parameter) for t in ex), "重みが持ち上がっていない"
    assert isinstance(ex[0], torch.nn.Parameter), "先頭が活性になった（前提の変化）"
    assert not list(gm.named_parameters()), "dynamo グラフに重みが属性として残っている"

    act = _activation_input(ex)
    assert act is not None and act.shape == (300, 64), "活性でなく重みを選んでいる"

    # 代表入力 1 本では束縛が一意に決まらない → 諦める（誤った実測を出さない）
    assert simulate_cross_vendor(gm, act) is None
    why = refusal_reason(gm, act)
    assert "input:" in why or "一意" in why, f"諦めた理由を述べていない: {why!r}"

    # 全引数を順に渡せば束縛できる。行を切るのは活性だけ（重みを切ると形が壊れる）
    ins = _sim_inputs(ex, max_rows=256)
    assert len(ins) == len(fx_to_ir(gm).report.bindings)
    assert ins[0].shape == (64, 64), "重みの行を切っている"
    sim = simulate_cross_vendor(gm, ins)
    assert sim is not None and sim.worst is not None
    assert sim.n_samples == 256, "標本数を入力側から数えている（重みの行数を拾う）"


def test_a_class_that_cannot_fire_is_named_not_reported_as_zero():
    """**偽OK を直す道具の中に偽OK があった**（第 62 回・自己問答で発覚）。

    当初 `tf32`/`rtz` を fp16 格納で回しており、両方とも「相対発散 0.00e+00」と出ていた。
    しかしこれは「差が無い」ではなく「**その差が表現できない**」——fp16 の仮数 10 bit は
    TF32 と同じなので丸めが恒等になり、`input_precision="ieee"` は丸めモードの指定ごと
    捨てられる。0 と表示すれば読み手は「TF32 起因の発散は無い」と読む。
    正しい格納（f32）で測ると `rtz` は f16 累積より大きい発散を示す。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 非適用クラスの扱いは未検証（正直に skip）")
        return
    from tsugi_torch.simulate import CLASS_REQUIRES, simulate_cross_vendor

    _, gm, x = _sim_model()
    full = simulate_cross_vendor(gm, x.numpy())
    assert full is not None
    by = {c.name: c for c in full.classes}
    # 正しい格納で測れば、前提つきクラスは実際に発散を示す（0 ではない）
    for name in CLASS_REQUIRES:
        c = by[name]
        assert c.applicable and c.rel_divergence > 0.0, (
            f"{name} が発現していない（格納 {c.storage}・{c.why}）")
    assert by["rtz"].rel_divergence > by["f16acc"].rel_divergence, (
        "丸めモード差が f16 累積より小さい——この主張が崩れたら文書も見直すこと")

    # 発現しない格納しか測らないなら「0」でなく「非適用」と名指しする
    f16_only = simulate_cross_vendor(gm, x.numpy(), storage=("float16",))
    assert f16_only is not None
    for c in f16_only.classes:
        if c.name in CLASS_REQUIRES:
            assert not c.applicable and c.why, f"{c.name} を 0 として報告している"
    assert all(c.applicable for c in f16_only.measured)
    txt = "\n".join(f16_only.to_lines())
    assert "非適用" in txt and "0.00e+00" not in txt

    # 最悪クラスは非適用を混ぜず、上界が並ぶときは相対発散で決着する
    w = full.worst
    assert w is not None and w.applicable
    assert w.rel_divergence == max(c.rel_divergence for c in full.measured)


def test_flip_semantics_follow_the_task_not_always_argmax():
    """argmax を非分類タスクに当てる**静かな誤用**を、新しい道具で踏み直さない。

    新視点11 が既に「argmax を非分類タスクに使うと flip_rate=0 に固まる」と記録して
    いるのに、模倣は `decision_flips`（argmax）を無条件に呼んでいた。同じモデル・
    同じ発散でも、分類の読みでは 0.0%、回帰の読みでは 94.8% になる——回帰モデルは
    「フリップ 0%」として出荷され、これは最も重い偽OK である。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: タスク意味論の分岐は未検証（正直に skip）")
        return
    from tsugi_torch.fxbridge import audit_torch
    from tsugi_torch.simulate import simulate_cross_vendor

    m, gm, x = _sim_model()
    xa = x.numpy()
    cls = simulate_cross_vendor(gm, xa)
    reg = simulate_cross_vendor(gm, xa, task="regression", task_kwargs={"rtol": 1e-4})
    assert cls is not None and reg is not None
    assert cls.worst.flip_rate == 0.0, "分類の読みが 0% でない（前提の変化）"
    assert reg.worst.flip_rate > cls.worst.flip_rate, (
        "回帰の読みが分類より小さい——タスク意味論が効いていない")
    assert reg.worst.task == "regression" and cls.worst.task == "classification"

    # 分類のときは argmax 前提であることを黙らない（誤用を検出できないので明示する）
    assert any("argmax" in ln for ln in cls.to_lines())
    assert not any("argmax（多クラス分類）前提" in ln for ln in reg.to_lines())

    # 製品入口も task を受け取り、判定行に何の意味論かを書く
    ref = m(x).detach().numpy()
    ad = audit_torch(gm, ref_logits=ref, sample=xa, task="regression",
                     task_kwargs={"rtol": 1e-4}, flip_budget=0.001)
    dec = next(p for p in ad.phases if p.name.startswith("decision"))
    assert "task=regression" in "\n".join(dec.lines)
    assert ad.exit_code == 2, "回帰の読みで予算を大きく超えたのに BLOCK していない"


def test_the_ceiling_is_only_a_ceiling_under_one_metric():
    """見出し数値それ自体への問答（第 62 回・追補4）。

    「静的値は実測の 200〜1700 倍」という主張は、実測を **スケール正規化**
    `max|Δ|/max|a|` で測ったときの比だった。`equivalence.compare` が使う正準の
    **要素ごと** `max(|Δ|/(|a|+1e-12))` で測ると、同じ差が静的値を **上回る**。
    つまり静的値は「あらゆる意味での上界」ではない。両方を報告し、どちらと
    比べているかを明記することを固定する。
    """
    if not HAVE_TORCH:
        print("  [SKIP] torch 無し: 尺度の取り違えは未検証（正直に skip）")
        return
    from tsugi_torch.fxbridge import audit_fx, audit_torch
    from tsugi_torch.simulate import simulate_cross_vendor

    m, gm, x = _sim_model()
    xa = x.numpy()
    ceiling = audit_fx(gm, sample=xa)["model_divergence"]
    sim = simulate_cross_vendor(gm, xa)
    assert sim is not None
    for c in sim.measured:
        # 2 尺度は別物であり、要素ごとの方が必ず大きい（分母が小さい要素に支配される）
        assert c.max_rel_elementwise >= c.rel_divergence, c.name
    exceed = [c.name for c in sim.measured if c.max_rel_elementwise > ceiling]
    assert exceed, "要素ごとの誤差が全クラスで天井以下——尺度の違いが消えている"
    assert sim.worst.rel_divergence < ceiling, "スケール正規化では天井を下回るはず"

    txt = "\n".join(sim.to_lines())
    assert "スケール正規化" in txt and "要素ごと" in txt
    ad = audit_torch(gm, ref_logits=m(x).detach().numpy(), sample=xa)
    sp = next(p for p in ad.phases if p.name.startswith("simulation"))
    body = "\n".join(sp.lines)
    assert "スケール正規化" in body
    assert "天井を上回る" in body, "天井が上界でない尺度があることを黙っている"


def main() -> int:
    ok = True
    for t in (test_lowering_produces_ir_that_both_vendors_assemble,
              test_unrepresentable_ops_make_the_module_partial_not_silently_dropped,
              test_exact_gelu_is_refused_rather_than_approximated,
              test_shape_only_ops_are_recorded_as_ignored,
              test_unspecified_eps_is_declared_as_an_assumption,
              test_call_module_targets_resolve_or_the_whole_audit_is_a_false_ok,
              test_lowered_ir_means_the_same_thing_as_the_model,
              test_matmul_paths_with_real_weights_match_eager,
              test_missing_binding_raises_rather_than_silently_reusing_a_tensor,
              test_real_model_reaches_machine_code_through_the_product_facade,
              test_a_realistic_transformer_block_is_handled_honestly,
              test_the_documented_product_entry_point_works_with_real_torch,
              test_static_divergence_is_a_ceiling_not_a_prediction,
              test_gate_blocks_on_measured_flips_not_on_the_ceiling,
              test_simulation_refuses_graphs_it_cannot_bind,
              test_the_representative_input_is_an_activation_not_a_weight,
              test_a_class_that_cannot_fire_is_named_not_reported_as_zero,
              test_flip_semantics_follow_the_task_not_always_argmax,
              test_the_ceiling_is_only_a_ceiling_under_one_metric):
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    print(f"  torch: {'あり（実 FX と意味論を照合済み）' if HAVE_TORCH else '無し（実 FX 経路は未検証）'}")
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
