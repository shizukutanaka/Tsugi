"""tsugi_torch._tsugi_compile のテスト（torch.compile backend の楔・市販品質強化）。

torch を使わず duck-typed の stand-in FX グラフで警告メッセージの内容を検証する。
`audit_fx()` は nondeterministic_ops/requires_noise_floor を既に計算していたが、
従来 `_tsugi_compile()` の警告メッセージには一切反映されていなかった（audit_fx の
戻り値がユーザー向け警告という facade に届いていなかった——このプロジェクトが
繰り返し見つけてきた「実装済みだが facade 未接続」と同型の欠陥）。

実 torch.compile との結線は torch 環境が要る（本環境では未検証・主張と実装の一致）。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi_torch import _tsugi_compile  # noqa: E402


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
    """torch.fx.GraphModule の最小 duck-type。forward は例外を送出し ref_logits=None に
    フォールバックさせる（本テストの対象は警告文言であり実行結果ではないため簡略化）。
    """
    def __init__(self, nodes):
        self.graph = _Graph(nodes)

    def forward(self, *args, **kwargs):
        raise RuntimeError("duck-type stand-in: no real forward")


def _nondet_graph():
    return _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),
        _Node("call_function", "aten.scatter_add.default"),    # 非決定（forward atomicAdd）
        _Node("call_function", "aten._softmax.default"),
        _Node("output", "output"),
    ])


def _deterministic_graph():
    return _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),
        _Node("call_function", "aten._softmax.default"),
        _Node("output", "output"),
    ])


def _normalization_graph():
    return _GM([
        _Node("placeholder", "x"),
        _Node("call_function", "aten.addmm.default", (8, 512)),
        _Node("call_function", "aten.native_layer_norm.default"),
        _Node("output", "output"),
    ])


def test_tsugi_compile_warns_about_nondeterministic_ops():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fwd = _tsugi_compile(_nondet_graph(), [])
    assert len(w) == 1, f"warning が期待通り 1 件出ていない: {w}"
    msg = str(w[0].message)
    assert "non-deterministic" in msg, f"nondeterminism 情報が警告に含まれない: {msg}"
    assert "scatter_add" in msg
    assert "noise floor" in msg
    assert callable(fwd)   # eager フォールバックは変わらず動く


def test_tsugi_compile_no_nondeterminism_tag_for_deterministic_graph():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _tsugi_compile(_deterministic_graph(), [])
    assert len(w) == 1
    msg = str(w[0].message)
    assert "non-deterministic" not in msg, f"決定論的グラフに nondeterminism タグが誤って付いた: {msg}"


def test_tsugi_compile_warns_about_normalization_layers():
    """正規化層を含むグラフでは、RMSNorm の中立性と LayerNorm の増幅を警告に明示する。

    旧警告は「model_divergence は scale リセット未考慮の保守的な上界（実際の発散は
    これより小さい可能性）」と *無条件に* 主張していたが、A-5 の数値実験でこれが
    誤りと判明した——LayerNorm は平均優勢入力で相対発散を amp≈RMS/σ に増幅する
    （shift=10 で実測 10.10）。「実際はもっと小さい」は偽OK 方向の未検証主張であり、
    ユーザーが WARN を過小評価する根拠になるため撤回した（FEATURE-AUDIT.md A-5）。
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _tsugi_compile(_normalization_graph(), [])
    assert len(w) == 1
    msg = str(w[0].message)
    assert "has_normalization" in msg, f"正規化層の情報が警告に含まれない: {msg}"
    assert "増幅" in msg, f"LayerNorm の増幅が警告に含まれない: {msg}"
    # 撤回した偽OK 方向の主張が復活していないこと（文言の回帰防止）
    assert "小さい可能性" not in msg and "保守的な上界" not in msg, \
        f"撤回済みの偽OK 主張が警告に復活している: {msg}"

    # 正規化層の無いグラフにはこのタグが付かない
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        _tsugi_compile(_deterministic_graph(), [])
    assert len(w2) == 1
    assert "has_normalization" not in str(w2[0].message)


def test_tsugi_compile_forward_delegates_to_eager():
    # codegen 未実装のため、返された forward は gm.forward にそのまま委譲する（性能利得なし・明示）。
    # 内部で ref_logits 取得のため example_inputs での forward 呼び出しが 1 回既に走るので、
    # そちらとは別に「返された callable 自体が正しく委譲するか」だけを見る。
    class _EagerGM(_GM):
        def __init__(self, nodes):
            super().__init__(nodes)
            self.calls = []

        def forward(self, *args, **kwargs):
            self.calls.append(args)
            return args[0] if args else None

    gm = _EagerGM([_Node("placeholder", "x"), _Node("output", "output")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd = _tsugi_compile(gm, [123])
    n_calls_from_ref_logits_extraction = len(gm.calls)
    result = fwd(456)
    assert result == 456
    assert gm.calls[-1] == (456,)
    assert len(gm.calls) == n_calls_from_ref_logits_extraction + 1


def main() -> int:
    ok = True
    tests = [
        test_tsugi_compile_warns_about_nondeterministic_ops,
        test_tsugi_compile_no_nondeterminism_tag_for_deterministic_graph,
        test_tsugi_compile_warns_about_normalization_layers,
        test_tsugi_compile_forward_delegates_to_eager,
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
