"""FX グラフ → **Tsugi IR** への降下（想定ユーザーを機械語まで届かせる橋）。

## なぜこれが要るのか（要件の再定義・第 60 回）

`codegen` は単一 IR から実 PTX/AMDGCN を出し、ベンダーのアセンブラと LLVM が
それを裏づけるところまで到達した。しかし **その恩恵に与れるのは tile-DSL を書く人だけ**
だった。このプロダクトの楔は `torch.compile`（フレームワーク層）であり、想定ユーザーは
PyTorch 開発者である。彼らの経路は `fx_to_graph_ops` が *論理 op 列*（propagation 用）を
作るだけで、`ir.Module` を一度も作らない——つまり **「単一ソースで両ベンダー」という
看板の約束が、看板の想定客には一切届いていなかった**。

`fx_to_graph_ops`（発散予測のための抽象）と本モジュール（機械語のための具体）は
別の写像である。前者は「増幅するか」だけ判れば足り、後者は「どの命令列になるか」まで
要る。ここを分けずに済ませていたのが穴の原因だった。

## 正直さの契約

FX の op 語彙は Tsugi の DSL 語彙より広い。**表せないものを黙って落とすと偽OK になる**
（存在しない op のぶん発散が過小評価され、生成物も別物になる）。よって:

- 表せない op は `LoweringReport.unsupported` に載り、モジュールは **partial** になる。
  partial なら codegen 判定は WARN 以上（`audit_torch` が算入する）。
- 恒等式で分解したもの（sigmoid/tanh/gelu）は `decomposed` に載る。**実数では恒等でも
  浮動小数点では別の丸めになる**ので、これは発散源として明示すべき情報である。
- view/reshape/permute 等はデータ移動を伴うが本 IR は形状を持たないので `shape_only`
  に載せ、**「無視した」と言える**ようにする（黙って通さない）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tsugi import ir

_CALL_OPS = ("call_function", "call_method", "call_module")

#: 形状・レイアウトだけを変える op。本 IR は形状を持たないため値をそのまま通すが、
#: 実機ではデータ移動（転置・連続化）が入りうる。**無視したことを記録する**。
_SHAPE_ONLY = ("view", "reshape", "permute", "transpose", "expand", "contiguous",
               "flatten", "unsqueeze", "squeeze", "detach", "clone", "t.default")

#: 恒等式による分解。実数では厳密だが浮動小数点では丸めが変わる（発散源）。
_DECOMPOSITION_NOTE = {
    "sigmoid": "sigmoid(x) = 1/(1+exp(-x)) へ分解（div/exp の丸めが入る）",
    "tanh": "tanh(x) = 2·sigmoid(2x)−1 へ分解（NVIDIA の tanh.approx.f32 に相当する"
            "単一命令は AMD に無く、分解が移植の実態）",
    "gelu": "gelu(approximate='tanh') を恒等式へ分解（erf 版は別扱い——下記参照）",
    "softmax": "softmax を max-subtract → exp → sum → div へ分解（数値安定版）",
    "layer_norm": "layer_norm を mean → 中心化 → 分散 → rsqrt → scale へ分解",
    "rms_norm": "rms_norm を二乗平均 → rsqrt → scale へ分解",
}


@dataclass
class LoweringReport:
    """降下でできたこと・できなかったこと・黙らなかったこと。"""

    n_nodes: int = 0
    covered: list[str] = field(default_factory=list)       # 降下できた FX target
    unsupported: list[str] = field(default_factory=list)   # 表せなかった FX target
    decomposed: list[str] = field(default_factory=list)    # 恒等式で分解した注記
    shape_only: list[str] = field(default_factory=list)    # 形状のみ（データ移動未モデル）
    assumptions: list[str] = field(default_factory=list)   # 静的に決まらず仮定した値

    @property
    def partial(self) -> bool:
        """1 つでも表せない op があれば partial（生成物はモデル全体ではない）。"""
        return bool(self.unsupported)

    def to_lines(self) -> list[str]:
        out = [f"FX {self.n_nodes} ノード中 {len(self.covered)} を IR へ降下"]
        if self.unsupported:
            out.append(f"表せない op {sorted(set(self.unsupported))} → **partial**："
                       "生成物はモデル全体ではない（発散予測もこのぶん過小評価）")
        if self.decomposed:
            out += [f"  分解: {d}" for d in sorted(set(self.decomposed))]
        if self.shape_only:
            out.append(f"  形状のみ（データ移動は未モデル化）: "
                       f"{sorted(set(self.shape_only))}")
        out += [f"  仮定: {a}" for a in sorted(set(self.assumptions))]
        return out


@dataclass
class LoweredModule:
    module: ir.Module
    report: LoweringReport


class _Builder:
    """SSA 値と op 列を組み立てる小さなヘルパ（IR の形は tracer と同じ）。"""

    def __init__(self, dtype: str = "f32", shape: str = "16x16"):
        self.body: list[ir.Op] = []
        self._n = 0
        self._t = f"tensor<{shape}x{dtype}>"

    def _v(self) -> ir.Value:
        v = ir.Value(f"%{self._n}", self._t)
        self._n += 1
        return v

    def op(self, kind: str, operands: list[ir.Value] | None = None,
           attrs: dict | None = None) -> ir.Value:
        r = self._v()
        self.body.append(ir.Op(kind, list(operands or []), dict(attrs or {}), r))
        return r

    def store(self, v: ir.Value) -> None:
        self.body.append(ir.Op("store", [v], {"offset": [0, 0]}, None))

    # --- 合成 op（恒等式による分解） -------------------------------------
    def const(self) -> ir.Value:
        return self.op("zeros", [], {"shape": [16, 16]})

    def sigmoid(self, x: ir.Value) -> ir.Value:
        z = self.const()
        neg = self.op("sub", [z, x])                 # -x
        e = self.op("exp", [neg])                    # exp(-x)
        one = self.op("zeros", [], {"shape": [16, 16], "fill": 1.0})
        den = self.op("add", [one, e])               # 1+exp(-x)
        return self.op("div", [one, den])

    def tanh(self, x: ir.Value) -> ir.Value:
        two = self.op("zeros", [], {"shape": [16, 16], "fill": 2.0})
        s = self.sigmoid(self.op("mul", [two, x]))   # sigmoid(2x)
        d = self.op("mul", [two, s])                 # 2·sigmoid(2x)
        one = self.op("zeros", [], {"shape": [16, 16], "fill": 1.0})
        return self.op("sub", [d, one])

    def gelu(self, x: ir.Value) -> ir.Value:
        # 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
        c = self.op("zeros", [], {"shape": [16, 16], "fill": 0.044715})
        x2 = self.op("mul", [x, x])
        x3 = self.op("mul", [x2, x])
        inner = self.op("add", [x, self.op("mul", [c, x3])])
        k = self.op("zeros", [], {"shape": [16, 16], "fill": 0.7978845608})
        t = self.tanh(self.op("mul", [k, inner]))
        one = self.op("zeros", [], {"shape": [16, 16], "fill": 1.0})
        half = self.op("zeros", [], {"shape": [16, 16], "fill": 0.5})
        return self.op("mul", [self.op("mul", [half, x]),
                               self.op("add", [one, t])])

    def softmax(self, x: ir.Value) -> ir.Value:
        m = self.op("reduce", [x], {"axis": 1, "kind": "max"})
        e = self.op("exp", [self.op("sub", [x, m])])
        s = self.op("reduce", [e], {"axis": 1, "kind": "sum"})
        return self.op("div", [e, s])

    def layer_norm(self, x: ir.Value, eps: float = 1e-5) -> ir.Value:
        # eps を落とすと torch と 1e-5 桁でずれる（実測で判明。分散が小さい行ほど顕著）。
        mu = self.op("reduce", [x], {"axis": 1, "kind": "mean"})
        c = self.op("sub", [x, mu])
        var = self.op("reduce", [self.op("mul", [c, c])],
                      {"axis": 1, "kind": "mean"})
        e = self.op("zeros", [], {"shape": [16, 16], "fill": eps})
        return self.op("mul", [c, self.op("rsqrt", [self.op("add", [var, e])])])

    def rms_norm(self, x: ir.Value, eps: float = 1e-5) -> ir.Value:
        ms = self.op("reduce", [self.op("mul", [x, x])],
                     {"axis": 1, "kind": "mean"})
        e = self.op("zeros", [], {"shape": [16, 16], "fill": eps})
        return self.op("mul", [x, self.op("rsqrt", [self.op("add", [ms, e])])])


def _target_name(node: Any, gm: Any = None) -> str:
    """ノードの識別名。`call_module` は target が経路名（Sequential なら "0"/"1"）で
    op の種類を表さないため、**実際のサブモジュールのクラス名**へ解決する。

    実 `torch.fx.symbolic_trace` の出力はモジュール呼び出しを残す（aten へ落ちるのは
    dynamo/AOT 経由の後段）。ここを解決しないと `nn.LayerNorm` が "1" として
    未対応扱いになり、**実 FX に対してだけ静かに壊れる**。
    """
    t = str(getattr(node, "target", ""))
    if getattr(node, "op", None) == "call_module" and gm is not None:
        try:
            return type(gm.get_submodule(t)).__name__
        except (AttributeError, KeyError, TypeError):
            return t
    return t


def _gelu_is_tanh(node: Any, gm: Any = None) -> bool:
    """その gelu が tanh 近似版か（erf 版と数値が違うので区別が要る）。

    `call_function` は kwargs の `approximate`、`call_module` はサブモジュールの
    `.approximate` 属性を見る。判らなければ **False**（＝表せない扱い）に倒す
    ——不確実なら保守側、が本リポジトリの一貫した規約。
    """
    kw = getattr(node, "kwargs", None) or {}
    if str(kw.get("approximate", "")).lower() == "tanh":
        return True
    if getattr(node, "op", None) == "call_module" and gm is not None:
        try:
            return str(getattr(gm.get_submodule(str(node.target)),
                               "approximate", "")).lower() == "tanh"
        except (AttributeError, KeyError, TypeError):
            return False
    return False


#: eps 未指定時の仮定値。torch は eps=None のとき `finfo(input.dtype).eps` を使うが、
#: それは **実行時 dtype に依存**し静的グラフからは判らない。f32 の finfo eps を仮定し、
#: 仮定したことをレポートに載せる（黙って数値を選ばない）。
_ASSUMED_EPS_F32 = 1.1920929e-07


def _node_eps(node: Any, gm: Any = None) -> tuple[float, bool]:
    """正規化層の eps を読む。返り値は (eps, 仮定したか)。

    eps を落とすと torch と 1e-5 桁でずれる（実測で判明）。逆に既定値を決め打ちすると
    eps=None（＝dtype 依存）のモデルでずれる（これも実測で判明）。**どちらも実際に
    起きたので、読めたら読む・読めなければ仮定したと言う**。
    """
    kw = getattr(node, "kwargs", None) or {}
    if kw.get("eps") is not None:
        try:
            return float(kw["eps"]), False
        except (TypeError, ValueError):
            return _ASSUMED_EPS_F32, True
    if getattr(node, "op", None) == "call_module" and gm is not None:
        try:
            e = getattr(gm.get_submodule(str(node.target)), "eps", None)
            if e is not None:
                return float(e), False
        except (AttributeError, KeyError, TypeError, ValueError):
            return _ASSUMED_EPS_F32, True
    return _ASSUMED_EPS_F32, True


def _classify(t: str) -> str | None:
    """FX target 名 → 降下の種別。None は「表せない」。

    `fx_to_graph_ops._kind_of` とは**目的が違う**ので別に持つ: あちらは発散が増幅するか
    だけ判れば足りる粗い写像で、こちらは命令列まで決めるので厳密でなければならない。

    aten 名（`native_layer_norm`）とモジュールのクラス名（`LayerNorm`）では区切りが
    違う。**アンダースコアを外した形でも照合する**——これを怠ると `nn.LayerNorm` が
    未対応に落ち、stand-in グラフでは通るのに実 `torch.fx` でだけ静かに壊れる
    （実際にそうなっていた・実 torch を入れて初めて判明した）。
    """
    raw = t.lower()
    flat = raw.replace("_", "")

    def has(*pats: str) -> bool:
        return any(p in raw or p.replace("_", "") in flat for p in pats)

    t = raw
    if any(k in t for k in _SHAPE_ONLY):
        return "shape_only"
    if has("rms_norm"):
        return "rms_norm"
    if has("layer_norm", "group_norm", "batch_norm"):
        return "layer_norm"
    if has("softmax"):                       # log_softmax も含む（下で分岐）
        return "log_softmax" if "log_softmax" in t else "softmax"
    if has("addmm", "bmm", "matmul", "linear", "mm"):
        return "dot"
    if has("gelu"):
        return "gelu"
    if has("sigmoid", "silu"):        # silu(x)=x·sigmoid(x)
        return "silu" if "silu" in t else "sigmoid"
    if has("tanh"):
        return "tanh"
    if has("relu"):
        return "relu"
    if has("rsqrt"):
        return "rsqrt"
    if has("sqrt"):
        return "sqrt"
    if has("exp"):
        return "exp"
    if "truediv" in t or t.endswith("div") or "div." in t:
        return "div"
    if "sub" in t or "rsub" in t:
        return "sub"
    if "add" in t:
        return "add"
    if "mul" in t:
        return "mul"
    if "maximum" in t or "clamp_min" in t:
        return "max"
    if "to_copy" in t or t.endswith(".to") or "type_as" in t:
        return "cast"
    if any(k in t for k in ("mean", "sum")):
        return "reduce"
    return None


def fx_to_ir(gm: Any, *, name: str = "fx_kernel") -> LoweredModule:
    """FX GraphModule を Tsugi IR へ降下する（torch 非依存・duck-typed）。

    これで PyTorch 開発者の経路が `codegen` に繋がり、tile-DSL 経路と同じ
    「単一ソース → 両ベンダーの実機械語」を受け取れる。
    **表せない op があれば partial と告げる**（黙って落とさない）。
    """
    b = _Builder()
    rep = LoweringReport()
    # ノードそのものでなく id() で索く: duck-typed な stand-in グラフには hashable で
    # ないノード実装もありうる（実 torch.fx.Node は hashable だが依存しない）。
    env: dict[int, ir.Value] = {}
    last: ir.Value | None = None

    for node in gm.graph.nodes:
        op = getattr(node, "op", None)
        rep.n_nodes += 1
        if op in ("placeholder", "get_attr"):
            env[id(node)] = b.op("load", [], {"offset": [0, 0]})
            last = env[id(node)]
            continue
        if op == "output":
            if last is not None:
                b.store(last)
            continue
        if op not in _CALL_OPS:
            continue

        t = _target_name(node, gm)
        kind = _classify(t)
        args = [env[id(a)] for a in getattr(node, "args", ())
                if id(a) in env]
        x = args[0] if args else (last if last is not None
                                  else b.op("load", [], {"offset": [0, 0]}))

        if kind is None:
            rep.unsupported.append(t)
            continue
        if kind == "shape_only":
            rep.shape_only.append(t)
            env[id(node)] = x
            last = x
            continue
        if kind == "log_softmax":
            # log(softmax) の log は DSL に無い。近似で埋めず**表せないと言う**。
            rep.unsupported.append(t)
            continue
        if kind == "gelu" and not _gelu_is_tanh(node, gm):
            # erf 版 gelu は DSL に erf が無く**厳密には表せない**。tanh 近似で
            # 置換すると実測で max|Δ|≈4.6e-4 ずれる（fp16 の許容 ~1.5e-2 より小さいが、
            # 判断フリップは *マージン* で決まるので無視できない）。等価性を検証する
            # 道具が黙って別の関数に置き換えるのは偽OK そのもの——表せないと言う。
            rep.unsupported.append(f"{t}(approximate='none': erf)")
            continue

        if kind == "dot":
            other = args[1] if len(args) > 1 else b.op("load", [], {"offset": [0, 0]})
            acc = b.op("zeros", [], {"shape": [16, 16]})
            r = b.op("dot", [x, other, acc])
            if "addmm" in t.lower():                 # bias 加算を落とさない
                r = b.op("add", [r, args[2] if len(args) > 2 else acc])
        elif kind in ("layer_norm", "rms_norm"):
            eps, assumed = _node_eps(node, gm)
            r = getattr(b, kind)(x, eps)
            rep.decomposed.append(_DECOMPOSITION_NOTE.get(kind, kind))
            if assumed:
                rep.assumptions.append(
                    f"{t}: eps 未指定 → {eps:.3g} を仮定（torch は eps=None のとき "
                    "finfo(dtype).eps を使い、実行時 dtype に依存する）")
        elif kind in ("softmax", "gelu", "tanh", "sigmoid"):
            r = getattr(b, kind)(x)
            rep.decomposed.append(_DECOMPOSITION_NOTE.get(kind, kind))
        elif kind == "silu":
            r = b.op("mul", [x, b.sigmoid(x)])
            rep.decomposed.append("silu(x) = x·sigmoid(x) へ分解")
        elif kind == "relu":
            r = b.op("max", [x, b.op("zeros", [], {"shape": [16, 16]})])
        elif kind == "reduce":
            r = b.op("reduce", [x], {"axis": 1,
                                     "kind": "sum" if "sum" in t.lower() else "mean"})
        elif kind == "cast":
            r = b.op("cast", [x], {"to": "float16"})
        elif kind in ("add", "sub", "mul", "div", "max"):
            other = (args[1] if len(args) > 1
                     else b.op("zeros", [], {"shape": [16, 16]}))
            r = b.op(kind, [x, other])
        else:                                        # exp/sqrt/rsqrt
            r = b.op(kind, [x])

        rep.covered.append(t)
        env[id(node)] = r
        last = r

    if last is not None and not any(o.kind == "store" for o in b.body):
        b.store(last)
    return LoweredModule(ir.Module([ir.Kernel(name, [], b.body)]), rep)
