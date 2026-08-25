"""Tsugi codegen — IR から **実 PTX / 実 AMDGCN テキスト**を生成し、ベンダー自身の
アセンブラ（`ptxas` / `llvm-mc`）で機械検証する層。

## なぜこれが可能なのか（要件の再定義）

長らく本リポジトリは「codegen は LLVM/MLIR + 実機が要るので不可能」と書いてきた。
これは**要件の誤り**だった。分解すると:

| 工程 | 必要なもの | この環境で可能か |
|---|---|---|
| IR → アセンブリ**テキスト**生成 | 純関数（文字列） | ✅ 可能 |
| アセンブリ → 機械語（**アセンブル**） | `ptxas` / `llvm-mc`（CPU 実行） | ✅ 可能 |
| 機械語の**実行** | GPU 実機 | ❌ 不可 |

必要なのは実機ではなく**アセンブラ**であり、それは CPU プログラムである。
`lowering.VENDOR_LOWERING` は「どの命令に落ちるか」を人手の表で主張していたが、
本層はその主張を**ベンダーのアセンブラに問い合わせて確かめる**。表がグラウンド
トゥルースに置き換わる。

## 検証レベル（正直な梯子・L3 は常に空）

- **L0 未対応** — その target 向けに命令列を持たない op
- **L1 生成のみ** — テキストは出るがアセンブラが無く未確認（fail-safe: 検証済みと言わない）
- **L2 アセンブル検証済み** — ベンダーのアセンブラが受理（命令の存在・構文・arch 可用性）
- **L3 実機実行検証済み** — **常に空**。実機が無い以上ここは埋められない

L2 が保証するのは「その命令がその arch に存在し構文が正しい」まで。
**保証しないもの**: データレイアウト（どのレーンがどの要素を持つか）の正しさ。
行列コア命令とスカラーレーンのレイアウト接合（stitching）は実機での照合が要り、
本層は該当 op に `layout-unstitched` の注記を付けて**黙らない**。

## それでも価値がある理由

アセンブラは arch 条件付きの可用性を**事実として**返す。例:

    v_mfma_f32_16x16x16f16 を gfx1100 へ  → "instruction not supported on this GPU"
    wmma.mma.sync を sm_60 へ            → "requires .target sm_70 or higher"

これは手書きの表では作り込めない種類の移植ブロッカーで、`feasibility` 層の
「起動可能性」をアセンブラ由来の根拠で補強する。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import ir

TARGETS = ("nvidia", "amd_cdna", "amd_rdna")

#: target 既定 arch。nvidia は WMMA 可用の最小世代より上（sm_80=Ampere）、
#: amd_cdna は MFMA を持つ gfx90a（MI200）、amd_rdna は WMMA を持つ gfx1100（RDNA3）。
DEFAULT_ARCH: dict[str, str] = {
    "nvidia": "sm_80",
    "amd_cdna": "gfx90a",
    "amd_rdna": "gfx1100",
}

VERIFY_LEVELS = ("L0-未対応", "L1-生成のみ", "L2-アセンブル検証済み", "L3-実機実行検証済み")

#: 生成した命令列が NVIDIA と AMD で**ビット同一**を期待できるか。
#: True = IEEE-754 が結果を一意に定める演算（両社とも正確丸め）。
#: False = 近似命令ないし精緻化列に依存し、ベンダー間でビット差が出うる。
#: これはベンダー ISA ドキュメントに基づく**構造的分類**であり、測定値ではない
#: （数値係数を発明しないというガードレールに従い ULP 数は持たない）。
BIT_EXACT_ACROSS_VENDORS: dict[str, bool] = {
    "add": True,      # IEEE 加算（正確丸め）
    "sub": True,
    "mul": True,
    "max": True,      # NaN 伝播規則の差はあるが有限値では一致
    "zeros": True,
    "load": True,
    "store": True,
    "cast": True,     # f32→f16 の RNE は IEEE が一意に定める
    "sqrt": False,    # NVIDIA sqrt.rn.f32 は正確丸め / AMD v_sqrt_f32 は近似
    "rsqrt": False,   # 双方とも近似（rsqrt.approx / v_rsq_f32）で実装が異なる
    "exp": False,     # 双方とも 2^x 近似（ex2.approx / v_exp_f32）で実装が異なる
    "div": False,     # NVIDIA div.rn.f32 は正確丸め / AMD は rcp+Newton
    "dot": False,     # 行列コアの累積順序・入力精度が異なる（equivalence 層が別途モデル化）
    "reduce": False,  # クロスレーン縮約の結合順序がベンダーで異なる
}


def approximate_ops() -> set[str]:
    """ベンダー間でビット同一を期待できない op（発散源）。"""
    return {k for k, v in BIT_EXACT_ACROSS_VENDORS.items() if not v}


# --------------------------------------------------------------------------
# 命令選択テーブル（実命令。lowering.VENDOR_LOWERING の散文表を実体化したもの）
# --------------------------------------------------------------------------

_LOG2E = "0f3FB8AA3B"      # log2(e) を f32 hex で（PTX リテラル形式）
_LOG2E_HEX = "0x3fb8aa3b"  # 同じ値の AMD リテラル形式

_PTX_BIN = {"add": "add.f32", "sub": "sub.f32", "mul": "mul.f32",
            "max": "max.f32", "div": "div.rn.f32"}
_GCN_BIN = {"add": "v_add_f32", "sub": "v_sub_f32", "mul": "v_mul_f32",
            "max": "v_max_f32"}

#: codegen が命令列を持つ op（L1 以上）。tracer.EMITTABLE_OPS の部分集合であること
#: を不変条件が検査する（新 op を足して codegen を忘れたら検出される）。
CODEGEN_OPS: frozenset[str] = frozenset(_PTX_BIN) | frozenset({
    "zeros", "load", "store", "cast", "exp", "sqrt", "rsqrt", "reduce", "dot",
})


@dataclass
class EmitResult:
    """生成物と、そこに残る正直な但し書き。"""

    target: str
    arch: str
    text: str
    covered: list[str] = field(default_factory=list)     # 命令列を出せた op
    uncovered: list[str] = field(default_factory=list)   # 出せなかった op（L0）
    notes: list[str] = field(default_factory=list)       # layout-unstitched 等

    @property
    def unstitched(self) -> list[str]:
        return [n for n in self.notes if n.startswith("layout-unstitched")]


# --------------------------------------------------------------------------
# PTX
# --------------------------------------------------------------------------

def emit_ptx(module: ir.Module, *, arch: str = "sm_80") -> EmitResult:
    """IR → PTX テキスト。各 op は実 PTX 命令列に落ちる。

    レーンモデル: 線形スレッド ID が 1 要素を担当（elementwise はレーン内、
    reduce は warp butterfly、dot は warp 単位の WMMA フラグメント）。
    dot / reduce はレーンとレイアウトが異なるため `layout-unstitched` を注記する。
    """
    # ptxas は非 ASCII を受け付けない（実測: "Unexpected non-ASCII character"）。
    # PTX 側のコメントは ASCII に限る。
    lines: list[str] = [
        "//",
        "// Tsugi codegen: generated. Assembles (ptxas); NOT executed on hardware.",
        f"// target=nvidia arch={arch}",
        "//",
        ".version 7.0",
        f".target {arch}",
        ".address_size 64",
        "",
    ]
    covered: list[str] = []
    uncovered: list[str] = []
    notes: list[str] = []

    for kernel in module.kernels:
        body: list[str] = []
        reg: dict[str, str] = {}      # SSA 名 → %f レジスタ
        nf = [0]                      # %f カウンタ（0 は予約せず 1 から）
        nh = [0]
        nr = [3]                      # %r1..3 は tid 計算で使用済み
        acc: dict[str, int] = {}      # dot の accumulator 番号

        def newf() -> str:
            nf[0] += 1
            return f"%f{nf[0]}"

        def src(v) -> str:
            return reg.get(v.name, "%f1")

        for op in kernel.body:
            k = op.kind
            if k in _PTX_BIN:
                d = newf()
                body.append(f"    {_PTX_BIN[k]} {d}, {src(op.operands[0])}, "
                            f"{src(op.operands[1])};")
                reg[op.result.name] = d
            elif k == "zeros":
                d = newf()
                body.append(f"    mov.f32 {d}, 0f00000000;")
                reg[op.result.name] = d
            elif k == "load":
                d = newf()
                body.append(f"    ld.global.f32 {d}, [%rd6];")
                reg[op.result.name] = d
            elif k == "store":
                body.append(f"    st.global.f32 [%rd7], {src(op.operands[0])};")
            elif k == "cast":
                nh[0] += 1
                d = newf()
                # f32->f16->f32 の往復。精度の切り捨ては実機と同じ RNE（コメントは出力しない）。
                body.append(f"    cvt.rn.f16.f32 %h{nh[0]}, {src(op.operands[0])};")
                body.append(f"    cvt.f32.f16 {d}, %h{nh[0]};")
                reg[op.result.name] = d
            elif k == "exp":
                t, d = newf(), newf()
                body.append(f"    mul.f32 {t}, {src(op.operands[0])}, {_LOG2E};")
                body.append(f"    ex2.approx.f32 {d}, {t};")
                reg[op.result.name] = d
            elif k in ("sqrt", "rsqrt"):
                d = newf()
                ins = "sqrt.rn.f32" if k == "sqrt" else "rsqrt.approx.f32"
                body.append(f"    {ins} {d}, {src(op.operands[0])};")
                reg[op.result.name] = d
            elif k == "reduce":
                a = src(op.operands[0]) if op.operands else newf()
                d = newf()
                body.append(f"    mov.f32 {d}, {a};")
                for delta in (16, 8, 4, 2, 1):   # warp butterfly（32 レーン）
                    nr[0] += 1
                    r0 = f"%r{nr[0]}"
                    nr[0] += 1
                    r1 = f"%r{nr[0]}"
                    t = newf()
                    body.append(f"    mov.b32 {r0}, {d};")
                    body.append(f"    shfl.sync.bfly.b32 {r1}, {r0}, {delta}, 31, -1;")
                    body.append(f"    mov.b32 {t}, {r1};")
                    body.append(f"    add.f32 {d}, {d}, {t};")
                reg[op.result.name] = d
                notes.append("layout-unstitched: reduce は warp 単位（32 レーン）で"
                             "縮約する。IR のタイル軸との対応は実機照合が要る。")
            elif k == "dot":
                i = len(acc) + 1
                acc[op.result.name] = i
                afr = ",".join(f"%fa{j}" for j in range(1, 9))
                bfr = ",".join(f"%fb{j}" for j in range(1, 9))
                cfr = ",".join(f"%fd{i}_{j}" for j in range(1, 9))
                prev = op.operands[2].name if len(op.operands) > 2 else None
                pi = acc.get(prev)
                if pi is None:
                    for j in range(1, 9):
                        body.append(f"    mov.f32 %fd{i}_{j}, 0f00000000;")
                    src_c = cfr
                else:
                    src_c = ",".join(f"%fd{pi}_{j}" for j in range(1, 9))
                body.append("    wmma.load.a.sync.aligned.row.m16n16k16.global.f16 "
                            f"{{{afr}}}, [%rd4], 16;")
                body.append("    wmma.load.b.sync.aligned.col.m16n16k16.global.f16 "
                            f"{{{bfr}}}, [%rd5], 16;")
                body.append("    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 "
                            f"{{{cfr}}}, {{{afr}}}, {{{bfr}}}, {{{src_c}}};")
                notes.append("layout-unstitched: dot は warp 単位の WMMA フラグメント"
                             "（m16n16k16）。スカラーレーンとのレイアウト接合は実機照合が要る。")
                covered.append(k)
                continue
            else:
                body.append(f"    // <UNSUPPORTED {k}> — L0")
                uncovered.append(k)
                continue
            covered.append(k)

        nd = 8
        lines.append(f".visible .entry {kernel.name}(")
        lines.append(f"    .param .u64 {kernel.name}_a,")
        lines.append(f"    .param .u64 {kernel.name}_b,")
        lines.append(f"    .param .u64 {kernel.name}_c")
        lines.append(")")
        lines.append("{")
        lines.append(f"    .reg .b64   %rd<{nd}>;")
        lines.append(f"    .reg .b32   %r<{nr[0] + 1}>;")
        lines.append(f"    .reg .f32   %f<{nf[0] + 1}>;")
        if nh[0]:
            lines.append(f"    .reg .b16   %h<{nh[0] + 1}>;")
        if acc:
            lines.append("    .reg .b32   %fa<9>;")
            lines.append("    .reg .b32   %fb<9>;")
            for i in range(1, len(acc) + 1):
                lines.append(f"    .reg .f32   %fd{i}_<9>;")
        lines += [
            "",
            f"    ld.param.u64 %rd1, [{kernel.name}_a];",
            f"    ld.param.u64 %rd2, [{kernel.name}_b];",
            f"    ld.param.u64 %rd3, [{kernel.name}_c];",
            "    cvta.to.global.u64 %rd4, %rd1;",
            "    cvta.to.global.u64 %rd5, %rd2;",
            "    mov.u32 %r1, %ctaid.x;",
            "    mov.u32 %r2, %ntid.x;",
            "    mov.u32 %r3, %tid.x;",
            "    mad.lo.s32 %r1, %r1, %r2, %r3;",   # 線形スレッド ID
            "    mul.wide.s32 %rd6, %r1, 4;",
            "    add.s64 %rd7, %rd3, %rd6;",
            "    add.s64 %rd6, %rd4, %rd6;",
            "",
        ]
        lines += body
        lines.append("    ret;")
        lines.append("}")
        lines.append("")

    return EmitResult("nvidia", arch, "\n".join(lines), covered, uncovered,
                      sorted(set(notes)))


# --------------------------------------------------------------------------
# AMDGCN
# --------------------------------------------------------------------------

def emit_amdgcn(module: ir.Module, *, arch: str = "gfx90a",
                isa: str | None = None) -> EmitResult:
    """IR → AMDGCN テキスト。CDNA(MFMA) と RDNA(WMMA) を arch で切り替える。

    `isa` を明示すると arch から推定せずその ISA 系統の命令を出す。
    これは**わざと不整合な組み合わせ**（CDNA の MFMA を RDNA の arch へ）を作って
    アセンブラに可否を問うための穴で、`probe_op` が移植ブロッカーの検出に使う。

    HSA カーネルディスクリプタ（`.amdhsa_kernel`）は出力しない。よって生成物は
    **アセンブル可能だがそのままではロード不可**。ローダブル化は実機ランタイムの
    ABI に依存し、実機検証と同じ枠（L3）に属する。
    """
    target = isa or ("amd_rdna" if arch.startswith(("gfx11", "gfx12"))
                     else "amd_cdna")
    lines: list[str] = [
        "//",
        "// Tsugi codegen — 生成物。アセンブル検証は可能・実機実行は未検証。",
        f"// target={target} arch={arch}",
        "// note: .amdhsa_kernel 記述子は未出力（ロード不可・実機 ABI 依存）",
        "//",
        "\t.text",
        f'\t.amdgcn_target "amdgcn-amd-amdhsa--{arch}"',
        "",
    ]
    covered: list[str] = []
    uncovered: list[str] = []
    notes: list[str] = ["load/store の記述子未出力: このアセンブリは検証用であり"
                        "ロード可能なオブジェクトではない。"]

    # RDNA3 は同じ機械語命令を別ニーモニックで綴る（CDNA の *_dword 系は別名として
    # 受理されるが、逆アセンブルすると RDNA3 の綴りで返る）。往復検証（verify_encoding）
    # がこの silently-aliased を検出したので、arch ごとに正しい綴りを出す。
    mem = ({"load": "global_load_b32", "store": "global_store_b32",
            "sload": "s_load_b128"} if target == "amd_rdna" else
           {"load": "global_load_dword", "store": "global_store_dword",
            "sload": "s_load_dwordx4"})

    for kernel in module.kernels:
        body: list[str] = []
        reg: dict[str, str] = {}
        nv = [10]              # v0..v9 は index/アドレス計算に予約
        acc_seen = [False]

        def newv() -> str:
            nv[0] += 1
            return f"v{nv[0]}"

        def src(v) -> str:
            return reg.get(v.name, "v10")

        for op in kernel.body:
            k = op.kind
            if k in _GCN_BIN:
                d = newv()
                body.append(f"\t{_GCN_BIN[k]} {d}, {src(op.operands[0])}, "
                            f"{src(op.operands[1])}")
                reg[op.result.name] = d
            elif k == "div":
                t, d = newv(), newv()
                # AMD に正確丸めの除算命令は無い。rcp + 乗算（Newton 精緻化は
                # 実機の精度要件で決まるため出力しない）。ゆえに NVIDIA の
                # div.rn.f32 とはビット同一にならない（BIT_EXACT=False）。
                body.append(f"\tv_rcp_f32 {t}, {src(op.operands[1])}")
                body.append(f"\tv_mul_f32 {d}, {src(op.operands[0])}, {t}")
                reg[op.result.name] = d
            elif k == "zeros":
                d = newv()
                body.append(f"\tv_mov_b32 {d}, 0")
                reg[op.result.name] = d
            elif k == "load":
                d = newv()
                body.append(f"\t{mem['load']} {d}, v[0:1], off")
                body.append("\ts_waitcnt vmcnt(0)")
                reg[op.result.name] = d
            elif k == "store":
                body.append(f"\t{mem['store']} v[2:3], {src(op.operands[0])}, off")
            elif k == "cast":
                t, d = newv(), newv()
                body.append(f"\tv_cvt_f16_f32 {t}, {src(op.operands[0])}")
                body.append(f"\tv_cvt_f32_f16 {d}, {t}")
                reg[op.result.name] = d
            elif k == "exp":
                t, d = newv(), newv()
                body.append(f"\tv_mul_f32 {t}, {_LOG2E_HEX}, {src(op.operands[0])}")
                body.append(f"\tv_exp_f32 {d}, {t}")
                reg[op.result.name] = d
            elif k in ("sqrt", "rsqrt"):
                d = newv()
                body.append(f"\t{'v_sqrt_f32' if k == 'sqrt' else 'v_rsq_f32'} "
                            f"{d}, {src(op.operands[0])}")
                reg[op.result.name] = d
            elif k == "reduce":
                a = src(op.operands[0]) if op.operands else newv()
                d = newv()
                body.append(f"\tv_mov_b32 {d}, {a}")
                for shr in (1, 2, 4, 8):     # DPP row shift による木縮約
                    body.append(f"\tv_add_f32_dpp {d}, {d}, {d} row_shr:{shr} "
                                "row_mask:0xf bank_mask:0xf")
                reg[op.result.name] = d
                notes.append("layout-unstitched: reduce は DPP row（16 レーン）単位。"
                             "IR のタイル軸との対応は実機照合が要る。")
            elif k == "dot":
                d = newv()
                if target == "amd_cdna":
                    if not acc_seen[0]:
                        for j in range(4):
                            body.append(f"\tv_accvgpr_write_b32 a{j}, 0")
                        body.append("\ts_nop 1")
                        acc_seen[0] = True
                    body.append("\tv_mfma_f32_16x16x16f16 a[0:3], v[4:5], v[6:7], a[0:3]")
                    body.append("\ts_nop 7")
                    body.append(f"\tv_accvgpr_read_b32 {d}, a0")
                else:
                    body.append("\tv_wmma_f32_16x16x16_f16 v[40:47], v[24:31], "
                                "v[32:39], v[40:47]")
                    body.append(f"\tv_mov_b32 {d}, v40")
                reg[op.result.name] = d
                notes.append("layout-unstitched: dot は wave 単位の行列コア"
                             "（CDNA=MFMA / RDNA=WMMA）。レイアウト接合は実機照合が要る。")
            else:
                body.append(f"\t; <UNSUPPORTED {k}> — L0")
                uncovered.append(k)
                continue
            covered.append(k)

        lines.append(f"\t.globl\t{kernel.name}")
        lines.append("\t.p2align\t8")
        lines.append(f"\t.type\t{kernel.name},@function")
        lines.append(f"{kernel.name}:")
        lines.append(f"\t{mem['sload']} s[0:3], s[4:5], 0x0")
        lines.append("\ts_waitcnt lgkmcnt(0)")
        lines.append("\tv_lshlrev_b32 v0, 2, v0")
        lines += body
        lines.append("\ts_endpgm")
        lines.append(f".Lfunc_end_{kernel.name}:")
        lines.append(f"\t.size\t{kernel.name}, .Lfunc_end_{kernel.name}-{kernel.name}")
        lines.append("")

    return EmitResult(target, arch, "\n".join(lines), covered, uncovered,
                      sorted(set(notes)))


def emit(module: ir.Module, *, target: str = "nvidia",
         arch: str | None = None) -> EmitResult:
    """IR → target 向けアセンブリテキスト（単一入口）。"""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    arch = arch or DEFAULT_ARCH[target]
    if target == "nvidia":
        return emit_ptx(module, arch=arch)
    return emit_amdgcn(module, arch=arch, isa=target)


# --------------------------------------------------------------------------
# 単一 op のプローブ（arch 条件付き可用性をアセンブラに問う）
# --------------------------------------------------------------------------

_F32 = "tensor<16x16xf32>"
_F16 = "tensor<16x16xf16>"


def _probe_module(kind: str) -> ir.Module:
    """kind ひとつだけを含む最小 IR。オペランドは load で用意する。"""
    a = ir.Value("%0", _F16)
    b = ir.Value("%1", _F16)
    c = ir.Value("%2", _F32)
    body = [ir.Op("load", [], {"offset": [0, 0]}, a),
            ir.Op("load", [], {"offset": [0, 0]}, b),
            ir.Op("zeros", [], {"shape": [16, 16]}, c)]
    r = ir.Value("%3", _F32)
    if kind in ("load", "zeros"):
        pass
    elif kind == "store":
        body.append(ir.Op("store", [c], {"offset": [0, 0]}, None))
    elif kind == "dot":
        body.append(ir.Op("dot", [a, b, c], {}, r))
    elif kind in ("add", "sub", "mul", "div", "max"):
        body.append(ir.Op(kind, [a, b], {}, r))
    else:                                   # 単項（exp/sqrt/rsqrt/cast/reduce）
        attrs = {"to": "float16"} if kind == "cast" else {}
        body.append(ir.Op(kind, [c], attrs, r))
    return ir.Module([ir.Kernel("probe", [], body)])


def probe_op(kind: str, *, target: str, arch: str | None = None,
             isa: str | None = None) -> AssembleResult:
    """「この op の命令列は、この arch に存在するか」をベンダーのアセンブラに問う。

    手書きの対応表ではなくツールチェインが真値を返すので、`v_mfma` が RDNA に
    無いといった**世代条件の移植ブロッカー**が事実として出てくる。
    `isa` は AMD 系で ISA 系統を強制する（`emit_amdgcn` 参照）。
    """
    if kind not in CODEGEN_OPS:
        return AssembleResult(target, arch or DEFAULT_ARCH.get(target, "?"),
                              available=True, ok=False,
                              stderr=f"no instruction sequence for op {kind!r} (L0)")
    module = _probe_module(kind)
    arch = arch or DEFAULT_ARCH[target]
    em = (emit_ptx(module, arch=arch) if target == "nvidia"
          else emit_amdgcn(module, arch=arch, isa=isa or target))
    return assemble(em.text, target=target, arch=arch)


# --------------------------------------------------------------------------
# アセンブラ（ベンダー自身のツールを真値に使う）
# --------------------------------------------------------------------------

@dataclass
class AssembleResult:
    """アセンブル結果。

    fail-safe: ツールが無いときは `available=False, ok=None`。
    **「検証していない」を「合格」に丸めない**（偽OK 禁止）。
    """

    target: str
    arch: str
    available: bool
    ok: bool | None
    tool: str = ""
    stderr: str = ""
    obj_bytes: int = 0

    @property
    def level(self) -> str:
        if not self.available:
            return VERIFY_LEVELS[1]      # L1-生成のみ
        return VERIFY_LEVELS[2] if self.ok else VERIFY_LEVELS[0]


def _which_ptxas() -> str | None:
    """`ptxas` を探す。PATH → 環境変数 → pip の nvidia-cuda-nvcc-cu12 同梱物。"""
    env = os.environ.get("TSUGI_PTXAS")
    if env and Path(env).exists():
        return env
    found = shutil.which("ptxas")
    if found:
        return found
    for base in {Path(p) for p in sys.path if p} | {Path(sys.prefix) / "lib"}:
        cand = base / "nvidia" / "cuda_nvcc" / "bin" / "ptxas"
        if cand.exists():
            return str(cand)
    return None


def _which_llvm_mc() -> str | None:
    """`llvm-mc` を探す（バージョン付き名も見る）。"""
    env = os.environ.get("TSUGI_LLVM_MC")
    if env and Path(env).exists():
        return env
    for name in ("llvm-mc", *(f"llvm-mc-{v}" for v in range(21, 13, -1))):
        found = shutil.which(name)
        if found:
            return found
    return None


def toolchain(target: str) -> str | None:
    """target のアセンブラのパス。無ければ None（＝L1 止まり）。"""
    return _which_ptxas() if target == "nvidia" else _which_llvm_mc()


def assemble(text: str, *, target: str, arch: str | None = None,
             timeout: float = 60.0) -> AssembleResult:
    """生成したアセンブリをベンダーのアセンブラにかけ、受理されるか確かめる。

    これが L2（アセンブル検証済み）の根拠。**実行はしない**（L3 には到達しない）。
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    arch = arch or DEFAULT_ARCH[target]
    tool = toolchain(target)
    if tool is None:
        # どちらも本体の依存としては宣言しない（ptxas は CUDA EULA・proprietary で
        # Apache-2.0 配布物に引き込まない。llvm-mc は pip 依存ではない）。
        hint = ("pip install nvidia-cuda-nvcc-cu12  (NVIDIA CUDA EULA)"
                if target == "nvidia" else "apt install llvm  (llvm-mc)")
        return AssembleResult(target, arch, available=False, ok=None,
                              stderr=f"assembler not found — {hint}")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        if target == "nvidia":
            src, out = d / "k.ptx", d / "k.cubin"
            src.write_text(text, encoding="utf-8")
            cmd = [tool, f"-arch={arch}", str(src), "-o", str(out)]
        else:
            src, out = d / "k.s", d / "k.o"
            src.write_text(text, encoding="utf-8")
            cmd = [tool, "-triple=amdgcn-amd-amdhsa", f"-mcpu={arch}",
                   "-filetype=obj", str(src), "-o", str(out)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return AssembleResult(target, arch, available=True, ok=False,
                                  tool=tool, stderr=str(exc))
        # llvm-mc は構文エラーでも 0 を返すことがある（実測）。stderr の
        # "error" と出力オブジェクトの存在で判定する（偽OK を作らない）。
        err = (p.stderr or "").strip()
        ok = (p.returncode == 0 and "error" not in err.lower() and out.exists())
        size = out.stat().st_size if out.exists() else 0
    return AssembleResult(target, arch, available=True, ok=ok, tool=tool,
                          stderr=err, obj_bytes=size)


@dataclass
class EncodingCheck:
    """「意図した命令が本当にその機械語になったか」を*第二のツール*で確かめた結果。

    アセンブラが受理する（L2）ことと、意図どおりに符号化される
    ことは別である。別名・別エンコーディングへ黙って解釈される可能性が残る。
    そこで出来上がったオブジェクトを**逆アセンブラ／シンボルリーダに読ませ直す**。
    真値が「自分で書いたテキスト」でなく「ツールが復号したもの」になるのが要点。
    """

    target: str
    arch: str
    available: bool
    ok: bool | None
    method: str = ""              # disasm-roundtrip | elf-symbols+resources
    intended: list[str] = field(default_factory=list)
    decoded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    spill_bytes: int | None = None       # NVIDIA のみ（ptxas -v）
    registers: int | None = None         # NVIDIA のみ（ptxas -v）
    detail: str = ""


def _mnemonics(text: str, target: str) -> list[str]:
    """生成テキストから *意図した* 命令ニーモニックを拾う（順序つき）。"""
    out: list[str] = []
    for raw in text.splitlines():
        ln = raw.strip()
        if (not ln or ln.startswith(("//", ";", ".", "{", "}", "@", ")", "("))
                or ln.endswith(":")):
            continue
        if target == "nvidia" and ln.startswith(("ld.param", "cvta", "ret")):
            continue
        head = ln.split()[0].rstrip(",")
        if head.startswith("%") or "=" in head:
            continue
        out.append(head)
    return out


def _run(cmd: list[str], timeout: float = 60.0):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def verify_encoding(module: ir.Module, *, target: str = "nvidia",
                    arch: str | None = None) -> EncodingCheck:
    """アセンブル済みオブジェクトを**読み直して**、意図した命令の実在を確かめる。

    AMD: `llvm-objdump -d` で逆アセンブルし、意図したニーモニックが復号結果に
    現れることを見る（`v_lshlrev_b32` → `v_lshlrev_b32_e32` のようにエンコーディング
    接尾辞が付くので前方一致で照合する）。これは往復検証であり、テキストと機械語の
    対応をツールが保証する。

    NVIDIA: SASS 逆アセンブラ（nvdisasm）は本環境に入手手段が無いため往復はできない。
    代わりに **cubin の ELF シンボル**（カーネル名が機械語オブジェクトに存在するか）と
    `ptxas -v` の資源レポート（レジスタ数・spill バイト）を根拠にする。
    **往復していないことは method フィールドで自己申告する**（同等と偽らない）。
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    arch = arch or DEFAULT_ARCH[target]
    tool = toolchain(target)
    em = emit(module, target=target, arch=arch)
    intended = sorted(set(_mnemonics(em.text, target)))
    if tool is None:
        return EncodingCheck(target, arch, available=False, ok=None,
                             intended=intended, detail="assembler not found")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        if target == "nvidia":
            src, obj = d / "k.ptx", d / "k.cubin"
            src.write_text(em.text, encoding="utf-8")
            p = _run([tool, "-v", f"-arch={arch}", str(src), "-o", str(obj)])
            if p is None or not obj.exists():
                return EncodingCheck(target, arch, available=True, ok=False,
                                     intended=intended,
                                     detail=(p.stderr if p else "ptxas failed")[:400])
            info = (p.stderr or "") + (p.stdout or "")
            regs = spill = None
            for ln in info.splitlines():
                if "registers" in ln:
                    for tok in ln.replace(",", " ").split():
                        if tok.isdigit():
                            regs = int(tok)
                            break
                if "spill stores" in ln:
                    parts = ln.split()
                    for i, tok in enumerate(parts):
                        if tok == "bytes" and i + 1 < len(parts) \
                                and parts[i + 1].startswith("spill"):
                            spill = int(parts[i - 1])
                            break
            syms = _elf_symbols(obj)
            names = {k.name for k in module.kernels}
            ok = bool(names & set(syms)) and (spill is None or spill == 0)
            return EncodingCheck(
                target, arch, available=True, ok=ok,
                method="elf-symbols+resources（SASS 往復は nvdisasm 不在ゆえ未実施）",
                intended=intended, symbols=syms, registers=regs, spill_bytes=spill,
                detail=f"kernels={sorted(names)} symbols={syms}")

        src, obj = d / "k.s", d / "k.o"
        src.write_text(em.text, encoding="utf-8")
        p = _run([tool, "-triple=amdgcn-amd-amdhsa", f"-mcpu={arch}",
                  "-filetype=obj", str(src), "-o", str(obj)])
        if p is None or not obj.exists():
            return EncodingCheck(target, arch, available=True, ok=False,
                                 intended=intended,
                                 detail=(p.stderr if p else "llvm-mc failed")[:400])
        dis = shutil.which("llvm-objdump") or shutil.which("llvm-objdump-18")
        if dis is None:
            return EncodingCheck(target, arch, available=False, ok=None,
                                 intended=intended,
                                 detail="llvm-objdump not found — 往復検証は未実施")
        q = _run([dis, "-d", f"--mcpu={arch}", str(obj)])
        if q is None or q.returncode != 0:
            return EncodingCheck(target, arch, available=True, ok=False,
                                 intended=intended,
                                 detail=(q.stderr if q else "objdump failed")[:400])
        decoded = sorted({ln.strip().split()[0]
                          for ln in q.stdout.splitlines()
                          if ln.startswith("\t") and ln.strip()})
        # 復号側は `_e32` 等のエンコーディング接尾辞が付くので前方一致で照合する。
        missing = [m for m in intended
                   if not any(dm.startswith(m) for dm in decoded)]
        syms = _elf_symbols(obj)
        return EncodingCheck(target, arch, available=True, ok=not missing,
                             method="disasm-roundtrip", intended=intended,
                             decoded=decoded, missing=missing, symbols=syms,
                             detail=f"{len(decoded)} 種の命令を復号")


def _elf_symbols(path: Path) -> list[str]:
    """オブジェクトの定義済みシンボル名（llvm-nm・無ければ空）。"""
    nm = shutil.which("llvm-nm") or shutil.which("llvm-nm-18") or shutil.which("nm")
    if nm is None:
        return []
    p = _run([nm, "--defined-only", str(path)])
    if p is None or p.returncode != 0:
        return []
    return [ln.split()[-1] for ln in p.stdout.splitlines() if ln.strip()]


def verify_codegen(module: ir.Module, *, target: str = "nvidia",
                   arch: str | None = None) -> tuple[EmitResult, AssembleResult]:
    """生成 → アセンブルを一息で。codegen 層の標準の使い方。"""
    em = emit(module, target=target, arch=arch)
    asm = assemble(em.text, target=target, arch=em.arch)
    return em, asm


def codegen_coverage(target: str) -> tuple[int, int]:
    """target が命令列を持つ op 数 / DSL が emit しうる op 数。"""
    from .tracer import EMITTABLE_OPS
    return len(CODEGEN_OPS & EMITTABLE_OPS), len(EMITTABLE_OPS)


def uncodegenned_ops(target: str) -> set[str]:
    """DSL が emit しうるのに codegen が命令列を持たない op（嘘をつかない）。"""
    from .tracer import EMITTABLE_OPS
    return set(EMITTABLE_OPS) - set(CODEGEN_OPS)
