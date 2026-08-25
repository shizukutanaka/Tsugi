"""codegen のテスト — 生成した PTX/AMDGCN が**ベンダー自身のアセンブラ**に通るか。

本スイートの真値は Tsugi の内部ではなく `ptxas` / `llvm-mc` である。ゆえにここで
固定するのは「Tsugi の出力が正しいと Tsugi が言う」ことではなく「NVIDIA と AMD の
ツールチェインが受理する」こと。

正直さの契約（偽OK 禁止）: アセンブラが**無い**環境では ok は True でなく None に
なり、レベルは L1-生成のみに落ちる。「検証していない」を「合格」に丸めない。
実行（L3）はこの環境では到達不能であり、どのテストもそれを主張しない。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tsugi  # noqa: E402
from tsugi import tile  # noqa: E402
from tsugi.codegen import (  # noqa: E402
    BIT_EXACT_ACROSS_VENDORS,
    CODEGEN_OPS,
    DEFAULT_ARCH,
    TARGETS,
    VERIFY_LEVELS,
    approximate_ops,
    assemble,
    emit,
    probe_op,
    toolchain,
    uncodegenned_ops,
    verify_codegen,
    verify_encoding,
)
from tsugi.tracer import EMITTABLE_OPS  # noqa: E402


@tsugi.jit
def _kernel(a, b, c, M, N, K, BM, BN, BK):
    pid_m = tsugi.program_id(0)
    pid_n = tsugi.program_id(1)
    acc = tile.zeros((BM, BN), tsugi.float32)
    for k in range(0, K, BK):
        ta = tile.load(a, (pid_m * BM, k), (BM, BK))
        tb = tile.load(b, (k, pid_n * BN), (BK, BN))
        acc = tile.dot(ta, tb, acc)
    tile.store(c, (pid_m * BM, pid_n * BN), acc.to(tsugi.float16))


@tsugi.jit
def _softmaxish(x, y, N, BN):
    """exp/reduce/div/sqrt/rsqrt/max を一通り含む経路（softmax + norm 相当）。"""
    t = tile.load(x, (0, 0), (BN, BN))
    m = tile.reduce(t, axis=1, kind="max")
    e = tile.exp(t - m)
    s = tile.reduce(e, axis=1, kind="sum")
    n = tile.rsqrt(tile.sqrt(s))
    tile.store(y, (0, 0), tile.maximum(e / s, n))


def _ir(kernel=None):
    import numpy as np
    if kernel is None:
        a = np.zeros((64, 64), dtype=np.float16)
        b = np.zeros((64, 64), dtype=np.float16)
        c = np.zeros((64, 64), dtype=np.float32)
        return tsugi.trace(_kernel, (a, b, c, 64, 64, 64, 32, 32, 32), {},
                           program_ids=(0, 0))
    x = np.zeros((64, 64), dtype=np.float32)
    y = np.zeros((64, 64), dtype=np.float32)
    return tsugi.trace(kernel, (x, y, 64, 64), {}, program_ids=(0, 0))


def test_one_ir_emits_assembly_for_every_vendor_target():
    """単一ソース → 3 ターゲットの実アセンブリ。生成そのものは常に成立する。"""
    mod = _ir()
    for t in TARGETS:
        em = emit(mod, target=t)
        assert em.text.strip(), t
        assert em.arch == DEFAULT_ARCH[t]
        assert not em.uncovered, f"{t}: 命令列を持たない op {em.uncovered}"
        # 各ターゲットの必須ディレクティブ（構造の最低限）
        if t == "nvidia":
            for d in (".version", ".target", ".address_size", ".visible .entry"):
                assert d in em.text, f"PTX に {d} が無い"
        else:
            assert ".amdgcn_target" in em.text
            assert "s_endpgm" in em.text


def test_vendor_assemblers_accept_generated_code_or_we_do_not_claim_verified():
    """アセンブラがあれば受理されねばならない。無ければ L1 に落ちる（True にしない）。"""
    mod = _ir()
    for t in TARGETS:
        em, asm = verify_codegen(mod, target=t)
        if toolchain(t) is None:
            assert asm.available is False
            assert asm.ok is None, "未検証を合格に丸めてはならない"
            assert asm.level == VERIFY_LEVELS[1], asm.level
            continue
        assert asm.ok is True, f"{t}/{em.arch} 不受理: {asm.stderr[:400]}"
        assert asm.level == VERIFY_LEVELS[2]
        assert asm.obj_bytes > 0, "オブジェクトが空"


def test_transcendental_kernel_also_assembles():
    """exp/reduce/div を含む経路（softmax 相当）も両社で通る。"""
    mod = _ir(_softmaxish)
    kinds = set(mod.op_kinds())
    assert {"exp", "reduce", "div", "sqrt", "rsqrt", "max"} <= kinds, kinds
    for t in TARGETS:
        em, asm = verify_codegen(mod, target=t)
        if toolchain(t) is None:
            assert asm.ok is None
            continue
        assert asm.ok is True, f"{t}/{em.arch}: {asm.stderr[:400]}"


def test_assembler_reports_arch_conditional_availability():
    """世代条件の移植ブロッカーを**ツールチェインが事実として**返す。

    ここが本層の存在理由。手書きの対応表では作り込めない種類の判定:
      - WMMA は sm_70 以上（sm_60 では不可）
      - MFMA は CDNA 専用（RDNA=gfx1100 には存在しない）
    アセンブラが無い環境では ok=None になるので、その場合は主張しない。
    """
    if toolchain("nvidia") is not None:
        assert probe_op("dot", target="nvidia", arch="sm_80").ok is True
        old = probe_op("dot", target="nvidia", arch="sm_60")
        assert old.ok is False, "sm_60 で WMMA が通ってしまった"
        assert "sm_70" in old.stderr, old.stderr[:300]
        # 世代非依存の op は sm_60 でも通る（上の失敗が arch 由来だと示す対照）
        assert probe_op("add", target="nvidia", arch="sm_60").ok is True
    if toolchain("amd_cdna") is not None:
        assert probe_op("dot", target="amd_cdna", arch="gfx90a").ok is True
        assert probe_op("dot", target="amd_rdna", arch="gfx1100").ok is True
        # CDNA の ISA を RDNA の arch へ持ち込むと成立しない（単一命令での移植不可）
        cross = probe_op("dot", target="amd_cdna", arch="gfx1100", isa="amd_cdna")
        assert cross.ok is False, "MFMA が gfx1100 で通ってしまった"
        assert "not supported" in cross.stderr.lower(), cross.stderr[:300]
        assert probe_op("add", target="amd_cdna", arch="gfx1100").ok is True


def test_every_codegen_op_is_a_real_dsl_op_and_gaps_are_reported():
    """codegen の語彙は DSL の語彙の部分集合。穴は隠さず uncodegenned_ops が返す。"""
    assert CODEGEN_OPS <= EMITTABLE_OPS, CODEGEN_OPS - EMITTABLE_OPS
    for t in TARGETS:
        gaps = uncodegenned_ops(t)
        assert gaps == set(EMITTABLE_OPS) - set(CODEGEN_OPS)
        # 現状は全 op に命令列がある。将来 DSL に op を足して忘れたらここが破れる。
        assert not gaps, f"{t}: codegen 未対応 op {gaps}"


def test_every_codegen_op_assembles_on_both_vendors():
    """op 単位でも両社のアセンブラが受理する（カーネル全体に紛れて見落とさない）。"""
    for kind in sorted(CODEGEN_OPS):
        for t in TARGETS:
            if toolchain(t) is None:
                continue
            r = probe_op(kind, target=t)
            assert r.ok is True, f"{kind} @ {t}/{r.arch}: {r.stderr[:300]}"


def test_approximate_ops_match_the_instructions_actually_emitted():
    """ビット同一でないと分類した op は、実際に近似命令を出していること。

    分類を散文で持つだけでは腐る。生成テキストと突き合わせて固定する。
    """
    assert set(BIT_EXACT_ACROSS_VENDORS) == set(CODEGEN_OPS), (
        set(BIT_EXACT_ACROSS_VENDORS) ^ set(CODEGEN_OPS))
    assert approximate_ops() == {"sqrt", "rsqrt", "exp", "div", "dot", "reduce"}
    ptx = emit(_ir(_softmaxish), target="nvidia").text
    gcn = emit(_ir(_softmaxish), target="amd_cdna").text
    # exp: 双方とも 2^x 近似命令（ゆえにビット同一を期待できない）
    assert "ex2.approx.f32" in ptx and "v_exp_f32" in gcn
    # div: NVIDIA は正確丸め命令 / AMD は逆数近似 → 非対称が分類の根拠
    assert "div.rn.f32" in ptx and "v_rcp_f32" in gcn
    # 逆向き: ビット同一と分類した add は双方とも IEEE 加算そのもの
    assert "add.f32" in ptx and "v_add_f32" in gcn


def test_unstitched_layouts_are_declared_not_hidden():
    """行列コア／クロスレーンのレイアウト未接合を黙らず注記する。"""
    em = emit(_ir(), target="nvidia")
    assert em.unstitched, "dot を含むのに layout-unstitched が無い"
    assert all(n.startswith("layout-unstitched") for n in em.unstitched)
    em2 = emit(_ir(_softmaxish), target="amd_cdna")
    assert any("reduce" in n for n in em2.unstitched)


def test_encoding_roundtrip_confirms_intended_instructions_were_encoded():
    """受理と「意図どおり符号化された」は別。第二のツールで読み直して確かめる。

    この検査は実際に欠陥を見つけた: RDNA3 は同じ機械語を別ニーモニックで綴り
    （`global_load_dword` → `global_load_b32`）、llvm-mc は CDNA の綴りを *別名として
    黙って受理*していた。アセンブル成功だけでは気づけない。
    """
    mod = _ir()
    for t in TARGETS:
        enc = verify_encoding(mod, target=t)
        if not enc.available:
            assert enc.ok is None, "未検証を合格に丸めてはならない"
            continue
        assert enc.ok is True, f"{t}/{enc.arch}: 機械語に現れない命令 {enc.missing}"
        assert enc.intended, "意図した命令が拾えていない"
        # カーネル名が機械語オブジェクトのシンボルとして実在する
        assert any(k.name in enc.symbols for k in mod.kernels), enc.symbols
        if t == "nvidia":
            # SASS 逆アセンブラが無いので往復していない——それを自己申告する
            assert "nvdisasm" in enc.method and not enc.decoded
            assert enc.spill_bytes == 0, f"spill が出ている: {enc.spill_bytes} B"
        else:
            assert enc.method == "disasm-roundtrip" and enc.decoded


def test_rdna_uses_its_own_memory_mnemonics_not_the_cdna_aliases():
    """RDNA3 の綴りで出す（往復検証が silently-aliased を検出した件の回帰固定）。"""
    rdna = emit(_ir(), target="amd_rdna").text
    cdna = emit(_ir(), target="amd_cdna").text
    assert "global_load_b32" in rdna and "s_load_b128" in rdna
    assert "global_load_dword" not in rdna and "s_load_dwordx4" not in rdna
    assert "global_load_dword" in cdna and "s_load_dwordx4" in cdna


def test_encoding_check_detects_a_mnemonic_that_never_reaches_machine_code():
    """検査の有効性: 復号結果に無い命令を intended に混ぜたら missing に出る。"""
    from tsugi.codegen import _mnemonics
    got = _mnemonics("\tv_add_f32 v1, v2, v3\n\ts_endpgm\n", "amd_cdna")
    assert got == ["v_add_f32", "s_endpgm"], got
    # ディレクティブ・ラベル・コメントは命令として拾わない
    assert _mnemonics("\t.text\nk:\n// c\n\ts_endpgm\n", "amd_cdna") == ["s_endpgm"]


def test_L3_is_never_claimed():
    """実機実行検証（L3）はこの環境で到達不能。どの経路も L3 を返さない。"""
    mod = _ir()
    for t in TARGETS:
        _, asm = verify_codegen(mod, target=t)
        assert asm.level != VERIFY_LEVELS[3], "実行していないのに L3 を主張した"


def test_broken_assembly_is_rejected_not_waved_through():
    """アセンブラが真値である以上、壊れた入力は必ず落ちる（テストの有効性の担保）。"""
    for t, junk in (("nvidia", ".version 7.0\n.target sm_80\nnot_an_instruction;\n"),
                    ("amd_cdna", "\tv_this_does_not_exist v0, v1\n")):
        if toolchain(t) is None:
            continue
        r = assemble(junk, target=t)
        assert r.ok is False, f"{t}: 壊れたアセンブリが通ってしまった"


def test_compile_emits_machine_code_and_reports_its_level():
    """`tsugi.compile(..., emit_machine_code=True)` が実アセンブリまで到達する。"""
    import numpy as np
    a = np.zeros((64, 64), dtype=np.float16)
    b = np.zeros((64, 64), dtype=np.float16)
    c = np.zeros((64, 64), dtype=np.float32)
    args = (a, b, c, 64, 64, 64, 32, 32, 32)
    art = tsugi.compile(_kernel, args, target="nvidia", emit_machine_code=True)
    assert art.asm is not None and ".visible .entry" in art.asm
    expect = VERIFY_LEVELS[2] if toolchain("nvidia") is not None else VERIFY_LEVELS[1]
    assert art.level == expect, art.level
    # dry-run は生成しない（既定の振る舞いは変わっていない）
    dry = tsugi.compile(_kernel, args, target="nvidia")
    assert dry.asm is None and dry.level == VERIFY_LEVELS[0]
    # SPIR-V は codegen 未対応であることを明示的に告げる（黙って空を返さない）
    try:
        tsugi.compile(_kernel, args, target="spirv", emit_machine_code=True)
    except NotImplementedError as e:
        assert "SPIR-V" in str(e)
    else:
        raise AssertionError("spirv で NotImplementedError が出なかった")


def test_audit_surfaces_codegen_phase():
    """facade 到達性: audit の判定に codegen 層が載る（不変条件 57 の慣例）。"""
    ad = tsugi.audit(_ir(), block_dims=(32,))
    ph = [p for p in ad.phases if p.name.startswith("codegen")]
    assert len(ph) == 1, [p.name for p in ad.phases]
    txt = ph[0].to_text()
    assert "L3" in txt and "常に空" in txt, "L3 が空である旨の明示が消えた"
    if toolchain("nvidia") is not None:
        assert VERIFY_LEVELS[2] in txt
    else:
        assert VERIFY_LEVELS[1] in txt


def main() -> int:
    ok = True
    for t in (test_one_ir_emits_assembly_for_every_vendor_target,
              test_vendor_assemblers_accept_generated_code_or_we_do_not_claim_verified,
              test_transcendental_kernel_also_assembles,
              test_assembler_reports_arch_conditional_availability,
              test_every_codegen_op_is_a_real_dsl_op_and_gaps_are_reported,
              test_every_codegen_op_assembles_on_both_vendors,
              test_approximate_ops_match_the_instructions_actually_emitted,
              test_unstitched_layouts_are_declared_not_hidden,
              test_encoding_roundtrip_confirms_intended_instructions_were_encoded,
              test_rdna_uses_its_own_memory_mnemonics_not_the_cdna_aliases,
              test_encoding_check_detects_a_mnemonic_that_never_reaches_machine_code,
              test_L3_is_never_claimed,
              test_broken_assembly_is_rejected_not_waved_through,
              test_compile_emits_machine_code_and_reports_its_level,
              test_audit_surfaces_codegen_phase):
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    for t_ in TARGETS:
        tc = toolchain(t_)
        print(f"  toolchain[{t_}] = {tc or '未検出（L1-生成のみに落ちる）'}")
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
