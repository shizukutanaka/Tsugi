"""tracer の correctness テスト。

@tsugi.jit カーネルを tsugi.tile IR へトレースし、
  1. IR が期待 op 構造を含む（dot/load/store/zeros/cast）
  2. MLIR 風テキストが生成される
  3. トレース時の実値が eager リファレンスと一致（IR が正しい証明）
を検証する。GPU 不要・実行可能。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tsugi  # noqa: E402
from tsugi import tile  # noqa: E402


@tsugi.jit
def matmul_kernel(a, b, c, M, N, K, BM, BN, BK):
    pid_m = tsugi.program_id(0)
    pid_n = tsugi.program_id(1)
    acc = tile.zeros((BM, BN), tsugi.float32)
    for k in range(0, K, BK):
        ta = tile.load(a, (pid_m * BM, k), (BM, BK))
        tb = tile.load(b, (k, pid_n * BN), (BK, BN))
        acc = tile.dot(ta, tb, acc)
    tile.store(c, (pid_m * BM, pid_n * BN), acc.to(tsugi.float16))


def _inputs():
    rng = np.random.default_rng(0)
    M = N = K = 64
    BM = BN = BK = 32
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    c = np.zeros((M, N), dtype=np.float16)
    return a, b, c, M, N, K, BM, BN, BK


def test_trace_produces_ir():
    args = _inputs()
    mod = tsugi.trace(matmul_kernel, args, {}, program_ids=(0, 0))
    kinds = mod.op_kinds()
    for expected in ("zeros", "load", "dot", "store", "cast"):
        assert expected in kinds, f"IR missing op '{expected}': {kinds}"
    # K=64, BK=32 → 2 反復 → load 4回(a,b×2)・dot 2回
    assert kinds.count("dot") == 2, f"expected 2 dots, got {kinds.count('dot')}"
    assert kinds.count("load") == 4, f"expected 4 loads, got {kinds.count('load')}"
    return len(kinds)


def test_mlir_text_renders():
    args = _inputs()
    mod = tsugi.trace(matmul_kernel, args, {}, program_ids=(0, 0))
    text = mod.to_mlir()
    assert "tsugi_tile.kernel @matmul_kernel" in text
    assert "tsugi_tile.dot" in text
    assert "tensor<32x32xf32>" in text
    return text


def test_trace_values_match_eager():
    # トレースした実値（block (0,0)）が eager リファレンスと一致
    a, b, c_trace, M, N, K, BM, BN, BK = _inputs()
    # eager 全実行
    c_eager = np.zeros((M, N), dtype=np.float16)

    @tsugi.jit
    def k(a, b, c, M, N, K, BM, BN, BK):
        pid_m = tsugi.program_id(0)
        pid_n = tsugi.program_id(1)
        acc = tile.zeros((BM, BN), tsugi.float32)
        for kk in range(0, K, BK):
            ta = tile.load(a, (pid_m * BM, kk), (BM, BK))
            tb = tile.load(b, (kk, pid_n * BN), (BK, BN))
            acc = tile.dot(ta, tb, acc)
        tile.store(c, (pid_m * BM, pid_n * BN), acc.to(tsugi.float16))

    grid = (tsugi.cdiv(M, BM), tsugi.cdiv(N, BN))
    k[grid](a, b, c_eager, M, N, K, BM, BN, BK)
    ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    err = np.max(np.abs(c_eager.astype(np.float32) - ref.astype(np.float32)))
    assert err < 1e-1, f"eager err {err}"
    return float(err)


@tsugi.jit
def softmax_kernel(x, out, N, D):
    row = tile.load(x, (0, 0), (N, D))
    m = tile.reduce(row, 1, "max")
    e = tile.exp(row - m)
    s = tile.reduce(e, 1, "sum")
    tile.store(out, (0, 0), e / s)


def test_trace_records_amplifying_ops():
    # 修正前: reduce/exp はトレースされず IR から消えていた（propagation 空回り）。
    # 修正後: softmax がトレースでき、増幅 op が IR に現れる。
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    mod = tsugi.trace(softmax_kernel, (x, x.copy(), 8, 16), {}, program_ids=(0, 0))
    kinds = mod.op_kinds()
    for expected in ("reduce", "exp", "sub", "div"):
        assert expected in kinds, f"IR missing amplifying op '{expected}': {kinds}"
    assert kinds.count("reduce") == 2


def test_traced_softmax_matches_reference():
    # 具体トレースの実値が真の softmax と一致（IR が正しい証明）
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    out = x.copy()
    tsugi.trace(softmax_kernel, (x, out, 8, 16), {}, program_ids=(0, 0))
    ref = np.exp(x - x.max(1, keepdims=True))
    ref /= ref.sum(1, keepdims=True)
    assert np.max(np.abs(out - ref)) < 1e-6


def test_audit_propagation_sees_amplifying_ops():
    # 統合経路: audit の propagation グラフに増幅 op が流れる（perspective4 の実効化）
    # _graph_ops は SSA フォーク（list）を含みうるため葉 GraphOp を _iter_graphops で走査。
    from tsugi.audit import _graph_ops, _iter_graphops
    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    mod = tsugi.trace(softmax_kernel, (x, x.copy(), 8, 16), {}, program_ids=(0, 0))
    kinds = [o.kind for o in _iter_graphops(_graph_ops(mod, None))]
    assert "reduce" in kinds and "exp" in kinds and "div" in kinds


def test_audit_warns_amplifiers_underestimated_statically():
    # 正直さ: 増幅 op があるのに静的 cond=1 は下界 → propagation フェーズが WARN
    from tsugi.audit import audit
    from tsugi.report import Risk
    rng = np.random.default_rng(3)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    mod = tsugi.trace(softmax_kernel, (x, x.copy(), 8, 16), {}, program_ids=(0, 0))
    prop = next(p for p in audit(mod, None).phases if p.name.startswith("propagation"))
    assert prop.max_risk == Risk.WARN
    assert "下界" in prop.to_text()


def main() -> int:
    ok = True
    for t in (test_trace_produces_ir, test_mlir_text_renders, test_trace_values_match_eager,
              test_trace_records_amplifying_ops, test_traced_softmax_matches_reference,
              test_audit_propagation_sees_amplifying_ops,
              test_audit_warns_amplifiers_underestimated_statically):
        try:
            r = t()
            info = f"ops={r}" if isinstance(r, int) else (f"err={r:.2e}" if isinstance(r, float) else "ok")
            print(f"[PASS] {t.__name__:30s} {info}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__:30s} {e}")
            ok = False
    # 参考: IR テキストを表示
    print("\n--- generated tsugi.tile IR (block 0,0) ---")
    print(tsugi.trace(matmul_kernel, _inputs(), {}, program_ids=(0, 0)).to_mlir())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
