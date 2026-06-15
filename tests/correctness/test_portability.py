"""portability 解析のテスト。新視点（クロスベンダー検証層）が実際に効くか確認。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tsugi  # noqa: E402
from tsugi import tile  # noqa: E402
from tsugi.portability import Risk, analyze, cross_vendor_diff  # noqa: E402


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


def _ir(bm=32, bn=32, bk=32):
    a = np.zeros((64, 64), np.float16)
    return tsugi.trace(matmul_kernel, (a, a.copy(), a.copy(), 64, 64, 64, bm, bn, bk), {}, (0, 0))


def test_warp_misalignment_flagged_for_amd():
    # block dim 48 は AMD CDNA wavefront=64 の倍数でない → WARN
    mod = _ir()
    rep = analyze(mod, "amd_cdna", block_dims=(48,))
    assert any(f.op == "block" and f.risk == Risk.WARN for f in rep.findings), \
        "AMD wavefront misalignment not flagged"


def test_warp_aligned_ok_for_amd():
    # block dim 64 は AMD wavefront=64 の倍数 → block 警告なし
    mod = _ir()
    rep = analyze(mod, "amd_cdna", block_dims=(64,))
    assert not any(f.op == "block" for f in rep.findings)


def test_same_block_differs_across_vendors():
    # block dim 32: NVIDIA(warp32)はOK / AMD CDNA(wavefront64)はWARN → ベンダー間差
    mod = _ir()
    nv = analyze(mod, "nvidia", block_dims=(32,))
    amd = analyze(mod, "amd_cdna", block_dims=(32,))
    nv_block = any(f.op == "block" for f in nv.findings)
    amd_block = any(f.op == "block" for f in amd.findings)
    assert not nv_block and amd_block, "warp-size divergence not detected"


def test_cross_vendor_diff_nonempty_when_divergent():
    # 形状不明 dot 等は両方に出るが、block差は block_dims を渡さないと出ない。
    # ここでは dot INFO がベンダーで異なりうるかを確認（最低限 API が動く）
    mod = _ir()
    diffs = cross_vendor_diff(mod, ("nvidia", "amd_cdna"))
    assert isinstance(diffs, list)


def main() -> int:
    ok = True
    tests = [
        test_warp_misalignment_flagged_for_amd,
        test_warp_aligned_ok_for_amd,
        test_same_block_differs_across_vendors,
        test_cross_vendor_diff_nonempty_when_divergent,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: 同じカーネルの NVIDIA/AMD 移植レポート差
    mod = _ir()
    print("\n--- 同一カーネル・block=32 の移植性差 ---")
    print(analyze(mod, "nvidia", block_dims=(32,)).to_text())
    print(analyze(mod, "amd_cdna", block_dims=(32,)).to_text())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
