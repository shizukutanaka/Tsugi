"""検証器そのものの検証テスト（偽OK の非対称コストと検出限界）。

許容ベース等価判定の検出限界の下に隠れる系統バグ（偽OK）を実証し、
scale/K 不変な相補計量（RMS 比）がそれを捕まえることを確かめる。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tsugi.calibration import (  # noqa: E402
    check_systematic,
    detectability_floor,
    evaluate,
    is_equivalent_combined,
    make_corpus,
    systematic_divergence,
)
from tsugi.equivalence import compare_gemm, simulate_vendor_matmul  # noqa: E402


def test_floor_grows_with_K():
    # 検出限界 = safety·√K·u は K とともに拡大する（偽OK 盲点が広がる）
    f256 = detectability_floor(256, "float16")["rel"]
    f8192 = detectability_floor(8192, "float16")["rel"]
    assert f8192 > f256 * 4  # √(8192/256)=√32≈5.7 倍


def test_max_abs_misses_subfloor_scale_bug():
    # 0.5% 系統スケール誤差は max_abs（導出許容）では偽OK になる
    rng = np.random.default_rng(0)
    K = 2048
    a = rng.standard_normal((64, K)).astype(np.float16)
    b = rng.standard_normal((K, 64)).astype(np.float16)
    base = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    bug = base * 1.005
    assert compare_gemm(base, bug, K, "float16").equivalent  # 見逃す（偽OK）


def test_systematic_check_catches_what_max_abs_misses():
    rng = np.random.default_rng(0)
    K = 2048
    a = rng.standard_normal((64, K)).astype(np.float16)
    b = rng.standard_normal((K, 64)).astype(np.float16)
    base = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    bug = base * 1.005
    rep = check_systematic(base, bug, K, "float16")
    assert not rep.ok                     # 系統検査は捕まえる
    assert abs(rep.bias - 0.005) < 1e-4   # RMS 比が 0.5% を正しく測る


def test_systematic_check_passes_legitimate_order_divergence():
    # 累積順序だけ違う等価ベンダーは系統バイアス ~0 → 通過（偽BLOCK にしない）
    rng = np.random.default_rng(1)
    K = 2048
    a = rng.standard_normal((64, K)).astype(np.float16)
    b = rng.standard_normal((K, 64)).astype(np.float16)
    va = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    vb = simulate_vendor_matmul(a, b, accum="f32", split_k=8)
    assert check_systematic(va, vb, K, "float16").ok
    assert abs(systematic_divergence(va, vb)) < 1e-3


def test_combined_verifier_is_trustworthy_corpus():
    # 合成判定（max_abs + 系統）はコーパスで偽OK ゼロ＝信頼に足る
    corpus = make_corpus(seed=0)
    conf = evaluate(corpus, lambda a, b, K: is_equivalent_combined(a, b, K, "float16"))
    assert conf.trustworthy            # false_ok == 0
    assert conf.false_ok == 0


def test_max_abs_alone_is_untrustworthy_corpus():
    # max_abs 単独はコーパスで偽OK を出す＝信頼できない（検証器の検証）
    corpus = make_corpus(seed=0)
    conf = evaluate(corpus, lambda a, b, K: compare_gemm(a, b, K, "float16").equivalent)
    assert not conf.trustworthy
    assert "scale" in conf.missed      # 系統スケールバグを見逃す


def main() -> int:
    ok = True
    tests = [
        test_floor_grows_with_K,
        test_max_abs_misses_subfloor_scale_bug,
        test_systematic_check_catches_what_max_abs_misses,
        test_systematic_check_passes_legitimate_order_divergence,
        test_combined_verifier_is_trustworthy_corpus,
        test_max_abs_alone_is_untrustworthy_corpus,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: 検出限界が K で拡大する様子と相補計量の効果
    print("\n--- max_abs 検出限界 vs 系統検査（0.5% スケールバグ）---")
    corpus = make_corpus(seed=0)
    c_alone = evaluate(corpus, lambda a, b, K: compare_gemm(a, b, K, "float16").equivalent)
    c_comb = evaluate(corpus, lambda a, b, K: is_equivalent_combined(a, b, K, "float16"))
    print("  max_abs 単独 : " + c_alone.to_text().replace("\n", "\n  "))
    print("  合成(相補)   : " + c_comb.to_text().replace("\n", "\n  "))
    for K in (256, 2048, 8192):
        print(f"  floor@K={K:<5} = {detectability_floor(K, 'float16')['rel'] * 100:.1f}%")
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
