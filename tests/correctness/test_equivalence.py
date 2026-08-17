"""equivalence 検出器のテスト。擬似ベンダーで数値発散を*捕まえる*ことを実証。

これは GPU シミュレーション（CPU）。実 GPU では同じ compare() を実機出力に適用する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.equivalence import (  # noqa: E402
    DV_DIVERGENT,
    DV_EQUIVALENT,
    DV_LAYOUT,
    classify_divergence,
    compare,
    input_precision_divergence,
    precision_policy_hint,
    simulate_vendor_matmul,
    truncate_to_tensorcore,
)


def _inputs(M=128, N=128, K=512):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    return a, b


def test_identical_is_equivalent():
    a, b = _inputs()
    ref = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    rep = compare(ref, ref.copy(), dtype="float16")
    assert rep.equivalent, rep.to_text()


def test_f32_vs_f32_within_tolerance():
    # 同じ f32 累積・分割違いは微差 → 等価のはず
    a, b = _inputs()
    v1 = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    v2 = simulate_vendor_matmul(a, b, accum="f32", split_k=4)
    rep = compare(v1, v2, dtype="float16")
    assert rep.equivalent, f"f32 split should be equivalent: {rep.to_text()}"


def test_f16_accum_detected_as_divergent():
    # fp16 累積は大 K で誤差が膨らむ → DIVERGENT を検出できるべき
    a, b = _inputs(K=2048)
    good = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
    bad = simulate_vendor_matmul(a, b, accum="f16", split_k=64)
    rep = compare(good, bad, dtype="float16")
    # 検出器が機能 = ズレを equivalent と誤判定しない
    assert not rep.equivalent, f"divergence not caught: {rep.to_text()}"
    assert rep.max_abs_err > 1e-2


def test_shape_mismatch_is_not_silently_broadcast():
    """compare() は形状不一致を NumPy の暗黙 broadcast に委ねず即 DIVERGENT にする。

    素朴な element-wise 比較（NaN/Inf 検出後の a-b）は shape が違っても NumPy の
    broadcast ルールが適用可能なら暗黙に実行されてしまう。方向次第で偽 DIVERGENT にも
    偽 OK にもなりうる —— 後者が特に危険（発散を等価と誤判定）。

    ここでは「vendor B が実装バグで先頭 1 行しか返さない」という現実的なシナリオを
    実証する: (8,8) の a に対し b=a[0]（形状 (8,)）を渡すと、NumPy の broadcast は
    a の全行と b を比較し、a の各行が b（=a の 0 行目）と一致していなくても
    element-wise 比較自体は「実行できてしまう」——broadcast された a-b の結果次第では
    誤って equivalent と判定されうる。shape_mismatch により比較そのものを拒否すべき。
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 8)).astype(np.float32)
    b_first_row_only = a[0].copy()   # vendor B のバグ: 先頭行しか返さない（shape (8,)）

    rep = compare(a, b_first_row_only, "float32")
    assert rep.shape_mismatch, f"形状不一致が検出されていない: {rep.to_text()}"
    assert not rep.equivalent, "形状不一致を equivalent=True にしてはいけない（偽OK）"
    assert rep.shape_a == (8, 8) and rep.shape_b == (8,)
    assert "SHAPE MISMATCH" in rep.to_text()

    # broadcast可能でない完全に非互換な形状でもクラッシュせず shape_mismatch を返す
    rep2 = compare(a, rng.standard_normal((3, 5)).astype(np.float32), "float32")
    assert rep2.shape_mismatch and not rep2.equivalent

    # 同形状の通常経路は無回帰（shape_mismatch=False のまま）
    rep3 = compare(a, a.copy(), "float32")
    assert rep3.equivalent and not rep3.shape_mismatch


def test_report_fields():
    a, b = _inputs()
    ref = simulate_vendor_matmul(a, b)
    rep = compare(ref, ref.copy(), "float16")
    assert rep.n_mismatch == 0
    assert rep.n_total == ref.size


def test_uniform_risk_interface():
    # EquivalenceReport も他レポートと同じ risk/max_risk/ok を持つ（検証層の統一）
    from tsugi.report import Risk
    a, b = _inputs()
    ref = simulate_vendor_matmul(a, b)
    eq = compare(ref, ref.copy(), "float16")
    assert eq.ok and eq.risk is Risk.OK and eq.max_risk is Risk.OK
    dv = compare(ref, ref + 10.0, "float16")
    assert not dv.ok and dv.risk is Risk.BLOCK


def test_classify_layout_vs_numerical_divergence():
    # element-wise 不一致を レイアウト不一致(値正しい) vs 真の数値発散 に区別
    rng = np.random.default_rng(0)
    K = 256
    a = rng.standard_normal((64, 64)).astype(np.float32)
    assert classify_divergence(a, a.copy(), K) == DV_EQUIVALENT
    # 転置・置換は値の多重集合を保存 → LAYOUT（数値は正しい・整列バグ）
    assert classify_divergence(a, a.T.copy(), K) == DV_LAYOUT
    shuffled = rng.permutation(a.reshape(-1)).reshape(64, 64).astype(np.float32)
    assert classify_divergence(a, shuffled, K) == DV_LAYOUT
    # 真のスケール発散は multiset も崩す → DIVERGENT
    assert classify_divergence(a, (a * 1.5).astype(np.float32), K) == DV_DIVERGENT
    # 要素数が違えば layout ではありえない → DIVERGENT
    assert classify_divergence(a, a[:, :32].copy(), K) == DV_DIVERGENT


def test_float64_does_not_fall_back_to_float32_tolerance():
    """float64 比較が float32 の緩い許容(1e-4)にフォールバックしないこと（偽OK 防止）。

    研究知見（PyTorch assert_close は float64=atol 1e-8）に基づく修正の回帰テスト。
    倍精度で 1e-6 ずれた 2 実装は、float32 許容(1e-4)では「等価」と誤判定されるが、
    float64 専用許容(1e-7)では正しく DIVERGENT を検出すべき。
    """
    base = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    perturbed = base + 1e-6   # float32 tol(1e-4)では隠れる / float64 tol(1e-7)では見える
    rep64 = compare(base, perturbed, dtype="float64")
    assert not rep64.equivalent, (
        f"float64 で 1e-6 のズレを見逃した（偽OK）: {rep64.to_text()}")
    # 対照: 同じズレを float32 許容で見ると（緩いので）等価扱い = フォールバックの危険性
    rep32 = compare(base, perturbed, dtype="float32")
    assert rep32.equivalent, "対照: float32 許容(1e-4)では 1e-6 は等価扱い（だから fallback は危険）"


def test_float64_accepts_genuine_double_precision_noise():
    """float64 で真の倍精度丸め(~1e-15)は等価と判定する（偽BLOCK を出さない）。"""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(100).astype(np.float64)
    noise = base + rng.standard_normal(100) * 1e-14  # 倍精度丸め級
    rep = compare(base, noise, dtype="float64")
    assert rep.equivalent, f"倍精度丸めを過剰検出（偽BLOCK）: {rep.to_text()}"


def test_nan_in_output_flagged_as_non_finite():
    """NaN を含む出力は has_non_finite=True かつ DIVERGENT と判定（データ破壊の識別）。

    精度発散（finite な差）と NaN 伝播（データ破壊）は根本原因が異なる。
    前者はアルゴリズム精度問題、後者は overflow/除零/入力破損。
    has_non_finite フラグでスタックトレースなしに根本原因を絞り込める。
    """
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_nan = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    rep = compare(a, b_nan, dtype="float32")
    assert not rep.equivalent, "NaN を含む出力が等価と誤判定された"
    assert rep.has_non_finite, "NaN が has_non_finite=False のまま（識別できていない）"
    assert "NaN/Inf" in rep.to_text(), f"to_text() に NaN/Inf が表示されない: {rep.to_text()}"

    # 双方に Inf を含む場合も非有限を識別できる
    b_inf = np.array([1.0, np.inf, 3.0], dtype=np.float32)
    rep_inf = compare(a, b_inf, dtype="float32")
    assert rep_inf.has_non_finite
    assert not rep_inf.equivalent

    # 有限同士の差は has_non_finite=False
    rep_finite = compare(a, a + 0.5, dtype="float32")
    assert not rep_finite.has_non_finite, "有限の差で has_non_finite=True（偽陽性）"


def test_tf32_tolerance_matches_float16():
    """TF32 dtype の許容誤差が float16 と同等（10-bit 仮数 → fp16 精度）。

    NVIDIA Ampere+ の float32 GEMM/conv は TF32 Tensor Core を使い仮数が 10 bit（fp16 と同等）。
    AMD ROCm は TF32 非対応 → NVIDIA vs AMD 比較では dtype="tf32" で fp16 級許容が必要。
    """
    from tsugi.equivalence import TOLERANCE
    tol_tf32 = TOLERANCE["tf32"]
    tol_f16 = TOLERANCE["float16"]
    assert tol_tf32["atol"] == tol_f16["atol"], (
        f"TF32 atol={tol_tf32['atol']} ≠ float16 atol={tol_f16['atol']}")
    assert tol_tf32["rtol"] == tol_f16["rtol"]

    # dtype="tf32" で compare が動作するか（KeyError が起きない）
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = a + 5e-3   # float32 許容(1e-4) では DIVERGENT / tf32 許容(1e-2) では EQUIVALENT
    rep_tf32 = compare(a, b, dtype="tf32")
    rep_f32 = compare(a, b, dtype="float32")
    assert rep_tf32.equivalent, f"TF32 許容内(5e-3)を DIVERGENT と誤検出: {rep_tf32.to_text()}"
    assert not rep_f32.equivalent, "float32 許容(1e-4)で 5e-3 差が等価（想定外）"


def test_tf32_unit_roundoff_matches_float16():
    """TF32 の unit_roundoff が float16 と同等（仮数 10 bit, u = 2^-11）。"""
    from tsugi.tolerance import UNIT_ROUNDOFF
    assert UNIT_ROUNDOFF["tf32"] == UNIT_ROUNDOFF["float16"], (
        f"TF32 unit_roundoff={UNIT_ROUNDOFF['tf32']} ≠ float16={UNIT_ROUNDOFF['float16']}")


def test_fp8_tolerance_is_coarser_than_fp16():
    """FP8 (E4M3/E5M2) の許容が fp16 より粗い（仮数 2〜3 bit で丸めが巨大）。

    H100/MI300/B200 推論で主流の FP8 は、クロスベンダーでは per-tensor amax スケール差も
    乗るため大幅に緩い許容が必要。E5M2(仮数 2bit) は E4M3(3bit) よりさらに粗い。
    """
    from tsugi.equivalence import TOLERANCE
    from tsugi.tolerance import UNIT_ROUNDOFF
    assert TOLERANCE["float8_e4m3"]["atol"] > TOLERANCE["float16"]["atol"], "E4M3 が fp16 より厳しい"
    assert TOLERANCE["float8_e5m2"]["atol"] > TOLERANCE["float8_e4m3"]["atol"], "E5M2 が E4M3 より厳しい"
    # unit roundoff も仮数ビット順（e5m2 が最も粗い）
    assert UNIT_ROUNDOFF["float8_e5m2"] > UNIT_ROUNDOFF["float8_e4m3"] > UNIT_ROUNDOFF["float16"]
    assert UNIT_ROUNDOFF["float8_e4m3"] == 2.0 ** -4   # 3 仮数ビット
    assert UNIT_ROUNDOFF["float8_e5m2"] == 2.0 ** -3   # 2 仮数ビット


def test_fp8_e4m3_catches_real_divergence_but_accepts_quantization_noise():
    """E4M3 許容が量子化級ノイズは許し、真の発散は捕まえる（fail-safe の両立）。"""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    # E4M3 の u=0.0625 級のノイズ（量子化由来）は等価扱い
    rep_noise = compare(a, a + 0.03, dtype="float8_e4m3")
    assert rep_noise.equivalent, f"FP8 量子化級ノイズを過剰検出（偽BLOCK）: {rep_noise.to_text()}"
    # 0.5 のスケール発散（許容 1e-1 超）は DIVERGENT
    rep_div = compare(a, a + 0.5, dtype="float8_e4m3")
    assert not rep_div.equivalent, "FP8 で真の発散(0.5)を見逃した（偽OK）"


def test_mxfp_tolerance_ordering_matches_mantissa_bits():
    """MXFP4/MXFP6 の許容が仮数ビット数の順に粗い（mxfp4 が全 dtype 中最粗）。

    OCP MX v1.0: MXFP4=E2M1(仮数1bit)・MXFP6=E3M2(仮数2bit)/E2M3(仮数3bit)。
    NVIDIA Blackwell と AMD CDNA4 の両方が HW ネイティブ対応する共通フォーマット群。
    """
    from tsugi.equivalence import TOLERANCE
    from tsugi.tolerance import UNIT_ROUNDOFF
    assert TOLERANCE["mxfp4_e2m1"]["atol"] > TOLERANCE["mxfp6_e3m2"]["atol"] > TOLERANCE["mxfp6_e2m3"]["atol"], (
        "MXFP tolerance が仮数ビット順(4>e3m2>e2m3)でない")
    assert TOLERANCE["mxfp4_e2m1"]["atol"] > TOLERANCE["float8_e5m2"]["atol"], (
        "mxfp4(1仮数bit)が fp8_e5m2(2仮数bit)より厳しい")
    assert UNIT_ROUNDOFF["mxfp4_e2m1"] > UNIT_ROUNDOFF["mxfp6_e3m2"] > UNIT_ROUNDOFF["mxfp6_e2m3"]
    assert UNIT_ROUNDOFF["mxfp4_e2m1"] == 2.0 ** -1   # 1 仮数ビット（全 dtype 中最粗）
    assert UNIT_ROUNDOFF["mxfp6_e3m2"] == 2.0 ** -2   # 2 仮数ビット
    assert UNIT_ROUNDOFF["mxfp6_e2m3"] == 2.0 ** -3   # 3 仮数ビット


def test_mxfp4_catches_real_divergence_but_accepts_quantization_noise():
    """MXFP4(E2M1) 許容が量子化級ノイズは許し、真の発散は捕まえる（fail-safe の両立）。"""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    # 1 仮数ビット級の量子化ノイズ（±0.3 程度）は等価扱い
    rep_noise = compare(a, a + 0.3, dtype="mxfp4_e2m1")
    assert rep_noise.equivalent, f"MXFP4 量子化級ノイズを過剰検出（偽BLOCK）: {rep_noise.to_text()}"
    # 10 倍のスケール発散は DIVERGENT（±0.3 では許容内に収まりうるほど粗いため強い発散を使う）
    rep_div = compare(a, a * 10.0, dtype="mxfp4_e2m1")
    assert not rep_div.equivalent, "MXFP4 で真の発散(×10)を見逃した（偽OK）"


# --- テンサーコア入力精度（TF32）をクロス発散源としてモデル化（累積順序差とは別源） ---

def test_tensorcore_truncation_matches_format_spec():
    """TF32/bf16 の入力仮数 truncation の相対誤差がフォーマット定義に一致する。"""
    rng = np.random.default_rng(0)
    v = (rng.standard_normal(100000).astype(np.float32) * 10)
    for prec, mant in (("tf32", 10), ("bf16", 7)):
        t = truncate_to_tensorcore(v, prec)
        rel = np.abs((t - v) / v)
        u = 2.0 ** -(mant + 1)
        assert rel.max() <= u + 1e-9, f"{prec}: max rel {rel.max():.2e} > u=2^-{mant+1}={u:.2e}"
        assert rel.max() > u * 0.5     # 実際に丸めている（no-op でない）
    # ieee は無改変
    assert np.array_equal(truncate_to_tensorcore(v, "ieee"), v.astype(np.float32))


def test_input_precision_divergence_is_flat_in_K_unlike_accumulation():
    """入力精度発散は K 非依存（~u）——累積順序差（√K·u）と *別源* であることを実証。

    入力仮数の丸めは各要素の相対摂動で、和をとっても相対発散は ~u のまま K に依らない。
    これは累積順序差（各累積ステップの丸めが √K で増える）と質的に異なる。
    """
    rng = np.random.default_rng(0)
    divs = []
    for K in (256, 2048, 8192):
        a = rng.standard_normal((64, K)).astype(np.float32)
        b = rng.standard_normal((K, 64)).astype(np.float32)
        ieee = simulate_vendor_matmul(a, b)
        tf32 = simulate_vendor_matmul(a, b, input_precision="tf32")
        r = float(np.linalg.norm(tf32 - ieee) / np.linalg.norm(ieee))
        divs.append(r)
        assert r <= input_precision_divergence("tf32"), "予測上界を超えた"
    # K が 32 倍になっても発散はほぼ一定（√K なら 5.6 倍になるはず）
    assert max(divs) / min(divs) < 1.5, f"K 依存が見える（flat でない）: {divs}"
    # bf16 は tf32 より大きい（仮数が少ない）
    a = rng.standard_normal((64, 2048)).astype(np.float32)
    b = rng.standard_normal((2048, 64)).astype(np.float32)
    ieee = simulate_vendor_matmul(a, b)
    d_tf32 = np.linalg.norm(simulate_vendor_matmul(a, b, input_precision="tf32") - ieee)
    d_bf16 = np.linalg.norm(simulate_vendor_matmul(a, b, input_precision="bf16") - ieee)
    assert d_bf16 > d_tf32


def test_simulate_vendor_matmul_input_precision_is_backward_compatible():
    """input_precision 既定（ieee）は従来と完全に同一（回帰なし）。"""
    rng = np.random.default_rng(1)
    a = rng.standard_normal((32, 512)).astype(np.float16)
    b = rng.standard_normal((512, 32)).astype(np.float16)
    assert np.array_equal(simulate_vendor_matmul(a, b, accum="f32", split_k=8),
                          simulate_vendor_matmul(a, b, accum="f32", split_k=8,
                                                 input_precision="ieee"))


def test_precision_policy_hint_discriminates_tf32_from_bug_and_noise():
    """fp32 の TF32-vs-IEEE 発散だけを兆候として拾い、バグ/ノイズ/非fp32 では黙る。"""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 2048)).astype(np.float32)
    b = rng.standard_normal((2048, 64)).astype(np.float32)
    ieee = simulate_vendor_matmul(a, b)
    tf32 = simulate_vendor_matmul(a, b, input_precision="tf32")
    # TF32 精度差 → 兆候を出す
    assert precision_policy_hint(ieee, tf32, 2048, "float32") is not None
    # 同一 → 黙る（発散なし）
    assert precision_policy_hint(ieee, ieee, 2048, "float32") is None
    # 粗いバグ（1% スケール）→ 黙る（TF32 帯より大きい＝本物の発散）
    assert precision_policy_hint(ieee, ieee * 1.01, 2048, "float32") is None
    # fp32 累積順序差のみ（TF32 帯より小さい）→ 黙る
    accum = simulate_vendor_matmul(a, b, split_k=8)
    assert precision_policy_hint(ieee, accum, 2048, "float32") is None
    # 非 fp32 系（fp16）→ そもそも TF32 の話でないので黙る
    assert precision_policy_hint(ieee, tf32, 2048, "float16") is None


def main() -> int:
    ok = True
    tests = [
        test_identical_is_equivalent,
        test_f32_vs_f32_within_tolerance,
        test_f16_accum_detected_as_divergent,
        test_shape_mismatch_is_not_silently_broadcast,
        test_report_fields,
        test_uniform_risk_interface,
        test_classify_layout_vs_numerical_divergence,
        test_float64_does_not_fall_back_to_float32_tolerance,
        test_float64_accepts_genuine_double_precision_noise,
        test_nan_in_output_flagged_as_non_finite,
        test_tf32_tolerance_matches_float16,
        test_tf32_unit_roundoff_matches_float16,
        test_fp8_tolerance_is_coarser_than_fp16,
        test_fp8_e4m3_catches_real_divergence_but_accepts_quantization_noise,
        test_mxfp_tolerance_ordering_matches_mantissa_bits,
        test_mxfp4_catches_real_divergence_but_accepts_quantization_noise,
        test_tensorcore_truncation_matches_format_spec,
        test_input_precision_divergence_is_flat_in_K_unlike_accumulation,
        test_simulate_vendor_matmul_input_precision_is_backward_compatible,
        test_precision_policy_hint_discriminates_tf32_from_bug_and_noise,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: 擬似 NVIDIA(f32) vs 擬似 AMD(f16累積) の照合
    a, b = _inputs(K=2048)
    nv = simulate_vendor_matmul(a, b, accum="f32")
    amd = simulate_vendor_matmul(a, b, accum="f16", split_k=64)
    print("\n--- 擬似ベンダー等価性照合（K=2048）---")
    print("nvidia(f32) vs amd(f16累積):", compare(nv, amd, "float16").to_text())
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
