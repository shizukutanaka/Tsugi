"""数値エンベロープ実行時検査のテスト（新視点・第5ラウンド）。

静的に認証した等価性が、本番入力のエンベロープ逸脱で無効化されることを実証。
oracle も第2ベンダーも使わず単一ベンダーの統計だけで危険を捕まえる。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tsugi.envelope import (  # noqa: E402
    certify_from_sample,
    certify_gemm,
    channel_scale_spread,
    check_outlier_features,
    check_softmax_input,
    check_tensor,
    dtype_limits,
)
from tsugi.portability import Risk  # noqa: E402


def test_in_envelope_passes():
    env = certify_gemm(K=256, dtype="float16", scale=1.0)
    x = np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32)
    rep = check_tensor(x, env)
    assert rep.in_envelope, rep.to_text()


def test_fp16_overflow_is_block():
    env = certify_gemm(K=256, dtype="float16", scale=1.0)
    x = np.full((4, 4), 70000.0, dtype=np.float32)   # > fp16 max 65504
    rep = check_tensor(x, env)
    assert not rep.in_envelope
    assert rep.max_risk == Risk.BLOCK


def test_scale_exceedance_voids_certification():
    # 認証は scale=1 で atol を保証。本番スケールが大きいと認証 atol は無効。
    env = certify_gemm(K=256, dtype="float16", scale=1.0)
    x = np.random.default_rng(1).standard_normal((64, 64)).astype(np.float32) * 50.0
    rep = check_tensor(x, env)
    assert rep.max_risk == Risk.BLOCK
    assert any("再認証" in f.message for f in rep.findings)


def test_denormal_flagged_for_ftz_divergence():
    env = certify_gemm(K=64, dtype="float16", scale=1.0)
    lim = dtype_limits("float16")
    x = np.zeros((8, 8), dtype=np.float32)
    x[0, 0] = lim.min_normal * 0.1   # denormal 域
    x[0, 1] = 1.0                    # スケールは正常に保つ
    rep = check_tensor(x, env)
    assert any("denormal" in f.message or "FTZ" in f.message for f in rep.findings)


def test_envelope_thresholds_are_sensitive_to_their_constants():
    """SOCRATIC-50 Q4: envelope の閾値定数（`_OVERFLOW_WARN_FRAC`・`_SCALE_BLOCK_RATIO`・
    `_EXP_WARN_FRAC`）が判定境界を *実際に* 支配することを境界±で固定する
    （Q6 の `SAFETY` 感度テストと同型・silent drift の番人）。
    """
    from tsugi.envelope import _EXP_WARN_FRAC, _OVERFLOW_WARN_FRAC, _SCALE_BLOCK_RATIO

    # --- _OVERFLOW_WARN_FRAC: max|x| が dtype 上限の何割で overflow 近接 WARN か ---
    lim16 = dtype_limits("float16")
    env_big_scale = certify_gemm(K=64, dtype="float16", scale=1000.0)   # scale 系の副作用を避ける余裕

    def overflow_near_warned(mult: float) -> bool:
        x = np.full((10000,), 1.0, dtype=np.float32)   # RMS を低く保つ多数派
        x[0] = _OVERFLOW_WARN_FRAC * lim16.max_normal * mult   # 単一の外れ値で max_abs を制御
        rep = check_tensor(x, env_big_scale)
        return any("overflow 近接" in f.message for f in rep.findings)

    assert overflow_near_warned(1.01), "閾値直上で overflow 近接 WARN が出ない"
    assert not overflow_near_warned(0.99), "閾値直下で overflow 近接 WARN が誤って出ている"

    # --- _SCALE_BLOCK_RATIO: 認証 scale_max の何倍を超えたら BLOCK か ---
    env = certify_gemm(K=64, dtype="float32", scale=1.0)

    def scale_risk(mult: float) -> Risk:
        x = np.full((100, 100), env.scale_max * _SCALE_BLOCK_RATIO * mult, dtype=np.float32)
        return check_tensor(x, env).max_risk

    assert scale_risk(1.01) == Risk.BLOCK, "閾値直上で scale BLOCK にならない"
    assert scale_risk(0.99) == Risk.WARN, "閾値直下で scale WARN にならない"

    # --- _EXP_WARN_FRAC: exp-overflow 閾値の何割で softmax 近接 WARN か ---
    lim32 = dtype_limits("float32")
    env32 = certify_gemm(K=64, dtype="float32", scale=1.0)

    def softmax_risk(mult: float) -> Risk:
        logit = np.array([[_EXP_WARN_FRAC * lim32.exp_overflow * mult]], dtype=np.float32)
        return check_softmax_input(logit, env32).max_risk

    assert softmax_risk(1.01) == Risk.WARN, "閾値直上で softmax 近接 WARN にならない"
    assert softmax_risk(0.99) == Risk.OK, "閾値直下で誤って WARN になっている"


def test_nan_is_block():
    env = certify_gemm(K=64, dtype="float16", scale=1.0)
    x = np.array([[1.0, np.nan]], dtype=np.float32)
    rep = check_tensor(x, env)
    assert rep.max_risk == Risk.BLOCK


def test_fp16_softmax_logit_overflow():
    # fp16 で生 logit が ln(65504)≈11.09 を超えると exp が inf → 片ベンダーで破綻
    env = certify_gemm(K=128, dtype="float16", scale=1.0)
    logits = np.array([[0.0, 12.5, 3.0]], dtype=np.float32)
    rep = check_softmax_input(logits, env)
    assert rep.max_risk == Risk.BLOCK
    # 同じ logit でも bf16/f32 は範囲が広く OK（dtype 依存の差を実証）
    env32 = certify_gemm(K=128, dtype="float32", scale=1.0)
    assert check_softmax_input(logits, env32).in_envelope


def test_real_fp16_overflow_actually_happens():
    # 検査が机上でなく実挙動: exp(12.5) を fp16 で計算すると本当に inf になる
    lim = dtype_limits("float16")
    with np.errstate(over="ignore"):
        assert np.isinf(np.exp(np.float16(12.5)))     # 12.5 > 11.09 → inf
        assert not np.isinf(np.exp(np.float16(10.0)))  # 10.0 < 11.09 → 有限
    assert 11.0 < lim.exp_overflow < 11.2


def test_float64_dtype_limits_are_correct():
    """float64 の dtype_limits が float32 にフォールバックしないこと。

    float64 max ≈ 1.8e308、float32 max ≈ 3.4e38 — 270 桁違う。
    float64 テンソルを float32 limits で検査すると、float64 正常値が overflow 判定される偽BLOCK。
    """
    lim64 = dtype_limits("float64")
    lim32 = dtype_limits("float32")
    assert lim64.max_normal > lim32.max_normal * 1e260, (
        f"float64 max_normal={lim64.max_normal:.3g} が float32 ({lim32.max_normal:.3g}) と同じ"
        "（フォールバック疑い）")
    # float64 正常値 (1e100) が float64 limits では overflow にならない
    env64 = certify_gemm(K=64, dtype="float64", scale=1.0)
    x64 = np.full((4, 4), 1e100, dtype=np.float64)
    rep = check_tensor(x64, env64)
    # scale 超過 WARN はあっても overflow BLOCK は出ないはず
    overflow_block = any("overflow" in f.message and f.risk == Risk.BLOCK for f in rep.findings)
    assert not overflow_block, (
        f"float64 で 1e100 が overflow BLOCK（float32 limits にフォールバックしている疑い）: {rep.to_text()}")


def test_tf32_dtype_limits_match_float32():
    """TF32 の dtype_limits が float32 と同等（TF32 は fp32 指数部 → overflow リスク同じ）。"""
    lim_tf32 = dtype_limits("tf32")
    lim_f32 = dtype_limits("float32")
    assert lim_tf32.max_normal == lim_f32.max_normal, (
        f"TF32 max_normal={lim_tf32.max_normal} ≠ float32 max_normal={lim_f32.max_normal}")
    assert lim_tf32.exp_overflow == lim_f32.exp_overflow


def test_fp8_e4m3_narrow_range_makes_overflow_the_main_risk():
    """FP8 E4M3 は max=448 と極端に狭く、小さな値でも overflow する（amax スケール必須の理由）。

    H100/MI300 推論主流の FP8 は値域が狭いため、未スケールの活性（例 1000）が即 overflow。
    エンベロープ検査が FP8 で特に重要であることを実証。
    """
    lim = dtype_limits("float8_e4m3")
    assert lim.max_normal == 448.0
    env = certify_gemm(K=64, dtype="float8_e4m3", scale=1.0)
    # fp16 なら余裕(65504)だが E4M3 では 1000 が overflow
    x = np.full((4, 4), 1000.0, dtype=np.float32)
    rep = check_tensor(x, env)
    assert not rep.in_envelope, "E4M3 で 1000 が overflow にならない（max=448 のはず）"
    assert rep.max_risk == Risk.BLOCK
    # 同じ 1000 は fp16(max=65504) では overflow しない → dtype 依存の overflow リスク差を実証
    fp16_findings = check_tensor(x, certify_gemm(K=64, dtype="float16", scale=1000.0)).findings
    assert not any("overflow" in f.message for f in fp16_findings), \
        "fp16(max=65504) で 1000 が overflow 判定された（想定外）"


def test_fp8_e5m2_wider_range_than_e4m3():
    """E5M2 は range 重視（max=57344）で E4M3(max=448) より広い（指数 5 vs 4 bit）。"""
    assert dtype_limits("float8_e5m2").max_normal > dtype_limits("float8_e4m3").max_normal
    assert dtype_limits("float8_e5m2").max_normal == 57344.0


def test_mxfp4_extremely_narrow_range_makes_overflow_the_dominant_risk():
    """MXFP4(E2M1) は max=6.0 と全 dtype 中最狭。8.0 のような小さな値でも overflow する。

    OCP MX v1.0: NVIDIA Blackwell / AMD CDNA4 の両方が HW ネイティブ対応する
    最粗フォーマット。block スケール(E8M0)があっても block 内要素の相対レンジは
    この表の通り極端に狭い（tolerance.py 冒頭 docstring 参照）。
    """
    lim = dtype_limits("mxfp4_e2m1")
    assert lim.max_normal == 6.0
    env = certify_gemm(K=64, dtype="mxfp4_e2m1", scale=1.0)
    x = np.full((4, 4), 8.0, dtype=np.float32)
    rep = check_tensor(x, env)
    assert not rep.in_envelope, "MXFP4 で 8.0 が overflow にならない（max=6.0 のはず）"
    assert rep.max_risk == Risk.BLOCK
    # 同じ 8.0 は fp16(max=65504) では overflow しない → dtype 依存の overflow リスク差を実証
    fp16_findings = check_tensor(x, certify_gemm(K=64, dtype="float16", scale=8.0)).findings
    assert not any("overflow" in f.message for f in fp16_findings), \
        "fp16(max=65504) で 8.0 が overflow 判定された（想定外）"


def test_mxfp6_variants_range_ordering():
    """MXFP6 の 2 バリアント: E3M2(max=28) が E2M3(max=7.5) より広い（指数 3 vs 2 bit）。"""
    assert dtype_limits("mxfp6_e3m2").max_normal > dtype_limits("mxfp6_e2m3").max_normal
    assert dtype_limits("mxfp6_e3m2").max_normal == 28.0
    assert dtype_limits("mxfp6_e2m3").max_normal == 7.5
    # mxfp4 が MX ファミリー中で最も狭い
    assert dtype_limits("mxfp4_e2m1").max_normal < dtype_limits("mxfp6_e2m3").max_normal


def test_outlier_features_break_single_scale():
    # outlier feature(massive activations): 数チャネルだけ巨大→単一 scale 仮定が破綻し WARN
    rng = np.random.default_rng(0)
    normal = rng.standard_normal((64, 256)).astype(np.float32)
    outlier = rng.standard_normal((64, 256)).astype(np.float32)
    outlier[:, 5] *= 200
    outlier[:, 100] *= 150
    assert channel_scale_spread(normal) < 3.0          # near-uniform は ~1
    assert channel_scale_spread(outlier) > 50.0        # outlier は桁違い
    assert check_outlier_features(normal).max_risk == Risk.OK
    assert check_outlier_features(outlier).max_risk == Risk.WARN
    assert any("outlier" in f.op for f in check_outlier_features(outlier).findings)


def test_certify_from_sample_measures_real_scale():
    """certify_from_sample は実サンプルの RMS を scale に使う（scale=1 仮定を排除）。

    Q14 の修正: certify_gemm(scale=1.0) で認証後に scale=50 のデータを check_tensor すると
    scale 超過 BLOCK が誤発火する。certify_from_sample は同じデータで認証・検査を一致させる。
    """
    rng = np.random.default_rng(42)
    # scale ≈ 50 のテンソル（LLM の未正規化活性等）
    x_large = rng.standard_normal((32, 128)).astype(np.float32) * 50.0

    # scale=1 で認証すると scale 超過 BLOCK が誤発火する
    env_wrong = certify_gemm(K=128, dtype="float32", scale=1.0)
    rep_wrong = check_tensor(x_large, env_wrong)
    assert rep_wrong.max_risk == Risk.BLOCK, "scale=1 認証でスケール超過 BLOCK が出ないと test 意味なし"

    # certify_from_sample は実 scale を測定 → 同じデータが BLOCK にならない
    env_correct = certify_from_sample(x_large, K=128, dtype="float32")
    assert env_correct.scale_max > 40.0, f"RMS が正しく測定されていない: scale={env_correct.scale_max:.2f}"
    rep_correct = check_tensor(x_large, env_correct)
    # scale 超過 BLOCK が出ない（scale が正しく認証された）
    scale_blocks = [f for f in rep_correct.findings if "scale" in f.message and f.risk == Risk.BLOCK]
    assert not scale_blocks, f"certify_from_sample 後も scale BLOCK: {rep_correct.to_text()}"


def test_certify_from_sample_zero_tensor():
    """ゼロテンソルで certify_from_sample が除算エラーを出さない（ゼロ除算防止）。"""
    x_zero = np.zeros((8, 8), dtype=np.float32)
    env = certify_from_sample(x_zero, K=64, dtype="float16")
    assert env.scale_max > 0.0, "ゼロテンソルで scale_max がゼロになってはいけない"
    # ゼロテンソルは IN-ENVELOPE（overflow/denormal/scale 超過のいずれも無し）
    rep = check_tensor(x_zero, env)
    assert rep.in_envelope, rep.to_text()


def test_certify_from_sample_small_scale():
    """scale << 1 のテンソル（正規化後の活性）で certify_from_sample が正しく機能する。"""
    rng = np.random.default_rng(7)
    x_small = rng.standard_normal((16, 64)).astype(np.float32) * 0.01
    env = certify_from_sample(x_small, K=64, dtype="float16")
    # scale が実 RMS に追従していること
    actual_rms = float(np.sqrt(np.mean(x_small ** 2)))
    assert abs(env.scale_max - actual_rms) / actual_rms < 0.01, \
        f"scale_max={env.scale_max:.4f} が実 RMS={actual_rms:.4f} と一致しない"
    rep = check_tensor(x_small, env)
    # 小 scale で overflow や scale 超過が起きない
    assert rep.in_envelope, rep.to_text()


def main() -> int:
    ok = True
    tests = [
        test_in_envelope_passes,
        test_fp16_overflow_is_block,
        test_scale_exceedance_voids_certification,
        test_denormal_flagged_for_ftz_divergence,
        test_envelope_thresholds_are_sensitive_to_their_constants,
        test_nan_is_block,
        test_fp16_softmax_logit_overflow,
        test_real_fp16_overflow_actually_happens,
        test_outlier_features_break_single_scale,
        test_float64_dtype_limits_are_correct,
        test_tf32_dtype_limits_match_float32,
        test_fp8_e4m3_narrow_range_makes_overflow_the_main_risk,
        test_fp8_e5m2_wider_range_than_e4m3,
        test_mxfp4_extremely_narrow_range_makes_overflow_the_dominant_risk,
        test_mxfp6_variants_range_ordering,
        test_certify_from_sample_measures_real_scale,
        test_certify_from_sample_zero_tensor,
        test_certify_from_sample_small_scale,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    # 参考: fp16 と bf16 のエンベロープ差（overflow vs precision）
    print("\n--- dtype 別エンベロープ（IEEE 754 実値）---")
    for d in ("float8_e4m3", "float8_e5m2", "mxfp4_e2m1", "mxfp6_e2m3", "mxfp6_e3m2",
              "float16", "bfloat16", "float32", "tf32", "float64"):
        lim = dtype_limits(d)
        print(f"  {d:9s} max={lim.max_normal:.3g} min_normal={lim.min_normal:.2e} "
              f"exp-overflow at |x|>{lim.exp_overflow:.2f}")
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
