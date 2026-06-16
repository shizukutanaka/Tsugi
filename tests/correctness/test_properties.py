"""Tsugi property-based tests（SOCRATIC Q35）— 数値主張を fuzz 入力で検証する。

固定 seed の単発でなく、多数のランダム入力に対して *性質* が常に成り立つことを確かめる。
ゼロ依存（numpy のみ・hypothesis 不使用）で軽量な property 検査を行う。
各 property は「どんな入力でも成り立つべき不変則」を 1 つ表す。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tsugi.calibration import detectability_floor, systematic_divergence  # noqa: E402
from tsugi.decision import (  # noqa: E402
    divergence_rms,
    flip_rate,
    predicted_flip_bound,
    residual_divergence_rms,
)
from tsugi.envelope import certify_gemm, check_tensor  # noqa: E402
from tsugi.nondeterminism import attribute  # noqa: E402
from tsugi.propagation import empirical_cond  # noqa: E402
from tsugi.tolerance import derive_tolerance  # noqa: E402

N_TRIALS = 200
DTYPES = ("float16", "bfloat16", "float32")


def _check_all(name: str, prop, n=N_TRIALS) -> None:
    """prop(rng, seed) を n 回回し、False を返したら AssertionError。"""
    for s in range(n):
        ok = prop(np.random.default_rng(s), s)
        assert ok, f"property '{name}' violated at trial {s}"


# --- 性質群 ---------------------------------------------------------------

def prop_tolerance_monotonic_in_K(rng, s):
    k1 = int(rng.integers(1, 1000))
    k2 = k1 + int(rng.integers(1, 3000))
    dt = DTYPES[s % 3]
    return derive_tolerance(k2, dt)["atol"] >= derive_tolerance(k1, dt)["atol"]


def prop_detectability_floor_monotonic_in_K(rng, s):
    k1 = int(rng.integers(1, 500))
    k2 = k1 + int(rng.integers(1, 2000))
    # scale 非依存・K 単調増加
    return (detectability_floor(k2, "float16", scale=float(rng.uniform(0.1, 100)))["rel"]
            >= detectability_floor(k1, "float16", scale=float(rng.uniform(0.1, 100)))["rel"])


def prop_residual_le_total(rng, s):
    n, c = int(rng.integers(40, 300)), int(rng.integers(8, 150))
    a = rng.standard_normal((n, c)).astype(np.float32)
    b = a + float(rng.uniform(0, 0.3)) * rng.standard_normal((n, c)).astype(np.float32)
    return residual_divergence_rms(a, b) <= divergence_rms(a, b) + 1e-6


def prop_flip_rate_scale_invariant(rng, s):
    n, c = int(rng.integers(40, 300)), int(rng.integers(8, 150))
    a = rng.standard_normal((n, c)).astype(np.float32)
    b = a + 0.05 * rng.standard_normal((n, c)).astype(np.float32)
    k = float(rng.uniform(0.1, 10))
    return flip_rate(a, b) == flip_rate(a * k, b * k)


def prop_residual_bound_is_upper_bound(rng, s):
    n, c = int(rng.integers(40, 400)), int(rng.integers(10, 200))
    a = rng.standard_normal((n, c)).astype(np.float32)
    # 系統(アフィン)＋乱雑の混合でも残差 bound は実フリップ率の上界
    shift = float(rng.uniform(0, 0.5)) * rng.standard_normal((n, 1)).astype(np.float32)
    b = (a * float(rng.uniform(0.8, 1.2)) + shift
         + float(rng.uniform(0, 0.1)) * rng.standard_normal((n, c)).astype(np.float32)).astype(np.float32)
    return flip_rate(a, b) <= predicted_flip_bound(a, residual_divergence_rms(a, b)) + 1e-9


def prop_affine_systematic_no_flip(rng, s):
    # スケール α(>0)・一様シフトのみ → argmax 不変（フリップ 0）
    n, c = int(rng.integers(40, 300)), int(rng.integers(8, 150))
    a = rng.standard_normal((n, c)).astype(np.float32)
    alpha = float(rng.uniform(0.2, 3.0))
    shift = float(rng.uniform(-1, 1)) * rng.standard_normal((n, 1)).astype(np.float32)
    return flip_rate(a, (a * alpha + shift).astype(np.float32)) == 0.0


def prop_systematic_divergence_recovers_scale(rng, s):
    a = rng.standard_normal((16, 16)).astype(np.float32)
    alpha = float(rng.uniform(0.85, 1.15))
    return abs(systematic_divergence(a, a * alpha) - (alpha - 1.0)) < 1e-4


def prop_empirical_cond_positive_is_well_conditioned(rng, s):
    pos = np.abs(rng.standard_normal((4, 128)))
    signed = rng.standard_normal((4, 128))
    cp = empirical_cond(pos, "reduce", axis=1)
    cs = empirical_cond(signed, "reduce", axis=1)
    return cp < 1.5 and cs >= cp - 1e-6   # 正の和は ~1・符号付きは相殺で >=


def prop_attribute_regimes(rng, s):
    noise = float(rng.uniform(1e-3, 1.0))
    tol = noise + float(rng.uniform(1e-3, 2.0))
    below = attribute(noise * float(rng.uniform(0, 0.99)), noise, tol)
    above = attribute(tol + float(rng.uniform(1e-3, 5.0)), noise, tol)
    return below == "INDISTINGUISHABLE" and above == "DIVERGENT"


def prop_envelope_overflow_always_out(rng, s):
    env = certify_gemm(int(rng.integers(1, 2048)), "float16", 1.0)
    over = np.full((4, 4), 70000.0, np.float32)   # > fp16 max 65504
    safe = rng.standard_normal((8, 8)).astype(np.float32)
    return (not check_tensor(over, env).in_envelope) and check_tensor(safe, env).in_envelope


PROPERTIES = {
    "tolerance monotonic in K": prop_tolerance_monotonic_in_K,
    "detectability floor monotonic in K": prop_detectability_floor_monotonic_in_K,
    "residual <= total divergence": prop_residual_le_total,
    "flip rate scale-invariant": prop_flip_rate_scale_invariant,
    "residual flip bound is upper bound": prop_residual_bound_is_upper_bound,
    "affine-systematic divergence never flips": prop_affine_systematic_no_flip,
    "systematic_divergence recovers scale": prop_systematic_divergence_recovers_scale,
    "empirical_cond well-conditioned for positive sums": prop_empirical_cond_positive_is_well_conditioned,
    "attribute regimes (below noise / above tol)": prop_attribute_regimes,
    "envelope flags overflow, accepts in-range": prop_envelope_overflow_always_out,
}


def main() -> int:
    ok = True
    for name, prop in PROPERTIES.items():
        try:
            _check_all(name, prop)
            print(f"[PASS] {name}  ({N_TRIALS} trials)")
        except AssertionError as e:
            print(f"[FAIL] {e}")
            ok = False
    print(f"\n{'ALL PASS' if ok else 'FAILURES'} — {len(PROPERTIES)} properties × {N_TRIALS} trials")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
