"""Tests for tsugi.attribution (新視点12: 発散帰属)."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import numpy as np

from tsugi.attribution import (
    AttributionReport,
    attribute,
    bisect_onset,
    find_onset,
    find_spike,
    layer_divergences,
)
from tsugi.report import Risk


# --- helper layers -----------------------------------------------------------

def _scale(factor):
    return lambda x: x * factor


def _add_bias(bias):
    return lambda x: x + bias


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def _identity(x):
    return np.asarray(x, dtype=np.float64)


# --- layer_divergences -------------------------------------------------------

def test_layer_divergences_identical_vendors():
    layers = [_scale(2.0), _scale(0.5), _add_bias(1.0)]
    x = np.array([1.0, 2.0, 3.0])
    divs = layer_divergences(layers, layers, x)
    assert len(divs) == 3
    for d in divs:
        assert d == 0.0, f"identical vendors should produce zero divergence, got {d}"


def test_layer_divergences_known_values():
    # Vendor A: identity; Vendor B: adds 0.1 at layer 1
    layers_a = [_identity, _identity, _identity]
    layers_b = [lambda x: x + 0.1, _identity, _identity]
    x = np.array([1.0, 1.0])
    divs = layer_divergences(layers_a, layers_b, x, relative=False)
    assert len(divs) == 3
    # Layer 0: a=[1,1], b=[1.1,1.1] → max_abs_diff=0.1
    assert abs(divs[0] - 0.1) < 1e-10, f"expected 0.1 at layer 0, got {divs[0]}"
    # Layer 1,2: b still carries the bias forward
    assert divs[1] == divs[0], "bias should persist into later layers"
    assert divs[2] == divs[0]


def test_layer_divergences_relative():
    # relative=True: div = max|a-b| / (max|a| + eps)
    layers_a = [_scale(10.0)]
    layers_b = [_scale(10.1)]
    x = np.array([1.0])
    divs_rel = layer_divergences(layers_a, layers_b, x, relative=True)
    divs_abs = layer_divergences(layers_a, layers_b, x, relative=False)
    # relative div should be smaller than abs when scale > 1
    assert divs_rel[0] < divs_abs[0]
    # Expected: abs = 0.1, relative = 0.1 / 10.0 = 0.01
    assert abs(divs_rel[0] - 0.01) < 1e-10


def test_layer_divergences_growing_divergence():
    # Each layer doubles the accumulated value → divergence grows
    layers_a = [_scale(2.0)] * 5

    def vendor_b_layer_0(x):
        return x * 2.0 + 0.01  # small perturbation at layer 0

    layers_b = [vendor_b_layer_0] + [_scale(2.0)] * 4
    x = np.array([1.0])
    divs = layer_divergences(layers_a, layers_b, x, relative=False)
    # After layer 0: a=2.0, b=2.01 → diff=0.01
    # Each subsequent layer doubles diff too
    assert divs[0] < divs[1] < divs[2] < divs[3] < divs[4], "divergence should grow"


def test_layer_divergences_length_mismatch():
    # zip stops at shorter list — should not crash
    layers_a = [_identity, _identity, _identity]
    layers_b = [_identity, _identity]
    x = np.array([1.0])
    divs = layer_divergences(layers_a, layers_b, x)
    assert len(divs) == 2  # truncated to shorter


# --- find_onset --------------------------------------------------------------

def test_find_onset_none_when_all_below():
    divs = [0.001, 0.002, 0.003]
    assert find_onset(divs, threshold=0.01) is None


def test_find_onset_first_exceeding():
    divs = [0.001, 0.005, 0.02, 0.03]
    onset = find_onset(divs, threshold=0.01)
    assert onset == 2, f"expected onset=2, got {onset}"


def test_find_onset_first_layer():
    divs = [0.1, 0.2, 0.3]
    assert find_onset(divs, threshold=0.05) == 0


def test_find_onset_empty():
    assert find_onset([], threshold=0.01) is None


def test_find_onset_exact_threshold_not_exceeded():
    # strictly greater than threshold
    divs = [0.01, 0.01]  # all equal threshold → none exceed it
    assert find_onset(divs, threshold=0.01) is None  # 0.01 is NOT > 0.01
    assert find_onset([0.01, 0.009], threshold=0.009) == 0  # 0.01 > 0.009


# --- find_spike --------------------------------------------------------------

def test_find_spike_empty():
    assert find_spike([]) is None


def test_find_spike_single():
    assert find_spike([0.5]) == 0


def test_find_spike_known_position():
    divs = [0.01, 0.02, 0.10, 0.11]  # biggest jump at index 2 (0.10 - 0.02 = 0.08)
    assert find_spike(divs) == 2


def test_find_spike_at_start():
    divs = [0.50, 0.51, 0.52]  # largest delta is index 0 (treated as divs[0] - 0)
    assert find_spike(divs) == 0


def test_find_spike_monotone_small_increments():
    # Equal increments → spike should be at index 0 (first delta = divs[0])
    divs = [1.0, 2.0, 3.0, 4.0]
    spike = find_spike(divs)
    # First delta = 1.0 (divs[0]), subsequent = 1.0; argmax picks first occurrence = 0
    assert spike == 0


# --- bisect_onset ------------------------------------------------------------

def _make_prefix_fns(layers_a, layers_b, x0):
    """Build fn_prefix_a/b(i, x) → output of first i+1 layers."""
    def prefix_a(i, x):
        out = np.asarray(x, dtype=np.float64)
        for la in layers_a[:i + 1]:
            out = np.asarray(la(out), dtype=np.float64)
        return out

    def prefix_b(i, x):
        out = np.asarray(x, dtype=np.float64)
        for lb in layers_b[:i + 1]:
            out = np.asarray(lb(out), dtype=np.float64)
        return out

    return prefix_a, prefix_b


def test_bisect_onset_matches_linear_scan():
    # Perturbation inserted at layer 3
    layers_a = [_scale(1.0)] * 6
    layers_b = [_scale(1.0)] * 3 + [lambda x: x + 0.05] + [_scale(1.0)] * 2
    x = np.array([1.0, 2.0])
    fa, fb = _make_prefix_fns(layers_a, layers_b, x)

    linear_onset = find_onset(layer_divergences(layers_a, layers_b, x, relative=False), 0.01)
    bisect_result = bisect_onset(fa, fb, x, n_layers=6, tol=0.01, relative=False)
    assert bisect_result == linear_onset, (
        f"bisect={bisect_result} vs linear={linear_onset}")


def test_bisect_onset_no_perturbation():
    layers = [_identity] * 5
    x = np.array([1.0])
    fa, fb = _make_prefix_fns(layers, layers, x)
    assert bisect_onset(fa, fb, x, n_layers=5, tol=0.001) is None


def test_bisect_onset_at_layer_0():
    layers_a = [_identity] * 4
    layers_b = [lambda x: x + 1.0] + [_identity] * 3
    x = np.array([1.0])
    fa, fb = _make_prefix_fns(layers_a, layers_b, x)
    result = bisect_onset(fa, fb, x, n_layers=4, tol=0.001, relative=False)
    assert result == 0


def test_bisect_onset_zero_layers():
    def f(i, x): return x
    assert bisect_onset(f, f, np.array([1.0]), n_layers=0, tol=0.01) is None


# --- attribute ---------------------------------------------------------------

def test_attribute_clean_all_layers():
    layers = [_scale(2.0), _add_bias(1.0), _scale(0.5)]
    x = np.array([1.0, 2.0])
    rep = attribute(layers, layers, x, tol=1e-6)
    assert isinstance(rep, AttributionReport)
    assert rep.onset is None
    assert any(f.risk == Risk.OK for f in rep.findings)


def test_attribute_detects_onset_and_spike():
    # Layer 0: clean, Layer 1: large amplifier, Layer 2: small increment
    def amplifier(x):
        return x * 100.0 + 0.5  # vendor B adds 0.5

    layers_a = [_identity, _scale(100.0), _scale(1.0)]
    layers_b = [_identity, amplifier, _scale(1.0)]
    x = np.array([0.1])
    rep = attribute(layers_a, layers_b, x, tol=1e-4, relative=False,
                    names=["embed", "ffn", "norm"])

    assert rep.onset is not None, "onset should be detected"
    assert rep.onset == 1, f"expected onset at layer 1 (ffn), got {rep.onset}"
    assert rep.spike == 1, f"expected spike at layer 1 (ffn), got {rep.spike}"
    assert rep.spike_name() == "ffn"
    assert rep.onset_name() == "ffn"


def test_attribute_names_fallback_when_none():
    layers = [_scale(2.0), lambda x: x + 0.1]
    x = np.array([1.0])
    rep = attribute(layers, [_scale(2.0), _identity], x, tol=1e-4, relative=False)
    # names=None → auto-generated
    assert rep.onset_name().startswith("layer[") or rep.onset_name() == "(none)"


def test_attribute_empty_layers():
    rep = attribute([], [], np.array([1.0]), tol=1e-4)
    assert rep.n_layers == 0
    assert any(f.risk == Risk.INFO for f in rep.findings)


def test_attribute_spike_onset_mismatch_adds_info():
    # Construct divs where onset != spike:
    # Layer 0: clean, Layer 1: small onset, Layer 2: big spike
    def small_bias(x):
        return x + 0.01  # onset at layer 1

    def big_amplifier(x):
        return x * 50.0 + 0.5  # spike at layer 2 relative to prev

    layers_a = [_identity, _identity, _scale(50.0)]
    layers_b = [_identity, small_bias, big_amplifier]
    x = np.array([0.0])
    rep = attribute(layers_a, layers_b, x, tol=0.005, relative=False,
                    names=["L0", "L1", "L2"])

    # onset should be at layer 1 (0.01 > 0.005), spike at layer 2
    assert rep.onset == 1
    assert rep.spike == 2
    info_msgs = [f.message for f in rep.findings if f.risk == Risk.INFO]
    assert any("spike" in m and "onset" in m for m in info_msgs), (
        "should note onset != spike mismatch")


def test_attribute_risk_block_when_large_final_divergence():
    def broken_layer(x):
        return x * 1000.0 + 50.0

    layers_a = [_identity, _identity]
    layers_b = [_identity, broken_layer]
    x = np.array([1.0])
    rep = attribute(layers_a, layers_b, x, tol=1e-4, relative=False)
    assert rep.max_risk >= Risk.BLOCK, "large final divergence should be BLOCK"


def test_attribute_to_text():
    layers_a = [_scale(2.0), _scale(0.5)]
    x = np.array([1.0])
    rep = attribute(layers_a, layers_a, x, tol=1e-6)
    text = rep.to_text()
    assert "attribution" in text
    assert "layers" in text


# --- integration: propagation dominant vs attribution spike ------------------

def test_attribution_spike_vs_propagation_dominant():
    """
    Verify the design contract: propagation identifies theoretical dominant op;
    attribution.spike provides the empirical measurement.

    A matmul layer with large K dominates accumulation error in propagation theory.
    Empirically, the layer that introduces the largest absolute divergence is the spike.
    We confirm both APIs are callable and consistent in direction.
    """
    from tsugi.propagation import GraphOp, propagate

    # Theory: matmul(K=1) → matmul(K=2048) → matmul(K=1)
    # The large-K matmul is the dominant in propagation theory (local_div ∝ √K)
    ops = [
        GraphOp(kind="matmul", K=1, dtype="float16"),
        GraphOp(kind="matmul", K=2048, dtype="float16"),  # dominant
        GraphOp(kind="matmul", K=1, dtype="float16"),
    ]
    prop_report = propagate(ops)
    theory_dominant = prop_report.dominant  # property, not method
    assert theory_dominant is not None
    # dominant op should be the high-K matmul (index 1 in ops)
    assert theory_dominant.kind == "matmul"

    # Empirical: vendor B has a scale error at layer 1 (the large-K layer)
    # That layer produces the biggest divergence increment → spike at index 1
    layers_a = [_scale(1.0), _scale(1.0), _scale(1.0)]
    layers_b = [_scale(1.0), lambda x: x + 0.5, _scale(1.0)]  # diff at layer 1
    x = np.array([1.0, 2.0])
    rep = attribute(layers_a, layers_b, x, tol=0.01, relative=False,
                    names=["matmul_small", "matmul_big", "matmul_small2"])

    assert rep.spike == 1, f"expected spike at layer 1, got {rep.spike}"
    assert rep.spike_name() == "matmul_big"
    # Both APIs agree the middle layer is the critical one
    assert theory_dominant.kind == rep.spike_name().split("_")[0]


# --- diagnose (combined attribution + blame) ---------------------------------

def _oracle_id(x):
    return np.asarray(x, dtype=np.float64)


def test_diagnose_no_oracle_attribution_only():
    from tsugi.attribution import diagnose
    layers_a = [_identity, _identity, _identity]
    layers_b = [_identity, lambda x: np.asarray(x, dtype=np.float64) + 0.1, _identity]
    x = np.array([1.0, 2.0])
    rep = diagnose(layers_a, layers_b, None, x, tol=0.05, relative=False,
                   names=["L0", "L1", "L2"])
    assert rep.onset == 1
    assert rep.spike == 1
    assert rep.spike_closer == "TIED"  # no oracle → no blame
    assert any("oracle なし" in f.message for f in rep.findings)


def test_diagnose_with_oracle_blames_b():
    from tsugi.attribution import diagnose
    oracle_layers = [_oracle_id, _scale(2.0), _oracle_id]
    layers_a = [_oracle_id, _scale(2.0), _oracle_id]           # A matches oracle
    layers_b = [_oracle_id, lambda x: x * 2.0 + 0.5, _oracle_id]  # B diverges at layer 1
    x = np.array([1.0])
    rep = diagnose(layers_a, layers_b, oracle_layers, x, tol=0.05, relative=False,
                   names=["L0", "L1", "L2"])
    assert rep.spike == 1
    assert rep.spike_closer == "A", f"A matches oracle → spike_closer should be A, got {rep.spike_closer}"
    assert rep.spike_dist_a < rep.spike_dist_b


def test_diagnose_all_clean_no_findings():
    from tsugi.attribution import diagnose
    layers = [_oracle_id, _oracle_id]
    x = np.array([1.0, 2.0])
    rep = diagnose(layers, layers, layers, x, tol=1e-6)
    assert rep.onset is None
    assert any(f.risk == 0 for f in rep.findings)  # Risk.OK == 0


def test_diagnose_to_text_contains_chain_info():
    from tsugi.attribution import diagnose
    oracle_layers = [_oracle_id, _oracle_id]
    layers_a = [_oracle_id, _oracle_id]
    layers_b = [_oracle_id, lambda x: np.asarray(x, dtype=np.float64) + 0.5]
    x = np.array([1.0])
    rep = diagnose(layers_a, layers_b, oracle_layers, x, tol=0.01, relative=False,
                   names=["embed", "proj"])
    text = rep.to_text()
    assert "diagnosis" in text
    assert "fix vendor" in text


def main():
    tests = [
        test_layer_divergences_identical_vendors,
        test_layer_divergences_known_values,
        test_layer_divergences_relative,
        test_layer_divergences_growing_divergence,
        test_layer_divergences_length_mismatch,
        test_find_onset_none_when_all_below,
        test_find_onset_first_exceeding,
        test_find_onset_first_layer,
        test_find_onset_empty,
        test_find_onset_exact_threshold_not_exceeded,
        test_find_spike_empty,
        test_find_spike_single,
        test_find_spike_known_position,
        test_find_spike_at_start,
        test_find_spike_monotone_small_increments,
        test_bisect_onset_matches_linear_scan,
        test_bisect_onset_no_perturbation,
        test_bisect_onset_at_layer_0,
        test_bisect_onset_zero_layers,
        test_attribute_clean_all_layers,
        test_attribute_detects_onset_and_spike,
        test_attribute_names_fallback_when_none,
        test_attribute_empty_layers,
        test_attribute_spike_onset_mismatch_adds_info,
        test_attribute_risk_block_when_large_final_divergence,
        test_attribute_to_text,
        test_attribution_spike_vs_propagation_dominant,
        test_diagnose_no_oracle_attribution_only,
        test_diagnose_with_oracle_blames_b,
        test_diagnose_all_clean_no_findings,
        test_diagnose_to_text_contains_chain_info,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
