"""Tests for tsugi.blame (新視点13: ベンダー責帰)."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import numpy as np

from tsugi.blame import (
    accuracy_relative,
    compare_accuracy,
    layer_blame,
)
from tsugi.report import Risk


# --- accuracy_relative -------------------------------------------------------

def test_accuracy_relative_exact_match():
    a = np.array([1.0, 2.0, 3.0])
    assert accuracy_relative(a, a) == 0.0


def test_accuracy_relative_known_value():
    out = np.array([1.1])
    ref = np.array([1.0])
    # max|out - ref| = 0.1, max|ref| = 1.0 → ratio = 0.1
    assert abs(accuracy_relative(out, ref) - 0.1) < 1e-10


def test_accuracy_relative_empty():
    assert accuracy_relative(np.array([]), np.array([])) == 0.0


def test_accuracy_relative_scale_invariant():
    # Scaling oracle and output by same factor → relative distance unchanged
    a = np.array([1.1, 2.2])
    ref = np.array([1.0, 2.0])
    d1 = accuracy_relative(a, ref)
    d2 = accuracy_relative(a * 100, ref * 100)
    assert abs(d1 - d2) < 1e-10, "relative distance should be scale invariant"


# --- compare_accuracy: basic cases -------------------------------------------

def test_compare_accuracy_both_ok():
    oracle = np.array([1.0, 2.0, 3.0])
    a = oracle + 1e-6
    b = oracle - 1e-6
    rep = compare_accuracy(a, b, oracle, tol=1e-4)
    assert rep.max_risk == Risk.OK
    assert rep.closer in ("A", "B", "TIED")


def test_compare_accuracy_b_is_culprit():
    oracle = np.array([1.0, 2.0, 3.0])
    a = oracle + 1e-7   # A very close
    b = oracle + 0.5    # B far away
    rep = compare_accuracy(a, b, oracle, tol=1e-4)
    assert rep.closer == "A", f"expected closer=A, got {rep.closer}"
    assert rep.max_risk >= Risk.WARN
    # Finding should mention vendor B
    assert any("B" in f.message for f in rep.findings)


def test_compare_accuracy_a_is_culprit():
    oracle = np.array([1.0, 2.0, 3.0])
    a = oracle + 0.5    # A far away
    b = oracle + 1e-7   # B very close
    rep = compare_accuracy(a, b, oracle, tol=1e-4)
    assert rep.closer == "B", f"expected closer=B, got {rep.closer}"
    assert rep.max_risk >= Risk.WARN
    assert any("A" in f.message for f in rep.findings)


def test_compare_accuracy_tied():
    oracle = np.array([1.0, 2.0, 3.0])
    # Both at same distance from oracle
    a = oracle + 0.1
    b = oracle - 0.1
    rep = compare_accuracy(a, b, oracle, tol=0.01)
    # ratio = dist_max / dist_min ≈ 1 → TIED
    assert rep.closer == "TIED", f"expected TIED, got {rep.closer}"


def test_compare_accuracy_block_when_far():
    oracle = np.array([1.0])
    a = oracle + 1e-9       # A essentially perfect
    b = oracle + 100.0      # B wildly off → > tol * 10
    rep = compare_accuracy(a, b, oracle, tol=1e-4)
    assert rep.max_risk == Risk.BLOCK


def test_compare_accuracy_empty():
    rep = compare_accuracy(np.array([]), np.array([]), np.array([]), tol=1e-4)
    assert any(f.risk == Risk.INFO for f in rep.findings)


# --- compare_accuracy: dist_a / dist_b values --------------------------------

def test_dist_values_stored_correctly():
    oracle = np.array([10.0])
    a = oracle + 1.0    # dist_a = 1 / 10 = 0.1
    b = oracle + 5.0    # dist_b = 5 / 10 = 0.5
    rep = compare_accuracy(a, b, oracle, tol=1e-2)
    assert abs(rep.dist_a - 0.1) < 1e-10
    assert abs(rep.dist_b - 0.5) < 1e-10
    assert rep.ratio > 2.0   # 0.5 / 0.1 = 5


def test_ratio_threshold_controls_tied():
    oracle = np.array([1.0])
    a = oracle + 0.2    # dist_a ≈ 0.2
    b = oracle + 0.3    # dist_b ≈ 0.3
    # ratio = 0.3/0.2 = 1.5
    rep_tight = compare_accuracy(a, b, oracle, tol=0.01, ratio_threshold=1.2)
    rep_loose = compare_accuracy(a, b, oracle, tol=0.01, ratio_threshold=2.0)
    # tight threshold: 1.5 > 1.2 → closer=A (A is nearer, B is blamed)
    assert rep_tight.closer == "A", f"expected closer=A (A is nearer), got {rep_tight.closer}"
    # loose threshold: 1.5 < 2.0 → TIED
    assert rep_loose.closer == "TIED", f"expected TIED, got {rep_loose.closer}"


def test_compare_accuracy_absolute_mode():
    oracle = np.array([100.0])
    a = oracle + 0.001   # tiny absolute diff
    b = oracle + 5.0     # large absolute diff
    rep = compare_accuracy(a, b, oracle, tol=0.01, relative=False)
    assert rep.closer == "A"


# --- compare_accuracy: to_text -----------------------------------------------

def test_compare_accuracy_to_text():
    oracle = np.array([1.0])
    a = oracle + 1e-7
    b = oracle + 0.5
    rep = compare_accuracy(a, b, oracle, tol=1e-4)
    text = rep.to_text()
    assert "blame" in text
    assert "dist_a" in text


def test_compare_accuracy_both_exceed_tol():
    # Both exceed tol but A closer — WARN/BLOCK should mention "B"
    oracle = np.array([1.0])
    a = oracle + 0.01   # dist_a = 0.01
    b = oracle + 1.0    # dist_b = 1.0
    rep = compare_accuracy(a, b, oracle, tol=1e-4)
    assert rep.closer == "A"
    assert any("B" in f.message for f in rep.findings)


# --- layer_blame -------------------------------------------------------------

def _id(x):
    return np.asarray(x, dtype=np.float64)


def _scale(k):
    return lambda x: np.asarray(x, dtype=np.float64) * k


def test_layer_blame_identical_to_oracle():
    oracle = np.array([1.0, 2.0])
    layers = [_id, _id, _id]
    result = layer_blame(layers, layers, layers, oracle)
    assert len(result) == 3
    for da, db in result:
        assert da == 0.0 and db == 0.0


def test_layer_blame_b_diverges_at_layer1():
    oracle_layers = [_id, _id, _id]
    layers_a = [_id, _id, _id]
    layers_b = [_id, lambda x: np.asarray(x, dtype=np.float64) + 0.5, _id]
    x = np.array([1.0])
    result = layer_blame(layers_a, layers_b, oracle_layers, x, relative=False)
    # Layer 0: both identical to oracle
    da0, db0 = result[0]
    assert da0 == 0.0 and db0 == 0.0
    # Layer 1: B diverges from oracle
    da1, db1 = result[1]
    assert da1 == 0.0
    assert db1 > 0.1, f"B should diverge from oracle at layer 1, got {db1}"
    # Layer 2: B's error persists
    da2, db2 = result[2]
    assert da2 == 0.0
    assert db2 > 0.1


def test_layer_blame_truncates_to_shortest():
    x = np.array([1.0])
    la = [_id, _id, _id]
    lb = [_id, _id]
    lo = [_id, _id, _id]
    result = layer_blame(la, lb, lo, x)
    assert len(result) == 2  # zip truncates


# --- integration: attribution + blame chain ----------------------------------

def test_attribution_blame_chain():
    """
    Complete diagnostic chain: which layer (attribution) + which vendor (blame).
    Layer 1 is the spike; vendor B is closer to oracle (A is the culprit).
    """
    from tsugi.attribution import attribute

    oracle_layers = [_id, _scale(2.0), _id]
    layers_b = [_id, _scale(2.0), _id]  # B matches oracle
    # A has a scale error at layer 1
    layers_a = [_id, lambda x: np.asarray(x, dtype=np.float64) * 2.0 + 0.5, _id]
    x = np.array([1.0, 2.0])
    oracle_x = np.array([1.0, 2.0])

    # attribution: find spike layer
    attr = attribute(layers_a, layers_b, x, tol=0.1, relative=False,
                     names=["embed", "proj", "out"])
    assert attr.spike == 1, f"expected spike at layer 1, got {attr.spike}"

    # blame at output: A is the culprit
    def run(layers, inp):
        out = np.asarray(inp, dtype=np.float64)
        for fn in layers:
            out = np.asarray(fn(out), dtype=np.float64)
        return out

    out_a = run(layers_a, x)
    out_b = run(layers_b, x)
    out_oracle = run(oracle_layers, oracle_x)

    rep = compare_accuracy(out_a, out_b, out_oracle, tol=0.01)
    assert rep.closer == "B", (
        f"B should be closer to oracle (A has the error), got closer={rep.closer}")
    # Diagnostic chain: spike at proj(1), vendor A is culprit
    assert attr.spike_name == "proj"
    assert rep.closer == "B"  # → "vendor A を直せ"


def test_compare_accuracy_tied_both_far_becomes_block():
    """Q31: TIED + both vendors far from oracle → BLOCK (systemic shared failure)."""
    oracle = np.array([1.0])
    tol = 1e-4
    # both > tol * 10 (= 1e-3) and same distance → TIED but both far
    a = oracle + 0.5
    b = oracle - 0.5
    rep = compare_accuracy(a, b, oracle, tol=tol)
    assert rep.closer == "TIED", f"expected TIED, got {rep.closer}"
    assert rep.max_risk == Risk.BLOCK, (
        f"expected BLOCK when both dist >> tol*10, got {rep.max_risk}")
    assert any("系統的共有誤り" in f.message or "both" in f.message.lower()
               or "両方" in f.message for f in rep.findings)


def main():
    tests = [
        test_accuracy_relative_exact_match,
        test_accuracy_relative_known_value,
        test_accuracy_relative_empty,
        test_accuracy_relative_scale_invariant,
        test_compare_accuracy_both_ok,
        test_compare_accuracy_b_is_culprit,
        test_compare_accuracy_a_is_culprit,
        test_compare_accuracy_tied,
        test_compare_accuracy_block_when_far,
        test_compare_accuracy_empty,
        test_dist_values_stored_correctly,
        test_ratio_threshold_controls_tied,
        test_compare_accuracy_absolute_mode,
        test_compare_accuracy_to_text,
        test_compare_accuracy_both_exceed_tol,
        test_layer_blame_identical_to_oracle,
        test_layer_blame_b_diverges_at_layer1,
        test_layer_blame_truncates_to_shortest,
        test_attribution_blame_chain,
        test_compare_accuracy_tied_both_far_becomes_block,
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
