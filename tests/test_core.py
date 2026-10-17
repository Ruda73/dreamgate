"""
tests/test_core.py

One test per claim. No more.
"""

import math
import pytest
import sys
sys.path.insert(0, '..')

from dreamgate.core import (
    gamma, surprisal, epsilon_Q, helstrom, bures,
    epistatic, many_worlds_tax, clock_tick, XgenEstimator,
)


# ── gamma ─────────────────────────────────────────────────────────────────────

def test_gamma_zero_at_zero_error():
    assert gamma(0.0, 0.5) == 0.0

def test_gamma_increases_with_error():
    assert gamma(0.5, 0.5) < gamma(1.0, 0.5) < gamma(2.0, 0.5)

def test_gamma_bounded():
    for err in [0.0, 0.5, 1.0, 5.0, 100.0]:
        g = gamma(err, 0.5)
        assert 0.0 <= g <= 1.0

def test_gamma_rejects_bad_sigma():
    with pytest.raises(ValueError):
        gamma(1.0, 0.0)
    with pytest.raises(ValueError):
        gamma(1.0, -1.0)


# ── surprisal ─────────────────────────────────────────────────────────────────

def test_surprisal_zero_at_zero_error():
    assert surprisal(0.0, 0.5) == pytest.approx(0.0, abs=1e-10)

def test_surprisal_increases_with_error():
    assert surprisal(0.1, 0.5) < surprisal(0.5, 0.5) < surprisal(1.0, 0.5)

def test_surprisal_quadratic_in_err():
    # h = err²/(2σ²ln2) — doubling err quadruples h
    h1 = surprisal(0.5, 0.5)
    h2 = surprisal(1.0, 0.5)
    assert h2 == pytest.approx(4.0 * h1, rel=1e-6)

def test_surprisal_one_sigma_is_0p72_bits():
    # err = sigma → h = 1/(2ln2) ≈ 0.7213 bits
    h = surprisal(0.5, 0.5)  # err = sigma = 0.5
    assert h == pytest.approx(1.0 / (2.0 * math.log(2)), rel=1e-6)


# ── Five theorems ─────────────────────────────────────────────────────────────

def test_helstrom_zero_at_zero_gamma():
    assert helstrom(0.0) == 0.0

def test_helstrom_half_at_gamma_one():
    assert helstrom(1.0) == pytest.approx(0.5, abs=1e-10)

def test_helstrom_monotone():
    assert helstrom(0.2) < helstrom(0.5) < helstrom(0.8)

def test_bures_is_gamma_squared():
    for g in [0.0, 0.3, 0.7, 1.0]:
        assert bures(g) == pytest.approx(g * g, rel=1e-10)

def test_epsilon_Q_zero_at_zero():
    assert epsilon_Q(0.0) == 0.0

def test_epsilon_Q_half_at_gamma_one():
    assert epsilon_Q(1.0) == pytest.approx(0.5, abs=1e-10)

def test_epistatic_independent_channels_when_g_ep_zero():
    # With g_ep=0: ℰ = β(h1² + h2²)
    beta = 0.7
    h1, h2 = 2.0, 1.5
    e = epistatic(h1, h2, beta=beta, g_ep=0.0)
    assert e == pytest.approx(beta * (h1**2 + h2**2), rel=1e-10)

def test_epistatic_synergistic_amplification():
    # Positive g_ep: co-activation is more expensive than independent
    h1, h2 = 2.0, 2.0
    e_ind = epistatic(h1, h2, beta=1.0, g_ep=0.0)
    e_syn = epistatic(h1, h2, beta=1.0, g_ep=0.5)
    assert e_syn > e_ind

def test_many_worlds_tax_symmetric():
    # c0=0.5 gives maximum branch uncertainty
    assert many_worlds_tax(0.5, c0=0.5) > many_worlds_tax(0.5, c0=0.9)

def test_many_worlds_tax_zero_at_zero_gamma():
    assert many_worlds_tax(0.0, c0=0.5) == 0.0


# ── clock_tick ────────────────────────────────────────────────────────────────

def test_clock_zero_when_h_zero():
    assert clock_tick(0.0, 0.8) == 0.0

def test_clock_zero_when_Xgen_zero():
    assert clock_tick(2.0, 0.0) == 0.0

def test_clock_zero_when_Xgen_negative():
    assert clock_tick(2.0, -0.5) == 0.0

def test_clock_quadratic_in_h():
    dt1 = clock_tick(1.0, 0.8)
    dt2 = clock_tick(2.0, 0.8)
    assert dt2 == pytest.approx(4.0 * dt1, rel=1e-10)

def test_clock_linear_in_Xgen():
    dt1 = clock_tick(2.0, 0.4)
    dt2 = clock_tick(2.0, 0.8)
    assert dt2 == pytest.approx(2.0 * dt1, rel=1e-10)


# ── XgenEstimator ─────────────────────────────────────────────────────────────

def test_xgen_starts_in_warming():
    est = XgenEstimator()
    r = est.update(1.0, 1.0)
    assert r["regime"] == "WARMING"

def test_xgen_detects_generalising():
    import random
    random.seed(1)
    est = XgenEstimator(alpha_fast=0.15, alpha_slow=0.02)
    for i in range(200):
        h_tr = max(0.01, 3.0 - i*0.012 + random.gauss(0, 0.1))
        h_hd = max(0.01, 2.9 - i*0.012 + random.gauss(0, 0.1))
        r = est.update(h_tr, h_hd)
    assert r["regime"] == "GENERALISING"

def test_xgen_detects_memorising():
    import random
    random.seed(2)
    est = XgenEstimator(alpha_fast=0.15, alpha_slow=0.02)
    for i in range(200):
        h_tr = max(0.01, 3.0 - i*0.012 + random.gauss(0, 0.1))
        h_hd = max(0.01, 3.0            + random.gauss(0, 0.1))
        r = est.update(h_tr, h_hd)
    assert r["regime"] == "MEMORISING"

def test_xgen_reset():
    est = XgenEstimator()
    for _ in range(50):
        est.update(2.0, 1.5)
    est.reset()
    assert est.X_gen == 0.0
    assert est.regime == "WARMING"
