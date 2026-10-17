"""
tests/test_config.py — Tests for calibrate() and vertical configs.
"""

import math
import random
import warnings
import pytest
import sys
sys.path.insert(0, '..')

from dreamgate.config import calibrate, industrial_config, edtech_config, security_config, gaming_config
from dreamgate.gate import DreamGate, GateConfig, DREAM, NORMAL


def stable_signal(n=300, seed=42, noise=0.05):
    random.seed(seed)
    return [math.sin(i*0.1)*0.3 + random.gauss(0, noise) for i in range(n)]


# ── calibrate() ───────────────────────────────────────────────────────────────

def test_calibrate_returns_gateconfig():
    cfg = calibrate(stable_signal())
    assert isinstance(cfg, GateConfig)

def test_calibrate_sigma_positive():
    cfg = calibrate(stable_signal())
    assert cfg.sigma > 0.0

def test_calibrate_sigma_reflects_noise():
    """Higher-noise signal should produce higher sigma."""
    cfg_low  = calibrate(stable_signal(noise=0.01))
    cfg_high = calibrate(stable_signal(noise=0.20))
    assert cfg_high.sigma > cfg_low.sigma

def test_calibrate_passes_tau_low():
    cfg = calibrate(stable_signal(), tau_low=2.5)
    assert cfg.tau_low == 2.5

def test_calibrate_passes_dream_sustain():
    cfg = calibrate(stable_signal(), dream_sustain=12)
    assert cfg.dream_sustain == 12

def test_calibrate_passes_adaptive_sigma():
    cfg = calibrate(stable_signal(), adaptive_sigma=True)
    assert cfg.adaptive_sigma is True

def test_calibrate_too_few_values_raises():
    with pytest.raises(ValueError, match="at least 5"):
        calibrate([1.0, 2.0])

def test_calibrate_warns_on_small_sample():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        calibrate(stable_signal(n=15))
        assert any("unstable" in str(warning.message).lower() for warning in w)

def test_calibrate_bad_percentile_raises():
    with pytest.raises(ValueError, match="percentile"):
        calibrate(stable_signal(), percentile=1.5)

def test_calibrate_gate_stays_normal_on_reference():
    """A gate calibrated on reference data should stay mostly NORMAL on similar data."""
    ref = stable_signal(n=300, seed=1)
    cfg = calibrate(ref)
    gate = DreamGate(config=cfg)
    states = [gate.tick(v).state for v in stable_signal(n=200, seed=99)]
    frac = states.count(NORMAL) / len(states)
    assert frac >= 0.85, f"Expected >=85% NORMAL on similar signal, got {frac:.1%}"

def test_calibrate_percentile_tighter_gives_smaller_sigma():
    ref = stable_signal(n=300)
    cfg_p50 = calibrate(ref, percentile=0.50)
    cfg_p90 = calibrate(ref, percentile=0.90)
    assert cfg_p90.sigma >= cfg_p50.sigma


# ── Vertical configs ──────────────────────────────────────────────────────────

def test_industrial_config_returns_gateconfig():
    assert isinstance(industrial_config(), GateConfig)

def test_industrial_config_adaptive_sigma():
    assert industrial_config().adaptive_sigma is True

def test_edtech_config_returns_gateconfig():
    assert isinstance(edtech_config(), GateConfig)

def test_security_config_low_tau():
    """Security config should be more sensitive (lower tau_low)."""
    assert security_config().tau_low < industrial_config().tau_low

def test_gaming_config_returns_gateconfig():
    assert isinstance(gaming_config(), GateConfig)
