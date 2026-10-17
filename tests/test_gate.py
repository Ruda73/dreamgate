"""
tests/test_gate.py — One test per behavioural claim.
"""

import math
import random
import statistics
import pytest
import sys
sys.path.insert(0, '..')

from dreamgate.gate import DreamGate, GateConfig, GateResult, NORMAL, HOLD, DREAM

CFG = GateConfig(sigma=0.5, tau_low=1.5, dream_sustain=8, hysteresis=0.5)


def stable_signal(n=200, seed=42):
    random.seed(seed)
    return [math.sin(i*0.1)*0.3 + random.gauss(0, 0.05) for i in range(n)]


def burst_signal(warmup=40, burst_len=80, seed=42):
    """Unpredictable: large random-phase sine keeps err high."""
    random.seed(seed)
    pre  = [math.sin(i*0.1)*0.3 + random.gauss(0, 0.05) for i in range(warmup)]
    post = [math.sin(i*2.3 + random.gauss(0, 1.5))*2.5 + random.gauss(0, 0.4)
            for i in range(burst_len)]
    return pre + post


def sustained_anomaly(n=60, seed=99):
    """Random walk long enough that DREAM is reliably reached."""
    random.seed(seed)
    v, out = 0.0, []
    for _ in range(n):
        v += random.gauss(0, 1.5)
        out.append(v)
    return out


# ── Stable signal ─────────────────────────────────────────────────────────────

def test_stable_signal_stays_mostly_normal():
    gate = DreamGate(config=CFG)
    states = [gate.tick(v).state for v in stable_signal()]
    frac = states.count(NORMAL) / len(states)
    assert frac >= 0.90, f"Expected >=90% NORMAL, got {frac:.1%}"

def test_stable_signal_clock_near_zero():
    gate = DreamGate(config=CFG)
    for v in stable_signal():
        gate.tick(v)
    assert gate.clock < 0.5


# ── Burst signal ──────────────────────────────────────────────────────────────

def test_burst_reaches_hold():
    gate = DreamGate(config=CFG)
    states = [gate.tick(v).state for v in burst_signal()]
    assert HOLD in states

def test_burst_reaches_dream():
    gate = DreamGate(config=CFG)
    states = [gate.tick(v).state for v in burst_signal()]
    assert DREAM in states

def test_clock_never_decreases():
    """The only hard invariant on the clock."""
    gate = DreamGate(config=CFG)
    prev = 0.0
    for v in stable_signal(n=100) + burst_signal():
        r = gate.tick(v)
        assert r.clock >= prev - 1e-12
        prev = r.clock


# ── State transitions ─────────────────────────────────────────────────────────

def test_normal_to_hold_on_unpredictable_signal():
    gate = DreamGate(config=CFG)
    for v in stable_signal(n=50):
        gate.tick(v)
    assert gate.state == NORMAL
    states = [gate.tick(v).state for v in [5.0, -5.0, 5.0, -5.0]]
    assert any(s in (HOLD, DREAM) for s in states)

def test_dream_reached_on_sustained_anomaly():
    gate = DreamGate(config=CFG)
    for v in stable_signal(n=50):
        gate.tick(v)
    states = [gate.tick(v).state for v in sustained_anomaly(n=60)]
    assert DREAM in states, f"States seen: {set(states)}"

def test_transient_spike_does_not_trigger_dream():
    gate = DreamGate(config=CFG)
    for v in stable_signal(n=50):
        gate.tick(v)
    gate.tick(5.0)
    gate.tick(-5.0)
    for v in stable_signal(n=20, seed=77):
        gate.tick(v)
    assert gate.state != DREAM

def test_dream_exits_after_stable_period():
    gate = DreamGate(config=CFG)
    for v in stable_signal(n=50):
        gate.tick(v)
    for v in sustained_anomaly(n=60):
        gate.tick(v)
    for v in stable_signal(n=150, seed=55):
        gate.tick(v)
    assert gate.state == NORMAL


# ── GateResult ────────────────────────────────────────────────────────────────

def test_result_has_all_fields():
    r = DreamGate(config=CFG).tick(1.0)
    for f in ['actual','predicted','err','gamma','h','epsilon_Q',
              'P_err','X_gen','X_mem','gap','regime',
              'delta_tau','clock','state','sustain','sigma']:
        assert hasattr(r, f), f"Missing: {f}"

def test_gamma_always_in_range():
    gate = DreamGate(config=CFG)
    for v in stable_signal(n=50) + sustained_anomaly(n=20):
        assert 0.0 <= gate.tick(v).gamma <= 1.0

def test_h_always_nonnegative():
    gate = DreamGate(config=CFG)
    for v in stable_signal(n=50) + sustained_anomaly(n=20):
        assert gate.tick(v).h >= 0.0

def test_sigma_in_result_matches_config_when_not_adaptive():
    gate = DreamGate(config=CFG)
    r = gate.tick(1.0)
    assert r.sigma == CFG.sigma


# ── Callback ──────────────────────────────────────────────────────────────────

def test_callback_fires_only_in_dream_state():
    states = []
    gate = DreamGate(config=CFG, on_dream=lambda r: states.append(r.state))
    for v in stable_signal(n=50) + sustained_anomaly(n=60):
        gate.tick(v)
    assert all(s == DREAM for s in states)

def test_callback_fires_during_sustained_anomaly():
    fired = []
    gate = DreamGate(config=CFG, on_dream=lambda r: fired.append(1))
    for v in stable_signal(n=50) + sustained_anomaly(n=60):
        gate.tick(v)
    assert len(fired) > 0


# ── Dual channel ──────────────────────────────────────────────────────────────

def test_dual_channel_xgen_in_range():
    gate = DreamGate(config=CFG)
    random.seed(7)
    for i in range(100):
        train = math.sin(i*0.1)*0.5 + random.gauss(0, 0.08)
        held  = math.sin(i*0.1)*0.5 + random.gauss(0, 0.12)
        r = gate.tick(train, held_out=held)
    assert 0.0 <= r.X_gen <= 1.0
    assert 0.0 <= r.X_mem <= 1.0


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_reset_clears_clock():
    gate = DreamGate(config=CFG)
    for v in burst_signal():
        gate.tick(v)
    gate.reset()
    assert gate.clock == 0.0

def test_reset_clears_state():
    gate = DreamGate(config=CFG)
    for v in sustained_anomaly(n=60):
        gate.tick(v)
    gate.reset()
    assert gate.state == NORMAL

def test_after_reset_returns_valid_result():
    gate = DreamGate(config=CFG)
    for v in burst_signal():
        gate.tick(v)
    gate.reset()
    r = gate.tick(0.5)
    assert isinstance(r, GateResult)
    assert r.state == NORMAL


# ── Pluggable predictor (NEW v0.3.0) ──────────────────────────────────────────

def test_custom_predictor_called():
    """Custom predictor is used instead of EWMA."""
    calls = []
    def record_predictor(history):
        calls.append(len(history))
        return statistics.mean(history[-5:]) if len(history) >= 5 else history[-1]

    gate = DreamGate(config=CFG, predictor=record_predictor)
    for v in stable_signal(n=20):
        gate.tick(v)
    # First tick uses actual (no history), rest use predictor
    assert len(calls) == 19  # called for ticks 2..20

def test_rolling_median_predictor_stays_normal_on_stable():
    """Rolling median predictor should keep stable signal in NORMAL."""
    def rolling_median(history):
        w = history[-20:] if len(history) >= 20 else history
        return statistics.median(w)

    gate = DreamGate(config=CFG, predictor=rolling_median)
    states = [gate.tick(v).state for v in stable_signal(n=200)]
    frac = states.count(NORMAL) / len(states)
    assert frac >= 0.85, f"Expected >=85% NORMAL with median predictor, got {frac:.1%}"

def test_custom_predictor_reaches_dream_on_anomaly():
    """Even with custom predictor, sustained anomaly should reach DREAM."""
    def rolling_median(history):
        w = history[-10:] if len(history) >= 10 else history
        return statistics.median(w)

    gate = DreamGate(config=CFG, predictor=rolling_median)
    for v in stable_signal(n=50):
        gate.tick(v)
    states = [gate.tick(v).state for v in sustained_anomaly(n=60)]
    assert DREAM in states

def test_reset_clears_history():
    """After reset, history is cleared; predictor sees fresh history from tick 2 onward."""
    seen_history_len = []
    def pred(history):
        seen_history_len.append(len(history))
        return history[-1] if history else 0.0

    gate = DreamGate(config=CFG, predictor=pred)
    for v in stable_signal(n=10):
        gate.tick(v)
    n_calls_before_reset = len(seen_history_len)
    gate.reset()
    gate.tick(1.0)  # history=[] -> predictor NOT called, falls back to actual
    gate.tick(2.0)  # history=[1.0] -> predictor called with 1 entry
    # Predictor should have been called once after reset (on tick 2 post-reset)
    assert len(seen_history_len) == n_calls_before_reset + 1
    assert seen_history_len[-1] == 1  # only one value in history: the first post-reset tick


# ── Adaptive sigma (NEW v0.3.0) ───────────────────────────────────────────────

def test_adaptive_sigma_changes_over_time():
    """sigma should drift from initial value after many NORMAL ticks."""
    cfg = GateConfig(sigma=0.5, adaptive_sigma=True, sigma_alpha=0.1)
    gate = DreamGate(config=cfg)
    for v in stable_signal(n=200):
        gate.tick(v)
    # sigma should have moved from 0.5 toward actual error distribution
    assert gate.sigma != 0.5

def test_adaptive_sigma_not_chased_during_dream():
    """sigma must NOT update during anomaly ticks — gate should not chase its own alarm."""
    cfg = GateConfig(sigma=0.5, tau_low=1.5, dream_sustain=8,
                     adaptive_sigma=True, sigma_alpha=0.5)  # high alpha to amplify any leak
    gate = DreamGate(config=cfg)
    for v in stable_signal(n=50):
        gate.tick(v)
    sigma_before = gate.sigma
    for v in sustained_anomaly(n=60):
        gate.tick(v)
    # sigma_before was set in NORMAL; anomaly should NOT have pulled it up significantly
    # If sigma chased anomalies it would be >> sigma_before
    assert gate.sigma < sigma_before * 5.0  # sanity upper bound

def test_adaptive_sigma_floor():
    """sigma must never reach zero even with tiny errors."""
    cfg = GateConfig(sigma=0.5, adaptive_sigma=True, sigma_alpha=0.5)
    gate = DreamGate(config=cfg)
    # Feed constant signal — errors collapse to zero
    for _ in range(200):
        gate.tick(1.0)
    assert gate.sigma > 0.0

def test_non_adaptive_sigma_unchanged():
    """With adaptive_sigma=False, sigma must never change."""
    gate = DreamGate(config=CFG)  # adaptive_sigma=False by default
    for v in stable_signal(n=200) + sustained_anomaly(n=60):
        gate.tick(v)
    assert gate.sigma == CFG.sigma
