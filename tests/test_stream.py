"""
tests/test_stream.py
"""

import math
import random
import pytest
import sys
sys.path.insert(0, '..')

from dreamgate.stream import stream, stream_pairs, collect_dream_events
from dreamgate.gate import GateConfig, GateResult, NORMAL, HOLD, DREAM

CFG = GateConfig(sigma=0.5, tau_low=1.5, dream_sustain=8)


def stable(n=100, seed=42):
    random.seed(seed)
    return [math.sin(i*0.1)*0.3 + random.gauss(0, 0.05) for i in range(n)]


def anomaly(n=60, seed=99):
    random.seed(seed)
    v, out = 0.0, []
    for _ in range(n):
        v += random.gauss(0, 1.5)
        out.append(v)
    return out


# ── stream() ─────────────────────────────────────────────────────────────────

def test_stream_yields_gate_results():
    results = list(stream(stable(n=10), config=CFG))
    assert len(results) == 10
    assert all(isinstance(r, GateResult) for r in results)

def test_stream_stable_signal_mostly_normal():
    states = [r.state for r in stream(stable(n=200), config=CFG)]
    assert states.count(NORMAL) / len(states) >= 0.90

def test_stream_detects_dream_on_anomaly():
    signal = stable(n=50) + anomaly(n=60)
    states = [r.state for r in stream(signal, config=CFG)]
    assert DREAM in states

def test_stream_empty_signal():
    results = list(stream([], config=CFG))
    assert results == []

def test_stream_single_value():
    results = list(stream([1.0], config=CFG))
    assert len(results) == 1
    assert results[0].state == NORMAL

def test_stream_dual_channel():
    random.seed(7)
    train = [math.sin(i*0.15)*0.5 + random.gauss(0, 0.08) for i in range(100)]
    held  = [math.sin(i*0.15)*0.5 + random.gauss(0, 0.12) for i in range(100)]
    results = list(stream(train, held_out=held, config=CFG))
    assert len(results) == 100
    assert all(0.0 <= r.X_gen <= 1.0 for r in results)

def test_stream_generator_input():
    """stream() must work with generators, not just lists."""
    def gen():
        for i in range(50):
            yield math.sin(i*0.1)*0.3

    results = list(stream(gen(), config=CFG))
    assert len(results) == 50


# ── stream_pairs() ────────────────────────────────────────────────────────────

def test_stream_pairs_equivalent_to_stream_dual():
    random.seed(7)
    train = [math.sin(i*0.15)*0.5 + random.gauss(0, 0.08) for i in range(50)]
    held  = [math.sin(i*0.15)*0.5 + random.gauss(0, 0.12) for i in range(50)]

    r1 = list(stream(train, held_out=held, config=CFG))
    r2 = list(stream_pairs(zip(train, held), config=CFG))

    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.h      == pytest.approx(b.h,      abs=1e-10)
        assert a.gamma  == pytest.approx(b.gamma,  abs=1e-10)
        assert a.state  == b.state


# ── collect_dream_events() ────────────────────────────────────────────────────

def test_collect_dream_events_all_dream():
    signal = stable(n=50) + anomaly(n=60)
    events = collect_dream_events(signal, config=CFG)
    assert all(e.state == DREAM for e in events)

def test_collect_dream_events_empty_on_stable():
    events = collect_dream_events(stable(n=200), config=CFG)
    assert events == []

def test_collect_dream_events_nonempty_on_anomaly():
    signal = stable(n=50) + anomaly(n=60)
    events = collect_dream_events(signal, config=CFG)
    assert len(events) > 0
