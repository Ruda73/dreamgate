"""
dreamgate/stream.py — Streaming API.

One function: stream().
Takes any iterable of signal values.
Yields GateResult at each step.

    from dreamgate import stream, DREAM

    for result in stream(my_values):
        if result.state == DREAM:
            act(result)

That is the entire public surface.
"""

from typing import Iterable, Iterator, Optional, Tuple
from .gate import DreamGate, GateConfig, GateResult


def stream(
    signal: Iterable[float],
    *,
    held_out: Optional[Iterable[float]] = None,
    config: Optional[GateConfig] = None,
) -> Iterator[GateResult]:
    """
    Process a signal stream tick by tick.

    Args:
        signal:   iterable of observed signal values
        held_out: optional parallel iterable of held-out values
                  for X_gen computation. Must be same length as signal.
                  If None, X_gen is computed in single-channel mode.
        config:   GateConfig (defaults if not provided)

    Yields:
        GateResult at each tick

    Example — single channel:
        for result in stream([1.0, 1.1, 0.9, 5.0, 4.8, 5.2]):
            print(result.state, result.h)

    Example — dual channel:
        for result in stream(train_values, held_out=held_values):
            if result.state == DREAM:
                trigger(result)

    Example — real-time generator:
        def sensor():
            while True:
                yield read_sensor()

        for result in stream(sensor()):
            if result.state == DREAM:
                alert()
    """
    gate = DreamGate(config=config)

    if held_out is None:
        for value in signal:
            yield gate.tick(value)
    else:
        for value, hval in zip(signal, held_out):
            yield gate.tick(value, held_out=hval)


def stream_pairs(
    pairs: Iterable[Tuple[float, float]],
    *,
    config: Optional[GateConfig] = None,
) -> Iterator[GateResult]:
    """
    Process (train, held_out) pairs.

    Convenience wrapper when signal and held_out are already zipped.

        for result in stream_pairs(zip(train, held)):
            ...
    """
    gate = DreamGate(config=config)
    for train_val, held_val in pairs:
        yield gate.tick(train_val, held_out=held_val)


def collect_dream_events(
    signal: Iterable[float],
    *,
    held_out: Optional[Iterable[float]] = None,
    config: Optional[GateConfig] = None,
) -> list:
    """
    Run the full stream and return only DREAM events.

    Useful for batch analysis of recorded signals.

        events = collect_dream_events(my_signal)
        print(f'{len(events)} anomalies detected')
        for e in events:
            print(f'  tick {e.sustain}: h={e.h:.3f}  εQ={e.epsilon_Q:.4f}')
    """
    from .gate import DREAM
    return [
        r for r in stream(signal, held_out=held_out, config=config)
        if r.state == DREAM
    ]


# ── Batch measure ─────────────────────────────────────────────────────────────

def measure(
    signal: list,
    *,
    held_out: Optional[list] = None,
    config: Optional[GateConfig] = None,
) -> dict:
    """
    One-shot batch analysis. Run the full stream, return summary statistics.

    Args:
        signal:   list of observed values
        held_out: optional parallel list for X_gen
        config:   GateConfig (defaults if not provided)

    Returns:
        dict with keys:
            n          — number of ticks
            state_pct  — {NORMAL, HOLD, DREAM} as percentages
            dream_events — list of GateResult where state==DREAM
            first_dream  — tick index of first DREAM, or None
            clock      — final learning clock value
            h_mean     — mean surprisal over all ticks
            h_max      — max surprisal
            epsilon_Q_mean — mean thermodynamic cost

    Example:
        result = measure(my_signal)
        print(result['state_pct'])
        print(f"First anomaly at tick {result['first_dream']}")
    """
    from .gate import DREAM as _DREAM, HOLD as _HOLD, NORMAL as _NORMAL

    results    = list(stream(signal, held_out=held_out, config=config))
    n          = len(results)
    if n == 0:
        return {"n": 0, "state_pct": {}, "dream_events": [],
                "first_dream": None, "clock": 0.0,
                "h_mean": 0.0, "h_max": 0.0, "epsilon_Q_mean": 0.0}

    counts     = {_NORMAL: 0, _HOLD: 0, _DREAM: 0}
    dreams     = []
    first      = None
    h_sum      = 0.0
    h_max      = 0.0
    eq_sum     = 0.0

    for i, r in enumerate(results):
        counts[r.state] += 1
        h_sum  += r.h
        eq_sum += r.epsilon_Q
        if r.h > h_max:
            h_max = r.h
        if r.state == _DREAM:
            dreams.append(r)
            if first is None:
                first = i

    return {
        "n":            n,
        "state_pct":    {k: v / n * 100 for k, v in counts.items()},
        "dream_events": dreams,
        "first_dream":  first,
        "clock":        results[-1].clock,
        "h_mean":       h_sum / n,
        "h_max":        h_max,
        "epsilon_Q_mean": eq_sum / n,
    }
