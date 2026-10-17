"""
dreamgate — thermodynamic signal intelligence SDK

    from dreamgate import DreamGate, stream, DREAM
"""

from .core import (
    gamma, surprisal, epsilon_Q,
    helstrom, bures, epistatic, many_worlds_tax,
    clock_tick, XgenEstimator,
)
from .gate import (
    DreamGate, GateConfig, GateResult,
    NORMAL, HOLD, DREAM,
)
from .stream import stream, stream_pairs, collect_dream_events, measure

__version__ = "0.2.0"
__all__ = [
    "DreamGate", "GateConfig", "GateResult",
    "NORMAL", "HOLD", "DREAM",
    "stream", "stream_pairs", "collect_dream_events", "measure",
    "gamma", "surprisal", "epsilon_Q",
    "helstrom", "bures", "epistatic", "many_worlds_tax",
    "clock_tick", "XgenEstimator",
]
