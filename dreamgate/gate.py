"""
dreamgate/gate.py — The DreamGate state machine.

Gate fires on h(s) alone — anomaly detection.
Clock advances on h²·X_gen — learning measurement.
These are independent. Neither depends on the other.

    NORMAL : h < tau_low              — prior confirmed
    HOLD   : h >= tau_low, brief      — anomaly, watching
    DREAM  : h >= tau_low, sustained  — hypothesis warranted

Predictor note
--------------
By default the gate uses a first-order EWMA as its one-step predictor.
This works well for stationary signals. For trending or seasonal signals,
pass a custom predictor callable:

    gate = DreamGate(predictor=my_ar_model)

The predictor receives the full history list (all values seen so far) and
must return a float. It is called BEFORE the current tick is appended, so
the most recent value in the list is the previous tick.

    def rolling_median_predictor(history):
        window = history[-20:] if len(history) >= 20 else history
        return statistics.median(window)

    gate = DreamGate(predictor=rolling_median_predictor)

Stateless helpers vs stateful DreamGate
----------------------------------------
stream() and collect_dream_events() each create a fresh DreamGate
internally — they are stateless helpers for one-shot batch analysis.
To process a signal across multiple batches while preserving gate state
(e.g. a live sensor feed chunked into windows), create a DreamGate
directly and call tick() on each value:

    gate = DreamGate(config=my_config)
    for chunk in sensor_chunks():
        for value in chunk:
            result = gate.tick(value)
            if result.state == DREAM:
                alert(result)

Reference: van Rensburg (2026).
"""

from dataclasses import dataclass
from typing import Optional, Callable, List
from .core import gamma, surprisal, epsilon_Q, helstrom, XgenEstimator, clock_tick

NORMAL = "NORMAL"
HOLD   = "HOLD"
DREAM  = "DREAM"


@dataclass
class GateConfig:
    """
    sigma         : match to typical prediction error in your signal.
                    Use calibrate() to fit sigma from a reference window.
                    Rule of thumb: sigma ~ std(signal) * 0.2 for a well-fitted predictor.
    tau_low       : h(bits) threshold for HOLD entry.
                    At tau_low=1.5: fires when err > 1.45*sigma.
                    At tau_low=2.0: fires when err > 1.67*sigma.
    dream_sustain : consecutive HOLD ticks before DREAM.
    hysteresis    : fraction of tau_low for NORMAL return (prevents oscillation).
    adaptive_sigma: if True, sigma is updated via EWMA on running prediction errors
                    during NORMAL ticks only (avoids chasing anomalies).
                    Useful for signals with non-stationary variance (seasonal, thermal drift).
                    sigma is then the initial estimate.
    sigma_alpha   : EWMA coefficient for adaptive sigma (default 0.01 -- slow adaptation).
    clock_scale   : delta_tau scale factor -- tune to your application's time units.
    """
    sigma:          float = 0.5
    tau_low:        float = 1.5
    dream_sustain:  int   = 8
    hysteresis:     float = 0.5
    adaptive_sigma: bool  = False
    sigma_alpha:    float = 0.01
    alpha_fast:     float = 0.15
    alpha_slow:     float = 0.02
    clock_scale:    float = 0.01


@dataclass
class GateResult:
    actual:    float
    predicted: float
    err:       float
    gamma:     float
    h:         float
    epsilon_Q: float
    P_err:     float
    X_gen:     float
    X_mem:     float
    gap:       float
    regime:    str
    delta_tau: float
    clock:     float
    state:     str
    sustain:   int
    sigma:     float


class DreamGate:
    """
    The DreamGate.

        gate   = DreamGate()
        result = gate.tick(value)
        if result.state == DREAM:
            act(result)

    Custom predictor (AR model, rolling median, domain-specific):

        gate = DreamGate(predictor=my_predictor)

    The predictor callable receives the history list (values seen so far,
    NOT including the current tick) and returns a predicted float.
    """

    def __init__(self,
                 config:    Optional[GateConfig] = None,
                 on_dream:  Optional[Callable] = None,
                 predictor: Optional[Callable[[List[float]], float]] = None):
        self.cfg         = config or GateConfig()
        self.on_dream    = on_dream
        self._predictor  = predictor
        self._xgen       = XgenEstimator(self.cfg.alpha_fast, self.cfg.alpha_slow)
        self._history: List[float] = []
        self._ewma_pred: Optional[float] = None
        self._state      = NORMAL
        self._sustain    = 0
        self._clock      = 0.0
        self._n          = 0
        self._sigma      = self.cfg.sigma

    def tick(self, actual: float,
             held_out: Optional[float] = None) -> GateResult:
        cfg = self.cfg

        # Predict BEFORE appending current value
        if self._predictor is not None:
            pred = self._predictor(self._history) if self._history else actual
        else:
            pred = actual if self._ewma_pred is None else self._ewma_pred

        self._history.append(actual)
        self._ewma_pred = cfg.alpha_fast * actual + (1.0 - cfg.alpha_fast) * pred

        err = abs(actual - pred)

        # Adaptive sigma: update only during NORMAL ticks, and only for
        # errors within 3*sigma of current estimate (outlier rejection).
        # self._state here is the state BEFORE _step() so HOLD/DREAM ticks are excluded.
        if cfg.adaptive_sigma and self._n > 0 and self._state == NORMAL:
            if err < 3.0 * self._sigma:  # outlier rejection — don't chase large errors
                self._sigma = max(1e-6,
                    cfg.sigma_alpha * err + (1.0 - cfg.sigma_alpha) * self._sigma)

        sigma = self._sigma

        g  = gamma(err, sigma)
        h  = surprisal(err, sigma)
        eQ = epsilon_Q(g)
        pe = helstrom(g)

        h_held = surprisal(
            abs((held_out if held_out is not None else actual) - pred),
            sigma
        )
        lr = self._xgen.update(h, h_held)

        dt          = clock_tick(h, lr["X_gen"], cfg.clock_scale)
        self._clock += dt
        self._n     += 1

        self._state, self._sustain = self._step(h)

        r = GateResult(
            actual=actual, predicted=pred, err=err,
            gamma=g, h=h, epsilon_Q=eQ, P_err=pe,
            X_gen=lr["X_gen"], X_mem=lr["X_mem"],
            gap=lr["gap"], regime=lr["regime"],
            delta_tau=dt, clock=self._clock,
            state=self._state, sustain=self._sustain,
            sigma=sigma,
        )
        if self._state == DREAM and self.on_dream:
            self.on_dream(r)
        return r

    def reset(self):
        self._xgen.reset()
        self._history   = []
        self._ewma_pred = None
        self._state     = NORMAL
        self._sustain   = 0
        self._clock     = 0.0
        self._n         = 0
        self._sigma     = self.cfg.sigma

    @property
    def state(self):  return self._state
    @property
    def clock(self):  return self._clock
    @property
    def sigma(self):  return self._sigma

    def _step(self, h: float):
        cfg = self.cfg
        s   = self._state
        sus = self._sustain
        hi  = h >= cfg.tau_low
        lo  = h <  cfg.tau_low * cfg.hysteresis

        if s == NORMAL:
            return (HOLD, 1) if hi else (NORMAL, 0)
        if s == HOLD:
            if hi:
                sus += 1
                return (DREAM, sus) if sus >= cfg.dream_sustain else (HOLD, sus)
            return (NORMAL, 0) if lo else (HOLD, sus)
        if s == DREAM:
            return (NORMAL, 0) if lo else (DREAM, sus + 1)
        return (NORMAL, 0)
