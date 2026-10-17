"""
dreamgate/core.py — The computation layer.

One primitive: γ = state overlap between predicted and actual.
Everything derives from γ.

    γ = 1 - exp(-err²/2σ²)       state overlap
    h = err²/(2σ²·ln2)            surprisal (bits) — direct, not through P_err
    P_err = ½(1 - √(1-γ²))       Helstrom bound
    εQ  = ½γ²                     Feinstein cost (kBT=1)
    d²B = γ²                      Bures geodesic
    ℰ   = h^T Γ h                 epistatic energy
    εMW = ½·c0·c1·γ²              Many Worlds tax

Reference: van Rensburg (2026).
"""

import math
from typing import Optional

KB_T_LN2 = 1.0   # natural units


# ── The primitive ─────────────────────────────────────────────────────────────

def gamma(err: float, sigma: float) -> float:
    """State overlap. γ=0 at err=0; γ→1 as err grows."""
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return 1.0 - math.exp(-(err * err) / (2.0 * sigma * sigma))


def surprisal(err: float, sigma: float) -> float:
    """
    Shannon surprisal h(s) = -log2 Q(s), Q Gaussian.

    h = err² / (2σ² ln2)

    err=0   → h=0      (perfect prediction, no surprise)
    err=σ   → h≈0.72   (one standard deviation off)
    err=2σ  → h≈2.89   (two sigma — clearly anomalous)
    err=3σ  → h≈6.50   (three sigma — strong anomaly)
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return (err * err) / (2.0 * sigma * sigma * math.log(2))


# ── Five theorems — all γ² in different units ─────────────────────────────────

def helstrom(g: float) -> float:
    """T1: P_err = ½(1 - √(1-γ²))"""
    g = max(0.0, min(1.0, g))
    return 0.5 * (1.0 - math.sqrt(1.0 - g * g))


def bures(g: float) -> float:
    """T2: d²_B ≈ γ²"""
    return g * g


def epsilon_Q(g: float) -> float:
    """T3: εQ = ½γ² (kBT=1)"""
    return 0.5 * g * g


def epistatic(h1: float, h2: float,
              beta: float = 0.7, g_ep: float = 0.3) -> float:
    """T4: ℰ = h^T Γ h, Γ = [[β, γ_ep],[γ_ep, β]]"""
    return beta * (h1*h1 + h2*h2) + 2.0 * g_ep * h1 * h2


def many_worlds_tax(g: float, c0: float = 0.5) -> float:
    """T5: εQ_MWI = ½·c0·(1-c0)·γ²"""
    c0 = max(0.001, min(0.999, c0))
    return 0.5 * c0 * (1.0 - c0) * g * g


# ── Generalisation-aware learning rate ───────────────────────────────────────

class XgenEstimator:
    """
    X_gen(t) = 1 - EWMA_fast(h_held) / EWMA_slow(h_held)

    Measures whether the model generalises (held-out surprisal falling)
    vs memorises (train falls, held stays high) vs overwhelmed (both rise).

    Requires warmup of ~3/alpha_slow steps before X_gen is reliable.
    At alpha_slow=0.02, warmup ≈ 150 steps.
    At alpha_slow=0.005, warmup ≈ 600 steps — too slow for most uses.

    Default alpha_slow=0.02 balances responsiveness and stability.
    """

    def __init__(self, alpha_fast: float = 0.15, alpha_slow: float = 0.02):
        if not (0 < alpha_slow < alpha_fast < 1):
            raise ValueError(
                f"Need 0 < alpha_slow < alpha_fast < 1, "
                f"got alpha_slow={alpha_slow}, alpha_fast={alpha_fast}"
            )
        self.af  = alpha_fast
        self.as_ = alpha_slow
        self._reset()

    def update(self, h_train: float, h_held: float) -> dict:
        """Update with train and held-out surprisal. Returns X_gen state."""
        self._n += 1

        self._tf = self._ewma(self._tf, h_train, self.af)
        self._ts = self._ewma(self._ts, h_train, self.as_)
        self._hf = self._ewma(self._hf, h_held,  self.af)
        self._hs = self._ewma(self._hs, h_held,  self.as_)

        r_train    = self._tf / (self._ts + 1e-10)
        r_held     = self._hf / (self._hs + 1e-10)
        self.X_mem = max(0.0, min(1.0, 1.0 - r_train))
        self.X_gen = max(0.0, min(1.0, 1.0 - r_held))
        self.gap   = self.X_gen - self.X_mem

        # Only trust regime after warmup
        warmed = self._n >= int(3.0 / self.as_)
        self.regime = self._classify(r_train, r_held) if warmed else "WARMING"

        return {
            "X_gen": self.X_gen, "X_mem": self.X_mem,
            "gap":   self.gap,   "regime": self.regime,
        }

    def reset(self) -> None:
        self._reset()

    def _reset(self):
        self._tf = self._ts = self._hf = self._hs = None
        self._n  = 0
        self.X_gen = self.X_mem = self.gap = 0.0
        self.regime = "WARMING"

    def _classify(self, r_train: float, r_held: float) -> str:
        if self.X_gen > 0.10 and self.gap > -0.15:
            return "GENERALISING"
        if self.X_mem > 0.10 and self.gap < -0.20:
            return "MEMORISING"
        if r_train > 1.4 and r_held > 1.4:
            return "OVERWHELMED"
        return "STABLE"

    @staticmethod
    def _ewma(cur: Optional[float], val: float, a: float) -> float:
        return val if cur is None else a * val + (1.0 - a) * cur


# ── Clock ─────────────────────────────────────────────────────────────────────

def clock_tick(h: float, X_gen: float, scale: float = 1.0) -> float:
    """
    Δτ = h² · max(0, X_gen) · scale

    Non-linear: doubling h quadruples Δτ.
    Gate: zero if h=0 (nothing surprising) or X_gen≤0 (not generalising).
    """
    return (h * h) * max(0.0, X_gen) * scale
