"""
dreamgate/config.py — Default configurations and calibration.

Pre-calibrated GateConfig instances for common signal domains.

The most important function here is calibrate():

    config = calibrate(reference_signal)
    gate   = DreamGate(config)

This fits sigma to your actual data so you don't have to guess.
"""

import math
from .gate import GateConfig


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(
    reference: list,
    *,
    tau_low:       float = 1.5,
    dream_sustain: int   = 8,
    hysteresis:    float = 0.5,
    alpha_fast:    float = 0.15,
    alpha_slow:    float = 0.02,
    clock_scale:   float = 0.01,
    adaptive_sigma: bool = False,
    percentile:    float = 0.68,
) -> GateConfig:
    """
    Fit sigma from a reference window of NORMAL-operating signal.

    Records prediction errors using the same EWMA predictor as DreamGate,
    then sets sigma to the given percentile of those errors (default: 68th
    percentile, i.e. roughly one sigma of the error distribution).

    Args:
        reference:     list of signal values from a known-normal period.
                       Minimum 30 values recommended; 200+ for stable estimates.
        tau_low:       surprisal threshold (bits). Default 1.5.
        dream_sustain: ticks in HOLD before DREAM. Default 8.
        hysteresis:    fraction of tau_low for return to NORMAL. Default 0.5.
        alpha_fast:    EWMA coefficient for predictor and X_gen fast track.
        alpha_slow:    EWMA coefficient for X_gen slow track.
        clock_scale:   learning clock scale.
        adaptive_sigma: pass through to GateConfig.
        percentile:    error percentile used to set sigma (0–1). Default 0.68.

    Returns:
        GateConfig with sigma fitted to your signal.

    Example — bearing vibration:
        normal_30min = load_csv('bearing_baseline.csv')['rms'].tolist()
        config = calibrate(normal_30min)
        gate   = DreamGate(config)

    Example — tighter sensitivity (fewer false positives):
        config = calibrate(reference, percentile=0.90, tau_low=2.0)

    Example — more sensitive (catch subtle drifts early):
        config = calibrate(reference, percentile=0.50, tau_low=1.2)
    """
    if len(reference) < 5:
        raise ValueError(
            f"calibrate() needs at least 5 reference values, got {len(reference)}. "
            f"For stable sigma estimates, use 200+ values."
        )
    if not (0 < percentile < 1):
        raise ValueError(f"percentile must be in (0, 1), got {percentile}")

    # Replay EWMA predictor to collect errors
    errors = []
    pred = reference[0]
    for v in reference[1:]:
        err = abs(v - pred)
        errors.append(err)
        pred = alpha_fast * v + (1.0 - alpha_fast) * pred

    errors.sort()
    idx   = max(0, min(len(errors) - 1, int(percentile * len(errors))))
    sigma = max(1e-6, errors[idx])

    if len(reference) < 30:
        import warnings
        warnings.warn(
            f"calibrate() used only {len(reference)} values. "
            f"Sigma estimate ({sigma:.4f}) may be unstable. "
            f"Use 200+ values from a known-normal period.",
            UserWarning, stacklevel=2
        )

    return GateConfig(
        sigma=sigma,
        tau_low=tau_low,
        dream_sustain=dream_sustain,
        hysteresis=hysteresis,
        adaptive_sigma=adaptive_sigma,
        alpha_fast=alpha_fast,
        alpha_slow=alpha_slow,
        clock_scale=clock_scale,
    )


# ── Pre-calibrated vertical configs ──────────────────────────────────────────
# These assume sigma=0.5 as a neutral starting point.
# Replace with calibrate() on your actual data for production use.

def industrial_config() -> GateConfig:
    """
    Slow-moving industrial sensor (vibration RMS, temperature, flow).
    High dream_sustain = fewer false alarms on slow drift.
    tau_low=2.0 = only fires on genuine > 1.67 sigma error.
    """
    return GateConfig(
        sigma=0.5, tau_low=2.0, dream_sustain=10,
        hysteresis=0.4, adaptive_sigma=True, sigma_alpha=0.005,
        clock_scale=0.01
    )


def edtech_config() -> GateConfig:
    """
    EdTech learner signal (response time, answer correctness, session length).
    Faster signal, more tolerance for natural spikes.
    """
    return GateConfig(
        sigma=0.5, tau_low=1.5, dream_sustain=6,
        hysteresis=0.5, adaptive_sigma=False,
        clock_scale=0.05
    )


def security_config() -> GateConfig:
    """
    Security / authentication signal (login timing, request rate, entropy).
    Low tolerance, fast response. Any sustained deviation is suspicious.
    """
    return GateConfig(
        sigma=0.5, tau_low=1.2, dream_sustain=5,
        hysteresis=0.4, adaptive_sigma=False,
        clock_scale=0.01
    )


def gaming_config() -> GateConfig:
    """
    Gaming flow state (reaction time, accuracy, session engagement).
    Moderate sensitivity; adaptive sigma tracks session warm-up.
    """
    return GateConfig(
        sigma=0.5, tau_low=1.5, dream_sustain=8,
        hysteresis=0.5, adaptive_sigma=True, sigma_alpha=0.02,
        clock_scale=0.02
    )
