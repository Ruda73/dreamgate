# DreamGate

Detects when a signal has genuinely changed state — not when a rule was violated.

```python
from dreamgate import DreamGate, stream, measure, DREAM

# Streaming
for result in stream(my_signal):
    if result.state == DREAM:
        act(result)

# Batch
summary = measure(my_signal)
print(summary['first_dream'])
print(summary['state_pct'])
```

## Install

```bash
pip install dreamgate
```

## How it works

At each tick the gate computes γ — the state overlap between the predicted
and actual signal. From γ, five thermodynamic quantities are derived:

| Theorem | Expression | Meaning |
|---|---|---|
| Helstrom | P_err = ½(1−√(1−γ²)) | Min discrimination error |
| Bures | d²_B ≈ γ² | Distance on statistical manifold |
| Feinstein | εQ = ½kBTγ² | Min thermodynamic cost |
| Epistatic | ℰ = h^TΓh | Cross-channel coupling |
| Many Worlds | εQ_MWI = ½kBTΣ\|ci\|²\|cj\|²γij² | Branch accessibility |

The learning clock advances as `Δτ = h(s)² · X_gen(t)` — zero when nothing
is surprising, zero when the model is memorising rather than generalising.

## Three states

- **NORMAL** — prior confirmed, signal predictable
- **HOLD**   — anomaly detected, watching
- **DREAM**  — sustained anomaly, hypothesis warranted

## Calibration

Set `sigma` to the mean prediction error in your signal's stable phase:

```python
import statistics

def measure_sigma(stable_signal, alpha=0.15, warmup=50):
    pred, errors = None, []
    for i, v in enumerate(stable_signal):
        if pred is None: pred = v
        elif i >= warmup: errors.append(abs(v - pred))
        pred = alpha * v + (1 - alpha) * pred
    return statistics.mean(errors)

from dreamgate import DreamGate
from dreamgate.gate import GateConfig

sigma = measure_sigma(my_stable_signal)
gate  = DreamGate(config=GateConfig(sigma=sigma, tau_low=1.5))
```

## Pre-calibrated vertical configs

```python
from dreamgate.config import INDUSTRIAL, EDTECH, SECURITY, GAMING
```

## Dual-channel generalisation (X_gen)

Provide a held-out channel to distinguish genuine learning from memorisation:

```python
for result in stream(train_signal, held_out=held_signal):
    print(result.X_gen, result.regime)
```

## Theory

van Rensburg (2026). *εQ(MB): The Thermodynamic Cost of Maintaining a
Quantum Markov Blanket Under Non-Orthogonal Perturbations.* arXiv preprint.

GitHub: https://github.com/Predictive-Entropy-Cryptography/dreamgate
