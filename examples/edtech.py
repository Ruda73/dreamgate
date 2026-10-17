"""
examples/edtech.py

sigma is set to the measured mean prediction error in the stable phase.
tau_low=1.5 bits means the gate fires when err > 1.45 * sigma.
"""

import math, random, statistics
from dreamgate import stream, DREAM
from dreamgate.gate import GateConfig

ALPHA = 0.15

def measure_sigma(signal, warmup=50):
    pred, errors = None, []
    for i, v in enumerate(signal):
        if pred is None: pred = v
        elif i >= warmup: errors.append(abs(v - pred))
        pred = ALPHA * v + (1 - ALPHA) * pred
    return statistics.mean(errors)

def student_session(n=300, seed=42):
    random.seed(seed)
    out = []
    for i in range(n):
        if i < 100:   t = 2.0 + random.gauss(0, 0.3)
        elif i < 150: t = 8.0 + math.sin(i*0.3)*2 + random.gauss(0, 1.5)
        else:         t = 3.5 - (i-150)*0.02 + random.gauss(0, 0.5)
        out.append(max(0.5, t))
    return out

signal = student_session()
stable  = signal[:100]
sigma   = measure_sigma(stable)

cfg = GateConfig(sigma=sigma, tau_low=1.5, dream_sustain=6, clock_scale=0.1)
results = list(stream(signal, config=cfg))

phase = {"routine": 0, "novel": 0, "adapting": 0}
for i, r in enumerate(results):
    if r.state == DREAM:
        if   i < 100: phase["routine"] += 1
        elif i < 150: phase["novel"] += 1
        else:         phase["adapting"] += 1

print(f"EdTech  sigma={sigma:.4f}s  (measured from stable phase)")
print(f"DREAM — routine:{phase['routine']}  novel:{phase['novel']}  adapting:{phase['adapting']}")
print(f"Clock: {results[-1].clock:.4f}")
