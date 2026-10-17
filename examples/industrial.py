"""
examples/industrial.py

sigma measured from normal operating phase.
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

def bearing(n=500, seed=7):
    random.seed(seed)
    out = []
    for i in range(n):
        base = math.sin(i*0.2)*0.4
        if i < 300:   noise = random.gauss(0, 0.06)
        elif i < 400: noise = random.gauss(0, 0.15) + math.sin(i*1.3)*0.3*((i-300)/100)
        else:         noise = random.gauss(0, 0.35) + math.sin(i*1.3)*0.9
        out.append(base + noise)
    return out

signal = bearing()
sigma  = measure_sigma(signal[:300])
cfg    = GateConfig(sigma=sigma, tau_low=2.0, dream_sustain=10)

phase = {"normal": 0, "degrading": 0, "fault": 0}
first_dream = None
for i, r in enumerate(stream(signal, config=cfg)):
    if r.state == DREAM:
        if first_dream is None: first_dream = i
        if   i < 300: phase["normal"] += 1
        elif i < 400: phase["degrading"] += 1
        else:         phase["fault"] += 1

print(f"Industrial  sigma={sigma:.4f}  (measured from normal phase)")
print(f"DREAM — normal:{phase['normal']}  degrading:{phase['degrading']}  fault:{phase['fault']}")
print(f"First DREAM at tick: {first_dream}")
