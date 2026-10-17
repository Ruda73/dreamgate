"""
examples/gaming.py

sigma measured from flow state phase.
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

def session(n=400, seed=11):
    random.seed(seed)
    out = []
    for i in range(n):
        if i < 80:    t = 400 + random.gauss(0, 60)
        elif i < 200: t = 220 + math.sin(i*0.2)*20 + random.gauss(0, 25)
        elif i < 280: t = 550 + random.gauss(0, 80)
        else:         t = 300 + random.gauss(0, 40)
        out.append(max(100, t))
    return out

signal = session()
sigma  = measure_sigma(signal[80:200])    # flow phase is the stable baseline
cfg    = GateConfig(sigma=sigma, tau_low=1.5, dream_sustain=8)

phase = {"warmup": 0, "flow": 0, "spike": 0, "recovery": 0}
for i, r in enumerate(stream(signal, config=cfg)):
    if r.state == DREAM:
        if   i < 80:  phase["warmup"] += 1
        elif i < 200: phase["flow"] += 1
        elif i < 280: phase["spike"] += 1
        else:         phase["recovery"] += 1

print(f"Gaming  sigma={sigma:.2f}ms  (measured from flow phase)")
for k, v in phase.items():
    print(f"  DREAM {k:10s}: {v:3d}  {'█'*min(v//2,30)}")
