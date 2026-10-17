"""
examples/security.py

sigma measured from legitimate user phase.
"""

import random, statistics
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

def keystrokes(n=200, takeover=120, seed=3):
    random.seed(seed)
    out = []
    for i in range(n):
        t = 150 + random.gauss(0,20) if i < takeover else 280 + random.gauss(0,60)
        out.append(max(50, t))
    return out

signal  = keystrokes()
sigma   = measure_sigma(signal[:120])
cfg     = GateConfig(sigma=sigma, tau_low=1.5, dream_sustain=5)

false_pos, first_detect = 0, None
for i, r in enumerate(stream(signal, config=cfg)):
    if r.state == DREAM:
        if i < 120: false_pos += 1
        elif first_detect is None: first_detect = i

print(f"Security  sigma={sigma:.2f}ms  (measured from legitimate user)")
print(f"False positives before takeover: {false_pos}")
if first_detect:
    print(f"First DREAM at tick {first_detect}  (+{first_detect-120} ticks after takeover)")
else:
    print("No DREAM after takeover")
