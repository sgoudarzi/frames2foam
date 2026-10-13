"""Robustness to frame noise: animation exports and segmented imaging carry small
vertex jitter. Velocity is a time derivative, so jitter is amplified as frames get
closer together. Compare methods, including a low-pass Fourier fit (frames2foam
--nharm), for frames carrying Gaussian noise of 1 % of the motion amplitude."""
import sys, json
import numpy as np
sys.path.insert(0, '..')
from frames2foam import PeriodicMotion
from kinematics import Aneurysm, LeftVentricle, T
from run_study import rate_polynomial

NS = [20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 200]
NOISE, NREAL = 0.01, 24
tf = np.linspace(0, T, 1601)[:-1]
rng = np.random.default_rng(7)
out = {}
for M in (Aneurysm(), LeftVentricle()):
    P = rate_polynomial(M); h = 1e-6
    ref = (M.factor(tf + h) - M.factor(tf - h)) / (2 * h) * np.polyval(P, M.factor(tf)); peak = np.abs(ref).max()
    famp = np.ptp(M.factor(tf))
    meth = {'linear': dict(method='linear'), 'spline': dict(method='spline'),
            'fourier (all harmonics)': dict(method='fourier'), 'fourier (auto cut-off)': dict(method='fourier', nharm='auto')}
    res = {k: [] for k in meth}
    for N in NS:
        tk = np.arange(N) * T / N
        for k, kw in meth.items():
            e = []
            for _ in range(NREAL):
                fk = M.factor(tk) + NOISE * famp * rng.standard_normal(N)
                mot = PeriodicMotion(np.repeat(fk[:, None, None], 3, axis=2), T, **kw)
                r = mot.velocity(tf)[:, 0, 0] * np.polyval(P, mot.position(tf)[:, 0, 0])
                e.append(np.abs(r - ref).max() / peak)
            res[k].append(float(np.median(e)))
        print(M.name, N, '  '.join(f'{k}: {100*res[k][-1]:6.1f}%' for k in meth), flush=True)
    out[M.name] = {'N': NS, 'noise_frac': NOISE, 'results': res}
json.dump(out, open('noise_results.json', 'w'), indent=1)
