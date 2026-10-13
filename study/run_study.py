"""Frame-rate study: error in the volume rate implied by N frames per cycle,
reconstructed with linear, periodic cubic spline and Fourier interpolation.
Every reconstruction goes through frames2foam.PeriodicMotion, the same code
path used to write OpenFOAM boundary data. Both reference motions are separable,
X(t) = base + f(t) * shape, and all three interpolants are linear in the data,
so interpolating every vertex is identical to interpolating f(t) and rebuilding
the surface. The study uses the cheap form; check_equivalence() confirms it
against full vertex-by-vertex interpolation."""
import sys, json, time
import numpy as np
sys.path.insert(0, '..')
from frames2foam import PeriodicMotion, surface_flux
from frames2foam.cli import volume_rate
from kinematics import Aneurysm, LeftVentricle, exact_rate, T

NS = [10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 128, 160, 200]
METHODS = ['linear', 'spline', 'fourier']
NOFF = 4                                    # frame phase offsets per N (fractions of a frame interval)
tf = np.linspace(0, T, 1601)[:-1]

def rate_polynomial(M):
    """For X = base + f*shape the surface flux is df * P(f) with P quadratic in f
    (area vectors are quadratic in position). Recover P exactly from three evaluations."""
    fs = np.array([0.0, 0.5, 1.0]) * (1.0 if M.name == 'aneurysm' else 4.0)
    X, U = M.from_factor(fs, np.ones(3))
    return np.polyfit(fs, surface_flux(X, U, M.tri), 2)

def check_equivalence(M, N=30, nt=60):
    tk = np.arange(N) * T / N; tt = np.linspace(0, T, nt, endpoint=False)
    for m in METHODS:
        full = volume_rate(PeriodicMotion(M.positions(tk), T, m), M.tri, tt)
        fk = M.factor(tk); mot = PeriodicMotion(np.repeat(fk[:, None, None], 3, axis=2), T, m)
        X, U = M.from_factor(mot.position(tt)[:, 0, 0], mot.velocity(tt)[:, 0, 0])
        cheap = surface_flux(X, U, M.tri)
        P = rate_polynomial(M)
        poly = mot.velocity(tt)[:, 0, 0] * np.polyval(P, mot.position(tt)[:, 0, 0])
        print(f'  polynomial form {M.name} {m}: max diff {np.abs(poly - cheap).max() / np.abs(cheap).max():.1e}', flush=True)
        print(f'  equivalence {M.name} {m}: max diff {np.abs(full - cheap).max() / np.abs(full).max():.1e}', flush=True)

def main():
    out = {}
    for M in (Aneurysm(), LeftVentricle()):
        check_equivalence(M)
        P = rate_polynomial(M)
        h = 1e-6
        ref = (M.factor(tf + h) - M.factor(tf - h)) / (2 * h) * np.polyval(P, M.factor(tf))
        print(f'  exact reference vs vertex-level exact: {np.abs(ref[::40] - exact_rate(M, tf[::40])).max() / np.abs(ref).max():.1e}', flush=True)
        peak = np.abs(ref).max()
        res = {m: {'N': NS, 'max': [], 'rms': [], 'peak': [], 'max_lo': [], 'max_hi': []} for m in METHODS}
        t0 = time.time()
        for N in NS:
            for m in METHODS:
                e_max, e_rms, e_pk = [], [], []
                for j in range(NOFF):
                    off = (j / NOFF) * T / N
                    tk = off + np.arange(N) * T / N
                    fk = M.factor(tk)
                    mot = PeriodicMotion(np.repeat(fk[:, None, None], 3, axis=2), T, m, t0=off)
                    r = mot.velocity(tf)[:, 0, 0] * np.polyval(P, mot.position(tf)[:, 0, 0])
                    e = r - ref
                    e_max.append(np.abs(e).max() / peak)
                    e_rms.append(np.sqrt(np.mean(e ** 2)) / peak)
                    # error in the largest-magnitude rate (what sets peak valve velocity)
                    e_pk.append((np.abs(r).max() - peak) / peak)
                d = res[m]
                d['max'].append(float(np.mean(e_max))); d['max_lo'].append(float(min(e_max))); d['max_hi'].append(float(max(e_max)))
                d['rms'].append(float(np.mean(e_rms))); d['peak'].append(float(np.mean(e_pk)))
            print(f'{M.name} N={N:3d}  ' + '  '.join(f'{m}: {100*res[m]["max"][-1]:6.2f}%' for m in METHODS), flush=True)
        out[M.name] = {'label': M.label, 'peak_rate_mL_s': float(peak * 1e6), 'results': res}
        print(M.name, 'done in %.0f s' % (time.time() - t0))
    json.dump(out, open('study_results.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
