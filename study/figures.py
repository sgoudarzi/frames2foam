"""Figures for the frame-rate study."""
import sys, json
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '..')
from frames2foam import PeriodicMotion
from kinematics import Aneurysm, LeftVentricle, T
from run_study import rate_polynomial

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
C = {'spline': '#2a78d6', 'linear': '#eb6834', 'fourier': '#1baf7a', 'auto': '#eda100'}
MK = {'spline': 'o', 'linear': 's', 'fourier': '^', 'auto': 'D'}
NAME = {'linear': 'Linear (common default)', 'spline': 'Periodic cubic spline', 'fourier': 'Fourier (all harmonics)',
        'auto': 'Fourier, automatic cut-off'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'legend.frameon': False})
FLOOR = 1e-3                                         # % ; plotting floor for spectrally exact results

def style_n_axis(ax):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([10, 15, 20, 30, 40, 60, 100, 200]); ax.set_xticklabels(['10', '15', '20', '30', '40', '60', '100', '200'])
    ax.minorticks_off()
    ax.axvspan(30, 60, color='#ebeae6', zorder=0, lw=0)
    ax.set_xlabel('Frames per cardiac cycle')

def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color=INK)

# ---------- Figure 1: clean convergence ----------
S = json.load(open('study_results.json'))
fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
for ax, key in zip(axs, ['lv', 'aneurysm']):
    d = S[key]; N = np.array(d['results']['linear']['N'])
    for m in ['linear', 'spline', 'fourier']:
        r = d['results'][m]
        y = np.maximum(100 * np.array(r['max']), FLOOR)
        lo = np.maximum(100 * np.array(r['max_lo']), FLOOR); hi = np.maximum(100 * np.array(r['max_hi']), FLOOR)
        ax.fill_between(N, lo, hi, color=C[m], alpha=0.15, lw=0)
        ax.plot(N, y, color=C[m], lw=2, marker=MK[m], ms=4.5, mec=SURF, mew=1, label=NAME[m])
    style_n_axis(ax)
    ax.set_ylim(FLOOR * 0.7, 900)
    ax.axhline(5, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(10.3, 5.6, '5 %', fontsize=7.5, color=INK2)
    ax.text(42.4, 820, '30–60 frames\n(typical animation)', ha='center', va='top', fontsize=7.5, color=INK2)
    ax.set_title(d['label'] + f'  (peak rate {d["peak_rate_mL_s"]:.3g} mL/s)', fontsize=9.5, loc='left', color=INK)
axs[0].set_ylabel('Max error in volume rate\n(% of peak)')
axs[0].set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100]); axs[0].set_yticklabels(['≤0.001', '0.01', '0.1', '1', '10', '100'])
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=8.5)
fig.suptitle('How many frames does a moving-wall CFD case need?  Error in dV/dt implied by the interpolated motion',
             fontsize=10.5, x=0.02, ha='left', color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
fig.savefig('figures/fig1_convergence.png', dpi=220); plt.close(fig)

# ---------- Figure 2: what linear interpolation does to the flow rate ----------
M = LeftVentricle(); P = rate_polynomial(M)
tf = np.linspace(0, T, 3201)[:-1]; h = 1e-6
ref = (M.factor(tf + h) - M.factor(tf - h)) / (2 * h) * np.polyval(P, M.factor(tf))
N = 30; tk = np.arange(N) * T / N
fig, ax = plt.subplots(figsize=(9.0, 3.2))
ax.plot(tf, -1e6 * ref, color=INK, lw=1.2, label='Exact motion')
for m in ['linear', 'spline']:
    mot = PeriodicMotion(np.repeat(M.factor(tk)[:, None, None], 3, axis=2), T, m)
    r = mot.velocity(tf)[:, 0, 0] * np.polyval(P, mot.position(tf)[:, 0, 0])
    ax.plot(tf, -1e6 * r, color=C[m], lw=2 if m == 'linear' else 1.6, ls='-' if m == 'linear' else (0, (5, 2)),
            label=f'{NAME[m]}, {N} frames')
for t in tk:
    ax.axvline(t, color=GRID, lw=0.6, zorder=0)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xlim(0, T); ax.set_xlabel('Time in cycle [s]'); ax.set_ylabel('Outflow implied by wall motion\n−dV/dt [mL/s]')
ax.text(0.13, 555, 'Ejection', ha='center', fontsize=8, color=INK2)
ax.text(0.46, -610, 'E wave', ha='center', fontsize=8, color=INK2)
ax.text(0.73, -320, 'A wave', ha='center', fontsize=8, color=INK2)
ax.set_ylim(-680, 600)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('Idealised left ventricle, 30 frames per cycle (grey lines). Linear interpolation turns the flow rate into a staircase.',
             fontsize=9.5, loc='left', color=INK)
fig.tight_layout(); fig.savefig('figures/fig2_staircase.png', dpi=220); plt.close(fig)

# ---------- Figure 3: noise ----------
Z = json.load(open('noise_results.json'))
key = {'linear': 'linear', 'spline': 'spline', 'fourier': 'fourier (all harmonics)', 'auto': 'fourier (auto cut-off)'}
fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
for ax, k in zip(axs, ['lv', 'aneurysm']):
    d = Z[k]; N = np.array(d['N'])
    for m in ['linear', 'spline', 'fourier', 'auto']:
        ax.plot(N, 100 * np.array(d['results'][key[m]]), color=C[m], lw=2, marker=MK[m], ms=4.5, mec=SURF, mew=1, label=NAME[m])
    style_n_axis(ax); ax.set_ylim(2, 250)
    ax.set_yticks([2, 5, 10, 20, 50, 100, 200]); ax.set_yticklabels(['2', '5', '10', '20', '50', '100', '200'])
    ax.set_title(S[k]['label'], fontsize=9.5, loc='left', color=INK)
axs[0].set_ylabel('Max error in volume rate\n(% of peak, median of 24 trials)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=8.2)
fig.suptitle('Frames carrying 1 % positional jitter: more frames make exact interpolants worse; a filtered fit converges',
             fontsize=10.5, x=0.02, ha='left', color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
fig.savefig('figures/fig3_noise.png', dpi=220); plt.close(fig)
print('figures written')

# ---------- Figure 4: CFD confirmation ----------
import os
if os.path.exists('cfd_results.json'):
    R = json.load(open('cfd_results.json'))
    runs = [('linear', 'linear', 'Linear fit, written every 2 ms'),
            ('spline_2ms', 'spline', 'Spline fit, written every 2 ms'),
            ('spline_auto', 'spline', 'Spline fit, written every 0.5 ms (auto)'),
            ('fourier_bc', 'fourier', 'Spline fit, fourierDisplacement at run time')]
    runs = [r for r in runs if r[0] in R]
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.2), sharey=True, sharex=True)
    axs = axs.ravel()
    for ax, (key, m, ttl) in zip(axs, runs):
        d = R[key]; t = np.array(d['t']) - T
        o = np.argsort(t)
        ax.plot(t[o], 1e6 * np.array(d['exact'])[o], color=INK, lw=1.2, label='Exact wall motion')
        ax.plot(t[o], 1e6 * np.array(d['net'])[o], color=C[m], lw=0, marker=MK[m], ms=3.6, mec=SURF, mew=0.7,
                label='CFD: inflow minus outflow')
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f'{ttl}\nmax error {d["cfd_vs_exact_max_pct"]:.1f} % of peak', fontsize=9, loc='left', color=INK)
        ax.set_xlim(0, 0.6)
    for ax in axs[2:]:
        ax.set_xlabel('Time in cycle [s]')
    for ax in axs[::2]:
        ax.set_ylabel('Rate of sac volume change\ndV/dt [mL/s]')
    axs[1].legend(loc='upper right', fontsize=7.5)
    fig.suptitle('OpenFOAM aneurysm case driven by the same 30 frames (cycle 2 of 2)', fontsize=10.5, x=0.02, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig('figures/fig4_cfd.png', dpi=220); plt.close(fig)
    print('cfd figure written')
