"""CFD confirmation: the aneurysm case run from the same 30 OBJ frames with wall
boundary data written by frames2foam in three ways:
  linear       --method linear, samples every 2 ms
  spline_2ms   --method spline, samples every 2 ms
  spline_auto  --method spline, --dt auto --rate-tol 1.2 (samples every 0.5 ms)
  fourier_bc   --method spline --format fourier, fourierDisplacement condition (OpenFOAM v2512)
Outflow = inflow - dV/dt for an incompressible fluid, so any error in the
written wall motion appears directly in the outflow.
Usage: python3 cfd_compare.py linear=<case> spline_2ms=<case> spline_auto=<case> fourier_bc=<case>"""
import sys, json, warnings
import numpy as np, pyvista as pv
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')
from kinematics import Aneurysm, T, exact_rate
from frames2foam import PeriodicMotion, read_frames
from frames2foam.cli import volume_rate

def patch_flux(pd, outward):
    pd = pd.extract_surface(algorithm=None).compute_cell_sizes(length=False, volume=False)
    pd = pd.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=False)
    n = np.asarray(pd.cell_data['Normals']); A = np.asarray(pd.cell_data['Area'])
    n = n * np.sign(n @ outward)[:, None]
    return float(np.sum(np.einsum('ij,ij->i', np.asarray(pd.cell_data['U']), n) * A))

def fluxes(case):
    open(f'{case}/case.foam', 'a').close()
    r = pv.OpenFOAMReader(f'{case}/case.foam'); r.enable_all_patch_arrays()
    tv = np.array(r.time_values); tv = tv[(tv >= T - 1e-9) & (tv <= 2 * T + 1e-9)]
    out = []
    for t in tv:
        r.set_active_time_value(float(t)); b = r.read()['boundary']
        qi = -patch_flux(b['inlet'], np.array([0, 0, -1.0])); qo = patch_flux(b['outlet'], np.array([0, 0, 1.0]))
        out.append((t, qi, qo))
    return np.array(out)

cases = dict(a.split('=', 1) for a in sys.argv[1:])
X, tri, _ = read_frames('frames30/aneurysm_*.obj')
M = Aneurysm(nth=96, nz=301)
res = {}
for m, case in cases.items():
    d = fluxes(case); t = d[:, 0]
    net = d[:, 1] - d[:, 2]                                   # = dV/dt realised by the CFD
    exact = exact_rate(M, t)
    recon = volume_rate(PeriodicMotion(X, T, 'linear' if m == 'linear' else 'spline'), tri, t)
    pk = np.abs(exact).max()
    res[m] = dict(t=t.tolist(), Q_in=d[:, 1].tolist(), Q_out=d[:, 2].tolist(), net=net.tolist(),
                  exact=exact.tolist(), reconstructed=recon.tolist(),
                  cfd_vs_exact_max_pct=float(100 * np.abs(net - exact).max() / pk),
                  cfd_vs_reconstructed_max_pct=float(100 * np.abs(net - recon).max() / pk),
                  outflow_error_max_pct_of_peak_outflow=float(100 * np.abs((d[:, 1] - exact) - d[:, 2]).max() / d[:, 2].max()))
    print(m, {k: round(v, 3) for k, v in res[m].items() if isinstance(v, float)})
json.dump(res, open('cfd_results.json', 'w'))
