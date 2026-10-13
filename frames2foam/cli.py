"""Command line interface.

  frames2foam check  "frames/heart_*.obj" --period 0.8
  frames2foam write  "frames/heart_*.obj" --period 0.8 --case myCase --patch wall \
                     --method spline --dt 0.002 --cycles 2 [--reference-frame 0] [--scale 0.001]
"""
import argparse, json, sys
import numpy as np
from .io import read_frames
from .fit import PeriodicMotion
from .geometry import surface_flux, signed_volume, SurfaceMap
from .foam import read_patch_points, write_boundary_data, write_fourier_coeffs

def volume_rate(motion, tri, t, chunk=64):
    out = []
    for i in range(0, len(t), chunk):
        tt = t[i:i + chunk]
        out.append(surface_flux(motion.position(tt), motion.velocity(tt), tri))
    return np.concatenate(out)

def output_sampling_error(motion, tri, period, n_candidates=(100, 200, 400, 800, 1600, 3200), nfine=6400):
    """OpenFOAM interpolates boundary data linearly between the written samples, so the
    written motion is itself a staircase in velocity. For each candidate number of samples
    per cycle, estimate the max error (% of peak) of the written volume rate against the
    smooth fit: the chord slope of V(t) over each sample interval versus the instantaneous
    rate within it. Returns {samples_per_cycle: error_pct}."""
    from .geometry import signed_volume
    t = np.arange(nfine) * period / nfine
    V = np.concatenate([signed_volume(motion.position(t[i:i + 64]), tri) for i in range(0, nfine, 64)])
    r = volume_rate(motion, tri, t[::max(nfine // 3200, 1)])
    peak = np.abs(r).max()
    rf = np.interp(t, t[::max(nfine // 3200, 1)], r, period=period)
    out = {}
    for n in n_candidates:
        if nfine % n:
            continue
        s = nfine // n
        Vs = np.r_[V[::s], V[0]]
        chord = np.diff(Vs) / (period / n)
        out[n] = float(100 * np.abs(np.repeat(chord, s) - rf).max() / peak)
    return out

def fourier_coefficients(motion, smap, ref, nsamp, nharm=None, vel_tol=1e-3):
    """Fourier coefficients of the patch displacement implied by a fitted motion.
    The motion is sampled at nsamp uniform phases from motion.t0 and mapped to the patch
    (both operations are linear, so the coefficients are exact for the sampled series).
    nharm=None keeps harmonics until the truncated velocity has a relative RMS error
    below vel_tol. Returns A0, A, B, K, achieved velocity error."""
    T = motion.T
    ts = motion.t0 + np.arange(nsamp) * T / nsamp
    D = np.concatenate([smap(motion.position(ts[i:i + 32]) - ref) for i in range(0, nsamp, 32)])
    c = np.fft.rfft(D, axis=0) / nsamp                               # [nsamp//2+1, np, 3]
    kmax = nsamp // 2
    p = (np.arange(len(c))[:, None, None] ** 2 * np.abs(c) ** 2).reshape(len(c), -1).sum(1)
    w = np.full(len(c), 2.0); w[0] = 1.0
    if nsamp % 2 == 0:
        w[kmax] = 1.0
    pw = w * p                                                       # velocity power per harmonic
    tail = np.cumsum(pw[::-1])[::-1]                                 # power from harmonic k upwards
    if nharm is None:
        ok = np.nonzero(tail[1:] <= vel_tol ** 2 * pw.sum())[0]
        K = int(ok[0]) if len(ok) else kmax
    else:
        K = min(int(nharm), kmax)
    K = max(K, 1)
    err = float(np.sqrt(tail[K + 1] / pw.sum())) if K + 1 < len(tail) else 0.0
    A0 = np.real(c[0])
    A = w[1:K + 1, None, None] * np.real(c[1:K + 1])
    B = -w[1:K + 1, None, None] * np.imag(c[1:K + 1])
    return A0, A, B, K, err

def choose_dt(motion, tri, period, tol_pct):
    err = output_sampling_error(motion, tri, period)
    ok = [n for n, e in sorted(err.items()) if e <= tol_pct]
    n = ok[0] if ok else max(err)
    return period / n, err

def check(X, tri, period, nfine=2000):
    """Compare the volume rate implied by linear, spline and Fourier fits of the frames.
    Their spread is a direct indicator of whether the frame rate resolves the motion."""
    t = np.linspace(0, period, nfine, endpoint=False)
    rates = {m: volume_rate(PeriodicMotion(X, period, m), tri, t) for m in ('linear', 'spline', 'fourier')}
    vol = signed_volume(X, tri)
    peak = np.abs(rates['spline']).max()
    rep = {
        'frames': int(len(PeriodicMotion(X, period).X)),
        'frame_interval_s': period / len(PeriodicMotion(X, period).X),
        'volume_min': float(vol.min()), 'volume_max': float(vol.max()),
        'stroke_volume': float(vol.max() - vol.min()),
        'peak_volume_rate_spline': float(peak),
        'linear_vs_spline_max_diff_pct_of_peak': float(100 * np.abs(rates['linear'] - rates['spline']).max() / peak),
        'fourier_vs_spline_max_diff_pct_of_peak': float(100 * np.abs(rates['fourier'] - rates['spline']).max() / peak),
        'closed_surface': bool(abs(vol.mean()) > 0),
    }
    err = output_sampling_error(PeriodicMotion(X, period, 'spline'), tri, period)
    rep['written_sample_error_pct_by_samples_per_cycle'] = {str(k): round(v, 3) for k, v in err.items()}
    d = rep['fourier_vs_spline_max_diff_pct_of_peak']
    rep['verdict'] = ('smooth fits agree; frame rate resolves the motion' if d < 2 else
                      'smooth fits disagree by %.1f%% of peak; motion is under-resolved in time, '
                      'more frames are recommended' % d)
    return rep, t, rates

def main(argv=None):
    ap = argparse.ArgumentParser(prog='frames2foam', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('check', 'write'):
        p = sub.add_parser(name)
        p.add_argument('pattern', help='glob for frame files, e.g. "frames/f_*.obj"')
        p.add_argument('--period', type=float, required=True, help='cycle duration [s]')
        p.add_argument('--scale', type=float, default=1.0, help='multiply coordinates (e.g. 0.001 for mm to m)')
        p.add_argument('--last-is-first', action='store_true', help='last frame repeats the first; drop it')
    w = sub.choices['write']
    w.add_argument('--case', required=True); w.add_argument('--patch', required=True)
    w.add_argument('--method', default='spline', choices=['spline', 'fourier', 'linear'])
    w.add_argument('--nharm', default=None, help="Fourier harmonics kept: an integer, or 'auto' to cut at the noise floor")
    w.add_argument('--dt', default='auto', help="output sampling interval [s], or 'auto' to meet --rate-tol")
    w.add_argument('--rate-tol', type=float, default=1.0, help='target max error of the written volume rate, %% of peak (with --dt auto)')
    w.add_argument('--cycles', type=float, default=1.0, help='number of cycles to write')
    w.add_argument('--reference-frame', type=int, default=0, help='frame the OpenFOAM mesh was built from')
    w.add_argument('--format', default='mapped', choices=['mapped', 'fourier'],
                   help="mapped: samples for timeVaryingMappedFixedValue; fourier: coefficients for the "
                        "fourierDisplacement boundary condition (openfoam/ in this repository), evaluated "
                        "smoothly at every solver time step")
    w.add_argument('--vel-tol', type=float, default=1e-3,
                   help='fourier format: relative RMS error allowed in the wall velocity from truncating harmonics')
    w.add_argument('--max-map-distance', type=float, default=None, help='fail if a patch point is farther than this from the surface')
    a = ap.parse_args(argv)

    X, tri, files = read_frames(a.pattern, a.scale)
    if a.last_is_first:
        X = X[:-1]
    print(f'read {len(X)} frames, {X.shape[1]} vertices, {len(tri)} triangles', file=sys.stderr)
    if a.cmd == 'check':
        rep, _, _ = check(X, tri, a.period)
        print(json.dumps(rep, indent=2)); return 0

    nh = a.nharm if a.nharm in (None, 'auto') else int(a.nharm)
    motion = PeriodicMotion(X, a.period, a.method, nharm=nh)
    if a.method == 'fourier':
        print(f'Fourier fit keeps {motion.nharm} of {len(X) // 2} harmonics', file=sys.stderr)
    if a.format == 'fourier' and a.method == 'linear':
        raise SystemExit('--format fourier needs a smooth fit; use --method spline or fourier')
    pts = read_patch_points(a.case, a.patch, order='patch' if a.format == 'fourier' else 'sorted')
    ref = X[a.reference_frame]
    smap = SurfaceMap(ref, tri, pts)
    span = np.ptp(ref, axis=0).max()
    print(f'mapped {len(pts)} patch points, max distance to frame surface {smap.distance.max():.3g} '
          f'({100 * smap.distance.max() / span:.2g}% of model size)', file=sys.stderr)
    if a.max_map_distance is not None and smap.distance.max() > a.max_map_distance:
        raise SystemExit('patch points do not lie on the reference frame surface; check units and frame index')
    if a.format == 'fourier':
        if a.method == 'fourier':
            A0, A, B, K, err = fourier_coefficients(motion, smap, ref, motion.n, nharm=motion.nharm)
        else:
            nsamp = max(512, 16 * motion.n); nsamp += nsamp % 2
            A0, A, B, K, err = fourier_coefficients(motion, smap, ref, nsamp, vel_tol=a.vel_tol)
        path = write_fourier_coeffs(a.case, a.patch, pts, a.period, motion.t0, A0, A, B,
                                    note=f'{len(X)} frames, {a.method} fit, {K} harmonics')
        print(f'kept {K} harmonics (wall velocity truncation error {100 * err:.3g}% RMS); wrote {path}',
              file=sys.stderr)
        return 0
    if a.dt == 'auto':
        dt, err = choose_dt(motion, tri, a.period, a.rate_tol)
        print('written-sample error by samples per cycle: ' +
              ', '.join(f'{n}: {e:.2f}%' for n, e in sorted(err.items())), file=sys.stderr)
        print(f'chose dt = {dt:.4g} s ({round(a.period / dt)} samples per cycle)', file=sys.stderr)
    else:
        dt = float(a.dt)
    times = np.round(np.arange(0.0, a.cycles * a.period + dt * 1.5, dt), 10)
    vals = []
    for i in range(0, len(times), 64):
        P = motion.position(times[i:i + 64])
        vals.extend(smap(P - ref))
    root = write_boundary_data(a.case, a.patch, pts, times, vals)
    print(f'wrote {len(times)} samples to {root}', file=sys.stderr)
    return 0

if __name__ == '__main__':
    sys.exit(main())
