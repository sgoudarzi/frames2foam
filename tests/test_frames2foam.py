import os, numpy as np, pytest
from frames2foam import PeriodicMotion, surface_flux, signed_volume, SurfaceMap, read_frames
from frames2foam.fit import auto_harmonics
from frames2foam.cli import main, volume_rate

T = 1.0

def sphere(n=20):
    u = np.linspace(0, np.pi, n)[1:-1]; v = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    U, V = np.meshgrid(u, v, indexing='ij')
    X = np.c_[np.sin(U.ravel()) * np.cos(V.ravel()), np.sin(U.ravel()) * np.sin(V.ravel()), np.cos(U.ravel())]
    X = np.vstack([X, [[0, 0, 1], [0, 0, -1]]])
    idx = np.arange(len(u) * len(v)).reshape(len(u), len(v)); nv = len(v); top, bot = len(X) - 2, len(X) - 1
    tri = [(top, idx[0, j], idx[0, (j + 1) % nv]) for j in range(nv)]
    for i in range(len(u) - 1):
        for j in range(nv):
            a, b, c, d = idx[i, j], idx[i, (j + 1) % nv], idx[i + 1, j], idx[i + 1, (j + 1) % nv]
            tri += [(a, c, d), (a, d, b)]
    tri += [(bot, idx[-1, (j + 1) % nv], idx[-1, j]) for j in range(nv)]
    return X, np.array(tri)

def scale(t):
    return 1 + 0.2 * np.sin(2 * np.pi * t / T) + 0.05 * np.cos(6 * np.pi * t / T)

def frames(N):
    X0, tri = sphere(); tk = np.arange(N) * T / N
    return scale(tk)[:, None, None] * X0, tri, X0, tk

def test_volume_positive_and_flux_matches_derivative():
    X, tri, X0, _ = frames(8)
    assert signed_volume(X0, tri) > 0
    t = 0.3; h = 1e-6
    Xt = scale(t) * X0; Ut = (scale(t + h) - scale(t - h)) / (2 * h) * X0
    dv = (signed_volume(scale(t + h) * X0, tri) - signed_volume(scale(t - h) * X0, tri)) / (2 * h)
    assert np.isclose(surface_flux(Xt, Ut, tri), dv, rtol=1e-6)

@pytest.mark.parametrize('method', ['linear', 'spline', 'fourier'])
def test_fits_pass_through_frames(method):
    X, tri, _, tk = frames(12)
    m = PeriodicMotion(X, T, method)
    assert np.allclose(m.position(tk), X, atol=1e-12)
    assert np.allclose(m.position(tk + T), X, atol=1e-12)          # periodic

def test_fourier_exact_for_bandlimited_motion():
    X, tri, X0, _ = frames(16)
    m = PeriodicMotion(X, T, 'fourier')
    t = np.linspace(0, T, 50)
    assert np.allclose(m.position(t), scale(t)[:, None, None] * X0, atol=1e-12)

def test_spline_beats_linear_on_rate():
    X, tri, X0, _ = frames(24)
    t = np.linspace(0, T, 400, endpoint=False); h = 1e-6
    exact = surface_flux(scale(t)[:, None, None] * X0, ((scale(t + h) - scale(t - h)) / (2 * h))[:, None, None] * X0, tri)
    e = {m: np.abs(volume_rate(PeriodicMotion(X, T, m), tri, t) - exact).max() for m in ('linear', 'spline')}
    assert e['spline'] < 0.1 * e['linear']

def test_closed_loop_frame_dropped():
    X, tri, _, _ = frames(10)
    m = PeriodicMotion(np.concatenate([X, X[:1]]), T)
    assert m.n == 10

def test_auto_harmonics_clean_and_noisy():
    X, tri, X0, tk = frames(64)
    assert auto_harmonics(np.fft.rfft(X, axis=0) / 64) == 32
    rng = np.random.default_rng(0)
    Xn = X + 1e-3 * rng.standard_normal(X.shape)
    k = auto_harmonics(np.fft.rfft(Xn, axis=0) / 64)
    assert 3 <= k <= 10

def test_surface_map_recovers_vertex_field():
    X0, tri = sphere()
    rng = np.random.default_rng(1)
    t = tri[rng.integers(len(tri), size=50)]; w = rng.dirichlet([1, 1, 1], size=50)
    pts = (w[:, :, None] * X0[t]).sum(1)
    smap = SurfaceMap(X0, tri, pts)
    assert smap.distance.max() < 1e-12
    F = X0 * 2.0                                                   # linear field
    assert np.allclose(smap(F), 2 * pts, atol=1e-12)

def test_obj_roundtrip_and_cli_check(tmp_path, capsys):
    X, tri, _, _ = frames(12)
    for k, Xk in enumerate(X):
        with open(tmp_path / f'f_{k}.obj', 'w') as f:
            f.writelines('v %.12g %.12g %.12g\n' % tuple(v) for v in Xk)
            f.writelines('f %d %d %d\n' % tuple(t + 1) for t in tri)
    Xr, trir, files = read_frames(str(tmp_path / 'f_*.obj'))
    assert files[2].endswith('f_2.obj') and files[10].endswith('f_10.obj')   # natural sort
    assert np.allclose(Xr, X) and np.array_equal(trir, tri)
    assert main(['check', str(tmp_path / 'f_*.obj'), '--period', '1.0']) == 0
    assert 'linear_vs_spline_max_diff_pct_of_peak' in capsys.readouterr().out

def _series(A0, A, B, T, t):
    w = 2 * np.pi / T
    return A0 + sum(np.cos((k + 1) * w * t) * A[k] + np.sin((k + 1) * w * t) * B[k] for k in range(len(A)))

def test_fourier_coefficients_match_fit():
    from frames2foam.cli import fourier_coefficients
    X, tri, X0, tk = frames(16)
    ident = SurfaceMap(X0, tri, X0)                     # patch points = frame vertices
    m = PeriodicMotion(X, T, 'fourier')
    A0, A, B, K, err = fourier_coefficients(m, ident, X0, m.n, nharm=m.nharm)
    for t in (0.0, 0.137, 0.5, 0.91):
        assert np.allclose(_series(A0, A, B, T, t), m.position(t) - X0, atol=1e-12)
    ms = PeriodicMotion(X[:, :, :], T, 'spline')
    A0, A, B, K, err = fourier_coefficients(ms, ident, X0, 512, vel_tol=1e-4)
    assert err <= 1e-4 and K < 256
    t = np.linspace(0, T, 37)
    d = np.array([_series(A0, A, B, T, ti) for ti in t]) - (ms.position(t) - X0)
    assert np.abs(d).max() < 1e-4 * np.abs(X - X0).max()

def test_patch_point_order(tmp_path):
    from frames2foam.foam import read_patch_points
    d = tmp_path / 'constant' / 'polyMesh'; d.mkdir(parents=True)
    hdr = 'FoamFile { version 2.0; format ascii; class x; object x; }\n'
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]], float)
    (d / 'points').write_text(hdr + '6\n(\n' + '\n'.join('(%g %g %g)' % tuple(p) for p in pts) + '\n)\n')
    (d / 'faces').write_text(hdr + '2\n(\n4(4 5 2 1)\n4(0 1 2 3)\n)\n')
    (d / 'boundary').write_text(hdr + '1\n(\n    wall\n    {\n        type wall;\n        nFaces 2;\n        startFace 0;\n    }\n)\n')
    assert np.array_equal(read_patch_points(str(tmp_path), 'wall', order='patch'), pts[[4, 5, 2, 1, 0, 3]])
    assert np.array_equal(read_patch_points(str(tmp_path), 'wall'), pts[[0, 1, 2, 3, 4, 5]])
