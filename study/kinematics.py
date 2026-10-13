"""Reference kinematics with a known smooth volume rate, used to measure the
error that temporal sampling and interpolation of frames introduce.

aneurysm : fusiform sac distending with a carotid-like waveform (as in the
           companion OpenFOAM case), closed tube surface with fixed end caps
lv       : idealised left ventricle, a truncated prolate ellipsoid whose volume
           follows a Wiggers-type curve (EDV 120 mL, ESV 50 mL, E and A filling waves)
Both return a closed, outward-oriented triangulated surface whose vertex
positions are an analytic function of time."""
import numpy as np

T = 0.8

# ---------------- aneurysm ----------------
R0, RMAX, L, ZC, H, EPS, LAG = 2.0e-3, 5.0e-3, 60.0e-3, 26.0e-3, 9.0e-3, 0.08, 0.03
def _s(z):
    x = (z - ZC) / H
    return np.where(np.abs(x) < 1, np.cos(0.5 * np.pi * x) ** 2, 0.0)
def _umean(t):
    # sum over periodic images so the waveform is smooth across the cycle boundary
    tau = np.subtract.outer(np.asarray(t, float), np.arange(-2, 3) * T)
    return 0.20 + (0.45 * np.exp(-((tau - 0.15) / 0.055) ** 2) + 0.08 * np.exp(-((tau - 0.38) / 0.05) ** 2)).sum(-1)
def _g(t):
    return (_umean(np.asarray(t) - LAG) - 0.20) / 0.45

def tube_surface(nth=48, nz=161):
    """Closed tube: side wall grid plus fan caps. Returns (param arrays, triangles)."""
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    z = np.linspace(0, L, nz)
    TH, Z = np.meshgrid(th, z)                      # [nz, nth]
    idx = np.arange(nz * nth).reshape(nz, nth)
    tri = []
    for i in range(nz - 1):
        for j in range(nth):
            a, b = idx[i, j], idx[i, (j + 1) % nth]
            c, d = idx[i + 1, j], idx[i + 1, (j + 1) % nth]
            tri += [(a, b, d), (a, d, c)]
    c0, c1 = nz * nth, nz * nth + 1                 # cap centres
    for j in range(nth):
        tri.append((c0, idx[0, (j + 1) % nth], idx[0, j]))
        tri.append((c1, idx[-1, j], idx[-1, (j + 1) % nth]))
    return (TH.ravel(), Z.ravel()), np.asarray(tri)

class Aneurysm:
    name, label = 'aneurysm', 'Aneurysm sac wall'
    def __init__(self, nth=48, nz=161):
        (self.th, self.z), self.tri = tube_surface(nth, nz)
        self.base = R0 * (1 + (RMAX / R0 - 1) * _s(self.z))
        self.amp = EPS * _s(self.z) * self.base
    def positions(self, t):
        t = np.atleast_1d(t)
        r = self.base + np.multiply.outer(_g(t), self.amp)          # [nt, nv]
        X = np.stack([r * np.cos(self.th), r * np.sin(self.th), np.broadcast_to(self.z, r.shape)], -1)
        caps = np.zeros((len(t), 2, 3)); caps[:, 1, 2] = L
        return np.concatenate([X, caps], 1)
    # separable form X(t) = base + f(t) * shape, used to reconstruct cheaply
    def factor(self, t):
        return _g(np.atleast_1d(t))
    def from_factor(self, f, df):
        nv = len(self.base); f = np.atleast_1d(f); df = np.atleast_1d(df)
        r = self.base + np.multiply.outer(f, self.amp); dr = np.multiply.outer(df, self.amp)
        c, s = np.cos(self.th), np.sin(self.th)
        X = np.stack([r * c, r * s, np.broadcast_to(self.z, r.shape)], -1)
        U = np.stack([dr * c, dr * s, np.zeros_like(dr)], -1)
        caps = np.zeros((len(f), 2, 3)); caps[:, 1, 2] = L
        return np.concatenate([X, caps], 1), np.concatenate([U, 0 * caps], 1)
    def reference(self):
        """Undeformed geometry (mesh is built from this)."""
        r = self.base
        X = np.stack([r * np.cos(self.th), r * np.sin(self.th), self.z], -1)
        return np.concatenate([X, [[0, 0, 0], [0, 0, L]]])

# ---------------- idealised LV ----------------
_K = np.arange(-3, 4)                                            # periodic images
def _bump(t, c, w):
    tau = np.subtract.outer(np.asarray(t, float), c + _K * T)
    return (np.exp(-0.5 * (tau / w) ** 2) / (w * np.sqrt(2 * np.pi))).sum(-1)
def _bump_int(t, c, w):
    from scipy.special import erf
    tau = np.subtract.outer(np.asarray(t, float), c + _K * T)
    return (0.5 * (1 + erf(tau / (w * np.sqrt(2))))).sum(-1)
EDV, SV, SV_E = 120e-6, 70e-6, 50e-6                             # m^3
WAVES = [(-SV, 0.13, 0.055), (SV_E, 0.46, 0.035), (SV - SV_E, 0.73, 0.030)]   # ejection, E, A
def lv_volume(t):
    t = np.asarray(t, float)
    # volume relative to end-diastole at t = 0 (phase 0 = end-diastole)
    v = EDV + sum(a * (_bump_int(t, c, w) - _bump_int(0.0, c, w)) for a, c, w in WAVES)
    return v
def lv_rate(t):
    return sum(a * _bump(t, c, w) for a, c, w in WAVES)

def ellipsoid_surface(nu=40, nv=64, trunc=0.45):
    """Prolate half-ellipsoid truncated at the base (valve plane) and closed with a cap."""
    u = np.linspace(0, np.pi / 2 + np.arcsin(trunc), nu)[1:]     # from apex (u=0) to base
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing='ij')
    idx = np.arange((nu - 1) * nv).reshape(nu - 1, nv)
    tri = []
    apex, base_c = (nu - 1) * nv, (nu - 1) * nv + 1
    for j in range(nv):
        tri.append((apex, idx[0, (j + 1) % nv], idx[0, j]))
    for i in range(nu - 2):
        for j in range(nv):
            a, b = idx[i, j], idx[i, (j + 1) % nv]
            c, d = idx[i + 1, j], idx[i + 1, (j + 1) % nv]
            tri += [(a, b, d), (a, d, c)]
    for j in range(nv):
        tri.append((base_c, idx[-1, j], idx[-1, (j + 1) % nv]))
    return U.ravel(), Vv.ravel(), np.asarray(tri)

class LeftVentricle:
    name, label = 'lv', 'Idealised left ventricle'
    A_LONG, A_SHORT = 1.0, 0.45                                   # shape ratios
    def __init__(self, nu=40, nv=64):
        self.u, self.v, self.tri = ellipsoid_surface(nu, nv)
        X = self._unit()
        from frames2foam import signed_volume
        self.V1 = signed_volume(X, self.tri)                      # volume at unit scale
    def _unit(self):
        # long axis along z, apex at z = -A_LONG; base plane at z = A_LONG*cos(u_max) ... closed by cap
        x = self.A_SHORT * np.sin(self.u) * np.cos(self.v)
        y = self.A_SHORT * np.sin(self.u) * np.sin(self.v)
        z = -self.A_LONG * np.cos(self.u)
        X = np.c_[x, y, z]
        zb = z.max()
        return np.vstack([X, [[0, 0, -self.A_LONG], [0, 0, zb]]])
    def positions(self, t):
        t = np.atleast_1d(t)
        k = (lv_volume(t) / self.V1) ** (1 / 3)                   # uniform scaling keeps shape
        return k[:, None, None] * self._unit()[None]
    def factor(self, t):
        return (lv_volume(np.atleast_1d(t)) / self.V1) ** (1 / 3)
    def from_factor(self, f, df):
        X0 = self._unit()
        return np.multiply.outer(np.atleast_1d(f), X0), np.multiply.outer(np.atleast_1d(df), X0)
    def reference(self):
        return self.positions(0.0)[0]

def exact_rate(model, t, h=1e-6):
    """Volume rate of the triangulated surface itself under the exact smooth motion
    (central difference of vertex positions), so that spatial discretisation is
    identical between reference and reconstruction."""
    from frames2foam import surface_flux
    t = np.atleast_1d(t)
    X = model.positions(t); U = (model.positions(t + h) - model.positions(t - h)) / (2 * h)
    return surface_flux(X, U, model.tri)
