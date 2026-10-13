"""Surface geometry: flux of a moving surface, enclosed volume, point mapping."""
import numpy as np
from scipy.spatial import cKDTree

def signed_volume(V, tri):
    """Enclosed volume of a closed, consistently oriented triangle surface."""
    a, b, c = V[..., tri[:, 0], :], V[..., tri[:, 1], :], V[..., tri[:, 2], :]
    return np.einsum('...i,...i->...', a, np.cross(b, c)).sum(-1) / 6.0

def surface_flux(V, U, tri):
    """Volume swept per unit time by a moving triangulated surface, sum over triangles of
    (mean vertex velocity . area vector). Equals dV/dt of the enclosed region for a closed
    outward-oriented surface; for an open surface it is the wall contribution to dV/dt.
    V, U : [..., nv, 3] positions and velocities."""
    a, b, c = V[..., tri[:, 0], :], V[..., tri[:, 1], :], V[..., tri[:, 2], :]
    S = 0.5 * np.cross(b - a, c - a)
    Um = (U[..., tri[:, 0], :] + U[..., tri[:, 1], :] + U[..., tri[:, 2], :]) / 3.0
    return np.einsum('...ti,...ti->...', Um, S)

def _closest_on_triangles(p, a, b, c):
    """Closest points on triangles (a,b,c) to points p, all [m,3]; returns barycentrics [m,3]."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b; d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c; d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    m = len(p); w = np.zeros((m, 3)); done = np.zeros(m, bool)
    def put(mask, bary):
        nonlocal done
        mask = mask & ~done; w[mask] = bary[mask]; done |= mask
    one = np.ones(m); zero = np.zeros(m)
    put((d1 <= 0) & (d2 <= 0), np.c_[one, zero, zero])
    put((d3 >= 0) & (d4 <= d3), np.c_[zero, one, zero])
    put((d6 >= 0) & (d5 <= d6), np.c_[zero, zero, one])
    with np.errstate(divide='ignore', invalid='ignore'):
        v = d1 / (d1 - d3); put((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.c_[1 - v, v, zero])
        v = d2 / (d2 - d6); put((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.c_[1 - v, zero, v])
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), np.c_[zero, 1 - v, v])
        den = 1.0 / (va + vb + vc); v = vb * den; u = vc * den
        put(np.ones(m, bool), np.c_[1 - v - u, v, u])
    return w

class SurfaceMap:
    """Attach target points (e.g. OpenFOAM patch points) to a reference surface by
    closest-point projection, so that any later surface state can be evaluated at them
    through fixed barycentric weights."""
    def __init__(self, V0, tri, points, k=12):
        cen = V0[tri].mean(1)
        tree = cKDTree(cen)
        _, cand = tree.query(points, k=min(k, len(tri)))
        cand = np.atleast_2d(cand) if cand.ndim == 1 else cand
        best_d = np.full(len(points), np.inf)
        self.tri_id = np.zeros(len(points), int); self.w = np.zeros((len(points), 3))
        for j in range(cand.shape[1]):
            t = tri[cand[:, j]]
            w = _closest_on_triangles(points, V0[t[:, 0]], V0[t[:, 1]], V0[t[:, 2]])
            q = (w[:, :, None] * V0[t]).sum(1)
            d = np.linalg.norm(q - points, axis=1)
            better = d < best_d
            best_d[better] = d[better]; self.tri_id[better] = cand[better, j]; self.w[better] = w[better]
        self.vid = tri[self.tri_id]
        self.distance = best_d

    def __call__(self, F):
        """Evaluate a vertex field F [..., nv, d] at the mapped points."""
        return np.einsum('pk,...pkd->...pd', self.w, F[..., self.vid, :])
