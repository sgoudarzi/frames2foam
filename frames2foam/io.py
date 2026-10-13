"""Readers for OBJ and STL surface frames."""
import glob, re, struct
import numpy as np

def _read_obj(path):
    v, f = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                v.append(line.split()[1:4])
            elif line.startswith('f '):
                idx = [int(tok.split('/')[0]) for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):          # fan-triangulate polygons
                    f.append((idx[0], idx[k], idx[k + 1]))
    v = np.asarray(v, float); f = np.asarray(f, int)
    f = np.where(f > 0, f - 1, f + len(v))                # 1-based and negative indices
    return v, f

def _read_stl(path):
    with open(path, 'rb') as fh:
        data = fh.read()
    if len(data) >= 84:
        n = struct.unpack('<I', data[80:84])[0]
        if 84 + 50 * n == len(data):                      # binary
            rec = np.frombuffer(data[84:], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
            tri = rec['v'].astype(float)
        else:
            tri = None
    if len(data) < 84 or tri is None:                     # ascii
        nums = re.findall(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)', data)
        tri = np.asarray(nums, float).reshape(-1, 3, 3)
    # STL stores triangle soups; weld identical vertices in first-seen order so that
    # frames written with the same triangle ordering share the same vertex ordering
    flat = tri.reshape(-1, 3)
    key = np.round(flat, 9)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first); rank = np.empty_like(order); rank[order] = np.arange(len(order))
    v = flat[np.sort(first)]
    return v, rank[inv.ravel()].reshape(-1, 3)

def read_surface(path):
    """Return (vertices [nv,3], triangles [nt,3]) of an OBJ or STL file."""
    if str(path).lower().endswith('.obj'):
        return _read_obj(path)
    if str(path).lower().endswith('.stl'):
        return _read_stl(path)
    raise ValueError(f'unsupported surface format: {path}')

def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def read_frames(pattern, scale=1.0):
    """Read a frame sequence matching a glob pattern, sorted naturally.
    Returns (X [nframes, nv, 3], triangles of the first frame, file list).
    All frames must share vertex count and ordering (vertex correspondence)."""
    files = sorted(glob.glob(pattern), key=_natural_key)
    if not files:
        raise FileNotFoundError(f'no frames match {pattern}')
    X, tri0 = [], None
    for fn in files:
        v, t = read_surface(fn)
        if tri0 is None:
            tri0 = t
        elif len(v) != len(X[0]) or t.shape != tri0.shape or not np.array_equal(t, tri0):
            raise ValueError(f'{fn}: vertex count or connectivity differs from {files[0]}; '
                             'frames need consistent vertex correspondence')
        X.append(v * scale)
    return np.asarray(X), tri0, files
