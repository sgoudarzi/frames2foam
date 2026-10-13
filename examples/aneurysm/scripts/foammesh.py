"""Minimal reader for ASCII OpenFOAM polyMesh (points, faces, boundary)."""
import re, numpy as np
def _list_body(txt):
    m = re.search(r'\n(\d+)\n\(\n', txt)
    return int(m.group(1)), txt[m.end():].rsplit('\n)', 1)[0]
def read_points(d='constant/polyMesh'):
    n, b = _list_body(open(f'{d}/points').read())
    return np.array([l.strip('()').split() for l in b.split('\n')], float)
def read_faces(d='constant/polyMesh'):
    n, b = _list_body(open(f'{d}/faces').read())
    return [list(map(int, l[l.index('(')+1:-1].split())) for l in b.split('\n')]
def read_boundary(d='constant/polyMesh'):
    txt = open(f'{d}/boundary').read()
    out = {}
    for name, body in re.findall(r'(\w+)\s*\{([^}]*)\}', txt):
        if 'startFace' in body:
            out[name] = (int(re.search(r'nFaces\s+(\d+)', body).group(1)),
                         int(re.search(r'startFace\s+(\d+)', body).group(1)))
    return out
def patch_data(name, d='constant/polyMesh'):
    pts, faces, bnd = read_points(d), read_faces(d), read_boundary(d)
    n, s = bnd[name]
    pf = faces[s:s+n]
    fc = np.array([pts[f].mean(axis=0) for f in pf])
    pid = np.unique(np.concatenate([np.array(f) for f in pf]))
    return fc, pts[pid]
