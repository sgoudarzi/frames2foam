"""OpenFOAM I/O: patch points of an ASCII polyMesh and timeVaryingMapped boundary data."""
import os, re
import numpy as np

def _list(path):
    txt = open(path).read()
    if 'format      binary' in txt or 'format binary' in txt:
        raise ValueError(f'{path} is binary; convert with foamFormatConvert (writeFormat ascii) first')
    m = re.search(r'\n(\d+)\s*\n\(\n', txt)
    n = int(m.group(1))
    body = txt[m.end():].rsplit('\n)', 1)[0].strip().split('\n')
    assert len(body) == n, path
    return body

def read_patch_points(case, patch, order='sorted'):
    """Unique points of a boundary patch, read from constant/polyMesh (ASCII).
    order='sorted'  : by global point label (timeVaryingMapped boundaryData)
    order='patch'   : OpenFOAM's local patch order (PrimitivePatch::meshPoints, first
                      appearance while walking the patch faces), so the fourierDisplacement
                      boundary condition can use the file without a position search"""
    d = os.path.join(case, 'constant', 'polyMesh')
    pts = np.array([l.strip('()').split() for l in _list(os.path.join(d, 'points'))], float)
    faces = _list(os.path.join(d, 'faces'))
    btxt = open(os.path.join(d, 'boundary')).read()
    m = re.search(r'\b' + re.escape(patch) + r'\s*\{([^}]*)\}', btxt)
    if not m:
        raise KeyError(f'patch {patch} not found')
    nF = int(re.search(r'nFaces\s+(\d+)', m.group(1)).group(1))
    sF = int(re.search(r'startFace\s+(\d+)', m.group(1)).group(1))
    if order == 'patch':
        seen, ids = set(), []
        for l in faces[sF:sF + nF]:
            for i in map(int, l[l.index('(') + 1:-1].split()):
                if i not in seen:
                    seen.add(i); ids.append(i)
        return pts[np.array(ids)]
    ids = set()
    for l in faces[sF:sF + nF]:
        ids.update(map(int, l[l.index('(') + 1:-1].split()))
    ids = np.array(sorted(ids))
    return pts[ids]

def _write_list(path, arr):
    with open(path, 'w') as f:
        f.write('%d\n(\n' % len(arr))
        f.writelines('(%.10g %.10g %.10g)\n' % tuple(v) for v in arr)
        f.write(')\n')

def write_boundary_data(case, patch, points, times, values, field='pointDisplacement'):
    """Write constant/boundaryData/<patch>/points and <time>/<field> for each time."""
    root = os.path.join(case, 'constant', 'boundaryData', patch)
    os.makedirs(root, exist_ok=True)
    _write_list(os.path.join(root, 'points'), points)
    for t, v in zip(times, values):
        d = os.path.join(root, '%.10g' % t)
        os.makedirs(d, exist_ok=True)
        _write_list(os.path.join(d, field), v)
    return root


def _vec_list(f, name, arr):
    f.write(f'{name}\n{len(arr)}\n(\n')
    f.writelines('(%.10g %.10g %.10g)\n' % tuple(v) for v in arr)
    f.write(');\n\n')

def write_fourier_coeffs(case, patch, points, period, t0, A0, A, B, note=''):
    """Write constant/fourierMotion/<patch> for the fourierDisplacement boundary condition.
    A0 [np,3]; A, B [K,np,3] cosine and sine coefficients of the displacement."""
    root = os.path.join(case, 'constant', 'fourierMotion')
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, patch)
    with open(path, 'w') as f:
        f.write('FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n'
                f'    object {patch};\n}}\n')
        f.write(f'// Written by frames2foam. {note}\n')
        f.write('// d(t) = A0 + sum_k A_k cos(k w (t - t0)) + B_k sin(k w (t - t0)),  w = 2 pi / period\n\n')
        f.write(f'period {period!r};\nt0 {t0!r};\nnHarmonics {len(A)};\n\n')
        _vec_list(f, 'points', points)
        _vec_list(f, 'A0', A0)
        for name, C in (('A', A), ('B', B)):
            f.write(f'{name}\n{len(C)}\n(\n')
            for Ck in C:
                f.write(f'{len(Ck)}\n(\n')
                f.writelines('(%.10g %.10g %.10g)\n' % tuple(v) for v in Ck)
                f.write(')\n')
            f.write(');\n\n')
    return path
