"""Pulsatile parabolic inflow for the inlet patch, written for timeVaryingMappedFixedValue
(constant/boundaryData/inlet). The wall motion comes from frames2foam."""
import os, sys
import numpy as np
sys.path.insert(0, 'scripts')
from geometry import R0, umean
from foammesh import patch_data

DT, TEND = 0.005, 1.65
times = np.round(np.arange(0.0, TEND + 1e-9, DT), 6)

def wlist(path, arr):
    with open(path, 'w') as f:
        f.write('%d\n(\n' % len(arr))
        f.writelines('(%.9g %.9g %.9g)\n' % tuple(v) for v in arr)
        f.write(')\n')

fc, _ = patch_data('inlet')
root = 'constant/boundaryData/inlet'
os.makedirs(root, exist_ok=True)
wlist(f'{root}/points', fc)
rr = (fc[:, 0] ** 2 + fc[:, 1] ** 2) / R0 ** 2
for t in times:
    U = np.zeros_like(fc); U[:, 2] = 2 * umean(t) * np.clip(1 - rr, 0, None)
    os.makedirs(f'{root}/{t:g}', exist_ok=True)
    wlist(f'{root}/{t:g}/U', U)
print(f'{len(times)} inlet samples, {len(fc)} faces')
