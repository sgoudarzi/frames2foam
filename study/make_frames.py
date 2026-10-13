"""Write the 30 OBJ frames of the aneurysm wall used for the CFD runs (frames30/)."""
import os, sys
import numpy as np
sys.path.insert(0, '..')
from kinematics import Aneurysm, T

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
M = Aneurysm(nth=96, nz=301)
os.makedirs(f'frames{N}', exist_ok=True)
for k in range(N):
    X = M.positions(k * T / N)[0]
    with open(f'frames{N}/aneurysm_{k:03d}.obj', 'w') as f:
        f.write(f'# aneurysm wall frame {k} of {N}, t = {k*T/N:.5f} s, units m\n')
        f.writelines('v %.9g %.9g %.9g\n' % tuple(v) for v in X)
        f.writelines('f %d %d %d\n' % tuple(t + 1) for t in M.tri)
print(f'wrote {N} frames')
