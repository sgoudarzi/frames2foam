"""Map the straight blockMesh pipe onto the fusiform aneurysm geometry
by radially scaling every point with r0(z)/R0."""
import re, sys, numpy as np
sys.path.insert(0, 'scripts'); from geometry import R0, r0
fn = 'constant/polyMesh/points'
txt = open(fn).read()
m = re.search(r'\n(\d+)\n\(\n', txt)
n = int(m.group(1)); start = m.end()
body, tail = txt[start:].split('\n)\n', 1)
pts = np.array([list(map(float, l.strip('()').split())) for l in body.split('\n')])
assert len(pts) == n
k = r0(pts[:, 2]) / R0
pts[:, 0] *= k; pts[:, 1] *= k
open(fn, 'w').write(txt[:start] + '\n'.join('(%.10g %.10g %.10g)' % tuple(p) for p in pts) + '\n)\n' + tail)
print('reshaped %d points, max radius %.2f mm' % (n, 1e3*np.hypot(pts[:,0], pts[:,1]).max()))
