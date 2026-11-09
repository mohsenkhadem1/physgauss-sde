"""Qualification checks for fem.py (run before any trial data is generated)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem

out = {}
# 1. Tangent equals finite-difference derivative of internal force.
m = fem.Mesh((4, 2, 2)); model = fem.Elastic(m, 1e6, .3, 1000.)
rng = np.random.default_rng(0)
u = np.zeros(3*m.n); u[m.free] = 2e-3*rng.standard_normal(len(m.free))
f, K = model.internal(u); d = np.zeros_like(u); d[m.free] = rng.standard_normal(len(m.free))
eps = 1e-7
fd = (model.internal(u+eps*d, False)[0]-model.internal(u-eps*d, False)[0])/(2*eps)
out['tangent_fd_rel_err'] = float(np.linalg.norm(K@d-fd)/np.linalg.norm(fd))
assert out['tangent_fd_rel_err'] < 1e-6, out
# 2. Mass: total = rho*V, rigid translation; unit-load integrals.
V = (fem.X1-fem.X0)*(fem.Y1-fem.Y0)*(fem.Z1-fem.Z0)
one = np.zeros(3*m.n); one[0::3] = 1
out['mass_rel_err'] = float(abs(one@(m.M1@one)-V)/V)
assert out['mass_rel_err'] < 1e-12
w, vol = m.region_weights(0.085, 0.115)
out['region_volume_rel_err'] = float(abs(vol-.03*.08*.08)/(.03*.08*.08))
out['region_centroid_x'] = float(w@m.X[:, 0])
# 3. Static linear cantilever with end-region load vs Timoshenko beam (point load at centroid).
E, nu = 4e6, .3; a = .08; I = a**4/12; A = a*a; G = E/(2*(1+nu)); ks = 10*(1+nu)/(12+11*nu)
res = {}
for cells in [(22, 4, 4), (33, 6, 6)]:
    mm = fem.Mesh(cells); mod = fem.Elastic(mm, E, nu, 1.)
    K = mod.linear_K().tocsr(); fr = mm.free
    P = 1.0; xs = .1; f = mm.body_force_region(xs-.015, xs+.015, [0, 0, P])
    uu = np.zeros(3*mm.n); uu[fr] = fem._factor(K[fr][:, fr])(f[fr])
    wl, _ = mm.region_weights(xs-.015, xs+.015)
    res[str(cells)] = float(wl@uu.reshape(-1, 3)[:, 2])
aa = xs-fem.CLAMP_X
beam = P*aa**3/(3*E*I)+P*aa/(ks*G*A)
out['static_load_point_disp_m'] = res; out['timoshenko_point_load_m'] = beam
out['static_note'] = 'FEM uses 30 mm distributed load and a 3D clamp face; Timoshenko is a point-load reference, so agreement is approximate (clamp warping, distribution).'
# 4. Linear vs nonlinear dynamic agreement at small load.
mm = fem.Mesh((22, 4, 4))
lin = fem.Elastic(mm, E, nu, 1000.); force = lambda t: 1e-3*(mm.gravity(1000.))
t0 = time.time()
U1, _ = fem.newmark(lin, force, .1, 1/480, 10, nonlinear=False)
U2, st = fem.newmark(lin, force, .1, 1/480, 10, nonlinear=True)
out['small_load_linear_vs_nonlinear_rel'] = float(np.abs(U1-U2).max()/np.abs(U1).max())
out['dynamic_seconds'] = time.time()-t0
assert out['small_load_linear_vs_nonlinear_rel'] < 5e-3, out
print(json.dumps(out, indent=1))
Path('reports/stem-trial-v01').mkdir(parents=True, exist_ok=True)
Path('reports/stem-trial-v01/fem-checks.json').write_text(json.dumps(out, indent=1))
