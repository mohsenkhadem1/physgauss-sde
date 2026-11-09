"""Labelled GIC adapter: known external body force on the MPM grid.

GIC (b523851) supports gravity, initial velocity and colliders but no applied
force. This adapter adds one Taichi kernel that scatters the impulse
dt * a(t) * f_p of a per-particle dead load f_p into grid momentum with the
same quadratic B-spline weights as native P2G, immediately after native P2G,
and its autodiff gradient in the backward pass. Native kernels are unchanged.

Specification (env STEM_FORCE_JSON -> file): {"lo": x0, "hi": x1,
"force": [Fx,Fy,Fz], "t_ramp": T}. The total force is split over particles
whose position at simulation step 0 has lo <= x < hi, in proportion to
particle mass (uniform body-force density). a(t) = 0.5(1-cos(pi t/T)) for
t<T, then 1. Without STEM_FORCE_JSON the adapter is inert (exact native path).
"""
import json, os
import numpy as np
import taichi as ti
from simulator.mpm_simulator import MPMSimulator

_SPEC = None
if os.environ.get('STEM_FORCE_JSON'):
    _SPEC = json.loads(open(os.environ['STEM_FORCE_JSON']).read())
AUDIT = {}


@ti.kernel
def _scatter(sim: ti.template(), s: ti.i32):
    for p in range(sim.n_particles[None]):
        base = ti.floor(sim.x[p, s] * sim.inv_dx[None] - 0.5).cast(int)
        fx = sim.x[p, s] * sim.inv_dx[None] - base.cast(sim.dtype)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        imp = sim.dt[None] * sim.ext_amp[None] * sim.ext_f[p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    weight = w[i][0] * w[j][1] * w[k][2]
                    sim.grid_v_in[base + ti.Vector([i, j, k])] += weight * imp


@ti.kernel
def _read_x0(sim: ti.template(), out: ti.types.ndarray(), m: ti.types.ndarray()):
    for p in range(sim.n_particles[None]):
        for d in ti.static(range(3)):
            out[p, d] = ti.cast(sim.x[p, 0][d], ti.f64)
        m[p] = ti.cast(sim.p_mass[p], ti.f64)


_orig_init = MPMSimulator.__init__


def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    n = max(int(self.n_particles[None]), 1)
    self.ext_f = ti.Vector.field(3, dtype=self.dtype, shape=n)
    self.ext_amp = ti.field(dtype=self.dtype, shape=())
    self._ext_on = _SPEC is not None


def _prepare(self):
    n = int(self.n_particles[None])
    x = np.zeros((n, 3)); m = np.zeros(n)
    _read_x0(self, x, m)
    sel = (x[:, 0] >= _SPEC['lo']) & (x[:, 0] < _SPEC['hi'])
    assert sel.sum() > 0, 'no particles in load region'
    f = np.zeros((self.ext_f.shape[0], 3))
    f[:n][sel] = np.asarray(_SPEC['force'])[None] * (m[sel] / m[sel].sum())[:, None]
    self.ext_f.from_numpy(f.astype(np.float32 if self.dtype == ti.f32 else np.float64))
    AUDIT.update(particles_loaded=int(sel.sum()), particles_total=n, total_force=f.sum(0).tolist(),
                 loaded_mass=float(m[sel].sum()), region=[_SPEC['lo'], _SPEC['hi']])
    if os.environ.get('STEM_FORCE_AUDIT'):
        open(os.environ['STEM_FORCE_AUDIT'], 'w').write(json.dumps(AUDIT))


def _set_amp(self, s):
    t = s * float(self.dt[None])
    q = min(max(t / _SPEC['t_ramp'], 0.0), 1.0)
    self.ext_amp[None] = 0.5 * (1 - np.cos(np.pi * q))


def substep(self, s, cache=True):
    local_index = s % self.cuda_chunk_size
    if self._ext_on:
        if s == 0:
            _prepare(self)
        _set_amp(self, s)
    self.grid.deactivate_all()
    self.compute_F_tmp(local_index)
    self.svd()
    self.project_F(local_index)
    self.p2g(local_index)
    if self._ext_on:
        _scatter(self, local_index)
    self.grid_op(local_index)
    self.g2p(local_index)
    self.check_cfl(local_index + 1)
    if (local_index == self.cuda_chunk_size - 1) and cache:
        self.push_to_memory()


def substep_grad(self, s):
    local_index = s % self.cuda_chunk_size
    if local_index == self.cuda_chunk_size - 1:
        self.pop_from_memory()
    if self._ext_on:
        _set_amp(self, s)
    self.grid.deactivate_all()
    self.compute_F_tmp(local_index)
    self.svd()
    self.project_F(local_index)
    self.p2g(local_index)
    if self._ext_on:
        _scatter(self, local_index)
    self.grid_op(local_index)
    self.clear_svd_grads()
    self.g2p.grad(local_index)
    self.grid_op.grad(local_index)
    if self._ext_on:
        _scatter.grad(self, local_index)
    self.p2g.grad(local_index)
    self.project_F.grad(local_index)
    self.svd_grad()
    self.compute_F_tmp.grad(local_index)


def install():
    if _SPEC is None:
        return None
    MPMSimulator.__init__ = _init
    MPMSimulator.substep = substep
    MPMSimulator.substep_grad = substep_grad
    return _SPEC
