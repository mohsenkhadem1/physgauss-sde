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


_ORIG = {}


def _init(self, *a, **k):
    _ORIG[type(self).__name__+str(id(type(self)))](self, *a, **k)
    n = max(int(self.n_particles[None]), 1)
    self.ext_f = ti.Vector.field(3, dtype=self.dtype, shape=n)
    self.ext_amp = ti.field(dtype=self.dtype, shape=())
    self._ext_on = _SPEC is not None


def _prepare(self):
    n = int(self.n_particles[None])
    x = np.zeros((n, 3)); m = np.zeros(n)
    _read_x0(self, x, m)
    if 'window' in _SPEC:  # block press: top `top` metres of the body inside an x-z window, measured from
        (x0, z0), (x1, z1) = _SPEC['window']  # the reconstruction's own top surface in that window
        inwin = (x[:, 0] >= x0) & (x[:, 0] < x1) & (x[:, 2] >= z0) & (x[:, 2] < z1)
        sel = inwin & (x[:, 1] >= x[inwin, 1].max()-_SPEC['top'])
    elif 'box' in _SPEC:  # 3D reference box [[x0,y0,z0],[x1,y1,z1]]
        lo3, hi3 = np.asarray(_SPEC['box'][0]), np.asarray(_SPEC['box'][1])
        sel = np.all((x >= lo3) & (x < hi3), 1)
    else:
        sel = (x[:, 0] >= _SPEC['lo']) & (x[:, 0] < _SPEC['hi'])
    assert sel.sum() > 0, 'no particles in load region'
    f = np.zeros((self.ext_f.shape[0], 3))
    f[:n][sel] = np.asarray(_SPEC['force'])[None] * (m[sel] / m[sel].sum())[:, None]
    self.ext_f.from_numpy(f.astype(np.float32 if self.dtype == ti.f32 else np.float64))
    AUDIT.update(particles_loaded=int(sel.sum()), particles_total=n, total_force=f.sum(0).tolist(),
                 loaded_mass=float(m[sel].sum()), region=_SPEC.get('box', [_SPEC.get('lo'), _SPEC.get('hi')]))
    if os.environ.get('STEM_FORCE_AUDIT'):
        open(os.environ['STEM_FORCE_AUDIT'], 'w').write(json.dumps(AUDIT))


def _set_amp(self, s):
    t = s * float(self.dt[None])
    q = min(max(t / _SPEC['t_ramp'], 0.0), 1.0)
    self.ext_amp[None] = 0.5 * (1 - np.cos(np.pi * q))


def substep(self, s, cache=True):
    local_index = s % self.cuda_chunk_size
    if self._ext_on:
        if s == 0 and cache and not getattr(self, '_ext_ready', False):
            _prepare(self); self._ext_ready = True  # loaded particle set fixed at the first forward pass
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


def install(cls=None):
    """Patch an MPM simulator class (GIC's by default; PAC-NeRF's lib.engine.MPMSimulator has the
    same substep structure). Inert when no force spec is given."""
    if _SPEC is None:
        return None
    if cls is None:
        from simulator.mpm_simulator import MPMSimulator as cls
    key = cls.__name__+str(id(cls))
    if key not in _ORIG:
        _ORIG[key] = cls.__init__
        cls.__init__ = _init
        cls.substep = substep
        cls.substep_grad = substep_grad
    return _SPEC
