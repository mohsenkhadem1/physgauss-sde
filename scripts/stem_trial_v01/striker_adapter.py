"""Labelled MPM adapter (forecast only): rigid spherical striker with the trial's volumetric penalty contact.
Each substep, particles inside the sphere receive f_p = kappa*(r-|x-c|)*n * V_p (V_p = p_mass/rho_config),
scattered into grid momentum with native P2G weights (same kernel as the force adapter); the sphere gets
the opposite total force and is integrated symplectically (v += dt F/m, c += dt v), no gravity on the sphere.
Spec (env STEM_STRIKER_JSON): {"r","m","kappa","c0","v0","rho_config"}. No gradient path (raises)."""
import json, os
import numpy as np
import taichi as ti
import gic_force_adapter as FA

SPEC = json.loads(open(os.environ['STEM_STRIKER_JSON']).read()) if os.environ.get('STEM_STRIKER_JSON') else None
STATE = dict(c=None, v=None, history=[])


@ti.kernel
def _contact(sim: ti.template(), s: ti.i32, cx: ti.f32, cy: ti.f32, cz: ti.f32, r: ti.f32, kappa: ti.f32,
             inv_rho: ti.f32) -> ti.types.vector(3, ti.f32):
    tot = ti.Vector([0.0, 0.0, 0.0])
    c = ti.Vector([cx, cy, cz])
    for p in range(sim.n_particles[None]):
        d = sim.x[p, s]-c; dist = d.norm()
        f = ti.Vector([0.0, 0.0, 0.0])
        if dist < r and dist > 1e-9:
            f = kappa*(r-dist)*(d/dist)*sim.p_mass[p]*inv_rho
        sim.ext_f[p] = f
        tot += f
    return tot


def _init_state():
    STATE['c'] = np.array(SPEC['c0'], float); STATE['v'] = np.array(SPEC['v0'], float); STATE['history'] = []


def substep(self, s, cache=True):
    local_index = s % self.cuda_chunk_size
    if s == 0:
        _init_state()
        n = int(self.n_particles[None])
        if self.ext_f.shape[0] < n:  # PAC-NeRF's replay builds the simulator before its particles exist
            self.ext_f = ti.Vector.field(3, dtype=self.dtype, shape=n)
    dt = float(self.dt[None])
    self.grid.deactivate_all()
    self.compute_F_tmp(local_index); self.svd(); self.project_F(local_index)
    self.p2g(local_index)
    c = STATE['c']
    tot = _contact(self, local_index, *map(float, c), float(SPEC['r']), float(SPEC['kappa']), 1.0/float(SPEC['rho_config']))
    self.ext_amp[None] = 1.0
    FA._scatter(self, local_index)
    self.grid_op(local_index); self.g2p(local_index); self.check_cfl(local_index+1)
    # sphere receives the opposite of the total force applied to the body
    STATE['v'] = STATE['v']-dt*np.array([tot[0], tot[1], tot[2]], float)/float(SPEC['m'])
    STATE['c'] = STATE['c']+dt*STATE['v']
    STATE['history'].append(((s+1)*dt, *STATE['c'], *STATE['v']))
    if (local_index == self.cuda_chunk_size-1) and cache:
        self.push_to_memory()


def substep_grad(self, s):
    raise RuntimeError('striker adapter is forecast-only')


def install(cls=None):
    if SPEC is None:
        return None
    if cls is None:
        from simulator.mpm_simulator import MPMSimulator as cls
    key = cls.__name__+str(id(cls))
    if key not in FA._ORIG:
        FA._ORIG[key] = cls.__init__
        cls.__init__ = FA._init
    cls.substep = substep; cls.substep_grad = substep_grad
    return SPEC
