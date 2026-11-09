"""G3 parity tests for gic_force_adapter (run with gic-env python from the GIC vendor dir).
Mode 'native' | 'zero' | 'momentum' | 'grad' ; each mode runs in its own process (Taichi fields persist).
"""
import sys, os, json
from pathlib import Path
from argparse import Namespace
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'runs/gap-crossmethod-v02/vendor/gic')); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, torch, taichi as ti
mode, out = sys.argv[1], Path(sys.argv[2])
mode0 = mode; mode = 'native' if mode == 'native2' else mode
spec = {'native': None, 'zero': dict(lo=-0.02, hi=0.02, force=[0, 0, 0], t_ramp=0.05),
        'momentum': dict(lo=-0.02, hi=0.02, force=[0, 0, 5.0], t_ramp=1e-9),
        'grad': dict(lo=-1, hi=1, force=[1.0, -2.0, 3.0], t_ramp=1e-9)}[mode]
if spec is not None:
    f = out.with_suffix('.spec.json'); f.write_text(json.dumps(spec)); os.environ['STEM_FORCE_JSON'] = str(f)
import gic_force_adapter as A
A.install()
ti.init(arch=ti.cuda, default_fp=ti.f32, device_memory_fraction=.3, random_seed=0, dynamic_index=True)
from simulator import Simulator
g = np.stack(np.meshgrid(np.arange(-.06, .06, .01), np.arange(.55, .65, .01), np.arange(-.04, .04, .01), indexing='ij'), -1).reshape(-1, 3)
vol = torch.tensor(g, dtype=torch.float32, device='cuda')
if mode == 'grad':
    from simulator.mpm_simulator import MPMSimulator
    sim = Simulator(Namespace(gravity=[0, 0, 0], voxel_size=.02, fps=24, mpm_iter_cnt=50, vel=[0, 0, 0], rho=1000., bc={},
                              mat_params=dict(material=10, E=1e5, nu=.3)), vol)
    sim.initialize(); s = sim.sim; s.ext_amp[None] = 1.0
    A._SPEC = spec; A._prepare(s)
    r = ti.Vector.field(3, ti.f32, shape=(64, 64, 64)); r.from_numpy(np.random.default_rng(0).standard_normal((64, 64, 64, 3)).astype(np.float32))
    L = ti.field(ti.f32, shape=(), needs_grad=True)
    @ti.kernel
    def loss():
        for I in ti.grouped(s.grid_v_in):
            L[None] += s.grid_v_in[I].dot(r[I + 32])
    def run():
        s.grid.deactivate_all(); L[None] = 0
        A._scatter(s, 0); loss(); return L[None]
    @ti.kernel
    def getp(p: ti.i32, d: ti.template()) -> ti.f32:
        return s.x[p, 0][d]
    @ti.kernel
    def setp(p: ti.i32, d: ti.template(), v: ti.f32):
        s.x[p, 0][d] = v
    @ti.kernel
    def getg(p: ti.i32, d: ti.template()) -> ti.f32:
        return s.x.grad[p, 0][d]
    L0 = run(); L.grad[None] = 1
    loss.grad(); A._scatter.grad(s, 0)
    errs = []
    for p in [37, 401, 900]:
        for d in range(3):
            ad = getg(p, d); x0 = getp(p, d); e = 2e-4
            setp(p, d, x0+e); Lp = run(); setp(p, d, x0-e); Lm = run(); setp(p, d, x0)
            errs.append(dict(p=p, d=d, ad=float(ad), fd=float((Lp-Lm)/(2*e))))
    out.write_text(json.dumps(dict(mode=mode, grad=errs))); print(errs); raise SystemExit
bc = {} if mode == 'momentum' else {'cylinder_clamp': [[-0.1, .6, 0], [-0.04, .6, 0], .08, 0]}
grav = [0, 0, 0] if mode == 'momentum' else [0, -9.8, 0]
sim = Simulator(Namespace(gravity=grav, voxel_size=.02, fps=24, mpm_iter_cnt=50, vel=[0, 0, 0], rho=1000., bc=bc,
                          mat_params=dict(material=10, E=1e5, nu=.3)), vol)
from new_trajectory import gen_xyz_list
seq = torch.stack(gen_xyz_list(sim, 12, diff=False)).cpu().numpy()
res = dict(mode=mode, final=seq[-1].tolist(), audit=A.AUDIT)
if mode == 'momentum':
    T = 11/24; n = len(g); M = 1000*(.01)**3*n
    v = (seq[-1]-seq[-2]).mean(0)*24
    res.update(expected_centroid_z=5.0*T**2/2/M, measured_centroid_z=float((seq[-1]-seq[0]).mean(0)[2]), total_mass=M)
np.save(out.with_suffix('.npy'), seq); out.write_text(json.dumps(res)); print({k: v for k, v in res.items() if k != 'final'})
