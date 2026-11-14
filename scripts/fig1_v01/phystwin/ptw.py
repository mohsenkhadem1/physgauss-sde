"""Official PhysTwin (spring-mass digital twin, Jiang et al. 2025) as an external Figure-1 baseline.

Library shared by fit.py / forecast.py / sanity.py.  Uses the isolated PhysTwin copy made by an earlier campaign
(runs/third_party/PhysTwin-v06 = upstream 81c71879 + controlled-physics.patch (opt-in gravity_scale) +
v06-prescribed-forces.patch (opt-in per-node external force file)); nothing in it is modified here.

Conventions
- Native PhysTwin uses UNIT MASS per node.  All our nodes carry the same mass m0 = rho*V/N (rho = 1000 fixed,
  as for every baseline), so the dynamics with masses m0 are identical to unit masses with every force divided by
  m0.  We therefore keep native unit masses, native parameter ranges (spring_Y, dashpot, drag are "per unit mass")
  and supply the known external force as force/m0.  Physical spring stiffness = spring_Y * m0 (reported).
- Gravity (-y, 9.8) is supplied in the same external-force array; native z-gravity is switched off with
  gravity_scale=0 (which also disables the native z=0 ground plane: there is no ground contact in our scenes).
- Fixed boundary (glued block bottom face / stem clamp) = native PhysTwin controller points held still, coupled to
  the object by native controller springs.
Nothing here reads held-out truth."""
import json, os, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parents[2]
sys.path.insert(0, str(R/'scripts/fig1_v01/mphygs'))
import common as MC  # scene constants (markers, stations, force files, forbidden truth paths); numpy only

OUT = R/'runs/fig1-v01/phystwin'
UP = R/'runs/third_party/PhysTwin-v06'
UPSTREAM_COMMIT = '81c718790a37e5e0102eb77af2c6edd34a9db25f'
RHO = MC.RHO_FIXED
G = 9.8
# Node spacing: cell-centred lattice.  Chosen so that no node lies on a load-window / slab / clamp / station
# boundary (block 1 cm; stem 8 mm because 1 cm cell centres hit the stem slab edges +-0.015, 0.085, 0.115).
SPACING = dict(block=0.01, stem=0.008)
FPS = 48
DT_NATIVE = 5e-5
# Scene-level departures from real.yaml (see README).  Filled in from the D0 stability/range scans.
# stem: best homogeneous D0 spring_Y ~1e5 sits on the native clamp spring_Y_max=1e5 -> 1e6; explicit integration
# diverges above 5.6e5 at 417 substeps, above 1.8e6 at 834 -> 834 substeps (dt 2.5e-5). Block: fully native.
SCENE_CFG = dict(block={}, stem=dict(spring_Y_max=1e6, num_substeps=834))


def check_path(p):
    return MC.check_path(p)


def ramp(t, tr):
    return 0.5*(1-np.cos(np.pi*np.clip(np.asarray(t)/tr, 0, 1)))


def geometry(scene):
    """Rest geometry as PhysTwin structure points + fixed controller points.
    Returns dict: x (N,3) object nodes ordered [markers | surface-layer nodes | interior nodes], n_markers,
    n_surface (markers + surface nodes, = native num_surface_points), ctrl (Nc,3) fixed boundary points, m0."""
    S = MC.SCENES[scene]; lo, hi = np.array(S['lo']), np.array(S['hi']); h = SPACING[scene]
    n = np.round((hi-lo)/h).astype(int); assert np.allclose(n*h, hi-lo)
    ax = [lo[d]+h*(np.arange(n[d])+0.5) for d in range(3)]
    X = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing='ij'), -1).reshape(-1, 3)
    M = S['markers'].astype(float)
    n_cells = len(X)
    if scene == 'block':  # glued bottom face y = lo_y: controllers ON the face, at the cell-centre (x,z) grid
        ctrl = np.stack(np.meshgrid(ax[0], [lo[1]], ax[2], indexing='ij'), -1).reshape(-1, 3)
        free = np.ones(len(X), bool)
    else:  # stem: every node with rest x <= clamp_x is fixed (as fem.py); they become the controllers
        free = X[:, 0] > S['clamp_x']
        ctrl = X[~free]
    Xf, idf = X[free], idx[free]
    surf = np.any((idf == 0) | (idf == n-1), 1)
    if scene == 'stem':  # the clamp interface is not a free surface
        surf = np.any((idf[:, 1:] == 0) | (idf[:, 1:] == n[1:]-1), 1) | (idf[:, 0] == n[0]-1)
    dmin = np.min(np.linalg.norm(Xf[:, None]-M[None], axis=-1))
    assert dmin > 0.3*h, 'marker coincides with a lattice node'
    x = np.concatenate([M, Xf[surf], Xf[~surf]])
    V = np.prod(hi-lo)
    m0 = RHO*V/(n_cells+len(M))  # uniform node mass; dynamic mass = m0 * len(x)
    return dict(x=x, n_markers=len(M), n_surface=len(M)+int(surf.sum()), ctrl=ctrl, m0=m0, spacing=h,
                n_cells=n_cells, total_mass_dynamic=m0*len(x), body_mass=RHO*V)


def load_sel(scene, spec, x):
    """Same selection rule as stem_trial_v01/gic_force_adapter._prepare, on the dynamic nodes' rest positions."""
    if 'window' in spec:
        (x0, z0), (x1, z1) = spec['window']
        inwin = (x[:, 0] >= x0) & (x[:, 0] < x1) & (x[:, 2] >= z0) & (x[:, 2] < z1)
        sel = inwin & (x[:, 1] >= x[inwin, 1].max()-spec['top'])
    else:
        sel = (x[:, 0] >= spec['lo']) & (x[:, 0] < spec['hi'])
    assert sel.sum() > 0
    return sel


def force_series(scene, ep, geo, F):
    """(F, N, 3) external force PER UNIT MASS (acceleration) at frame times k/48: gravity + ramp(t) * command
    distributed by mass over the selected nodes (uniform masses -> equal shares).  Stem D0 = gravity only."""
    x, m0 = geo['x'], geo['m0']; t = np.arange(F)/FPS
    acc = np.zeros((F, len(x), 3)); acc[:, :, 1] = -G
    audit = dict(episode=ep, m0=m0, frames=F)
    if scene == 'stem' and ep == 'D0':
        audit.update(note='stem D0 is gravity release only', particles_loaded=0, command=[0, 0, 0]); return acc, audit
    spec = MC.load_force(scene, ep); sel = load_sel(scene, spec, x)
    f_node = np.zeros_like(x); f_node[sel] = np.asarray(spec['force'])[None]/sel.sum()  # N per node
    acc += ramp(t, spec['t_ramp'])[:, None, None]*(f_node/m0)[None]
    audit.update(particles_loaded=int(sel.sum()), command=spec['force'], t_ramp=spec['t_ramp'],
                 total_force_full_ramp=(f_node.sum(0)).tolist(),
                 loaded_mass=float(m0*sel.sum()), ramp_sampling='0.5(1-cos) sampled at frames, linear within a frame (native prescribed-force interpolation)')
    return acc, audit


def observations(scene, ep, world, geo):
    """RGB-triangulated marker tracks of a FITTING episode (D0/D1 only)."""
    assert ep in ('D0', 'D1'), 'only fitting episodes are observed'
    z = np.load(check_path(MC.SCENES[scene]['obs'](ep, world)))
    y = z['y'].astype(np.float32); st = z['status'].astype(bool)
    M = geo['x'][:geo['n_markers']]
    assert np.abs(y[0]-M).max() < 1e-4, 'marker order / rest mismatch'
    y[0] = M  # frame 0 = known rest geometry
    return y, st


def dataset(scene, ep, world, geo, frames=None):
    """Native PhysTwin final_data.pkl-style dict (markers = tracked 'object_points')."""
    y, st = observations(scene, ep, world, geo)
    if frames:
        y, st = y[:frames], st[:frames]
    F = len(y); nm, ns = geo['n_markers'], geo['n_surface']
    return dict(object_points=y, object_colors=np.full((F, nm, 3), .5, np.float32), object_visibilities=st,
                object_motions_valid=st[1:] & st[:-1],
                controller_points=np.broadcast_to(geo['ctrl'], (F,)+geo['ctrl'].shape).astype(np.float32).copy(),
                surface_points=geo['x'][nm:ns].astype(np.float32), interior_points=geo['x'][ns:].astype(np.float32))


# --------------------------------------------------------------------------------------------- PhysTwin import
def import_phystwin():
    os.environ['PHYSTWIN_HEADLESS'] = '1'; os.environ['WANDB_MODE'] = 'disabled'
    sys.path.insert(0, str(UP))
    src = (UP/'qqtt/model/diff_simulator/spring_mass_warp.py').read_text()
    assert 'getattr(cfg, "gravity_scale", 1.0)' in src and 'external_force_path' in src, 'need the v06 PhysTwin copy'
    import qqtt.data.real_data as real_data
    import qqtt.engine.cma_optimize_warp as cma_module
    import qqtt.engine.trainer_warp as trainer_module
    nop = lambda *a, **k: None
    real_data.visualize_pc = nop; cma_module.visualize_pc = nop; trainer_module.visualize_pc = nop  # previews only
    import qqtt
    return qqtt


def configure(cfg, scene, force_path, overrides=None):
    """Native real.yaml, then only the scene adaptations listed in README (all recorded in params JSON)."""
    cfg.load_from_yaml(str(UP/'configs/real.yaml'))
    n_sub = round(1/FPS/DT_NATIVE)
    cfg.FPS = FPS; cfg.num_substeps = n_sub; cfg.dt = 1/FPS/n_sub
    cfg.gravity_scale = 0.0; cfg.reverse_z = False  # gravity -y supplied in the force array; no ground plane
    cfg.external_force_path = str(force_path)
    cfg.vis_interval = 20
    for k, v in {**SCENE_CFG[scene], **(overrides or {})}.items():
        setattr(cfg, k, v)
    cfg.num_substeps = int(cfg.num_substeps); cfg.dt = 1/FPS/cfg.num_substeps  # keep frame time exact
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in cfg.to_dict().items()
            if k not in ('c2ws', 'w2cs', 'intrinsics')}


# --------------------------------------------------------------------------------------------- monitoring
class GpuPeak:
    """Samples this process's GPU memory (nvidia-smi) every 2 s."""
    def __init__(self):
        self.peak = 0; self._stop = False; self.pid = os.getpid()
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _run(self):
        while not self._stop:
            try:
                out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=20).stdout
                for line in out.strip().splitlines():
                    p, m = [s.strip() for s in line.split(',')]
                    if int(p) == self.pid:
                        self.peak = max(self.peak, int(m))
            except Exception:
                pass
            time.sleep(2)

    def stop(self):
        self._stop = True; return self.peak


def gpu_info():
    dev = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    try:
        name = subprocess.run(['nvidia-smi', '-i', dev, '--query-gpu=name', '--format=csv,noheader'], capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:
        name = '?'
    return dict(physical_gpu=dev, gpu_name=name)


def fit_dir(scene, ep, world, tag=''):
    return OUT/scene/f'{ep}-{world}{tag}'


def station_centroid(xyz, rest):
    return np.stack([xyz[:, (rest[:, 0] >= lo) & (rest[:, 0] < hi)].mean(1) for lo, hi in MC.STATIONS], 1)
