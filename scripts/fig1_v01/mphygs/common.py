"""Scene definitions and input synthesis helpers for the M-PhyGs material-estimation adapter (numpy only).

Everything here is information every method in the benchmark receives: rest geometry, the glued / clamped
boundary, the known applied command, RGB-triangulated marker tracks of the FITTING episodes (D0, D1).
Nothing reads held-out truth (block D2, stem D2c / D3 references)."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[3]
OUT = R/'runs/fig1-v01/mphygs'
VENDOR = R/'runs/gap-crossmethod-v02/vendor/m-phygs'

H = 1/150  # particle spacing (m); MPM cell = 2H (8 particles per cell)
GRID_PER_M = 75  # M-PhyGs "stage_*_grid_G" number: cells per metre -> dx = 1/G = 2H
K_CLUSTERS = 12
RHO_FIXED = 1000.0
GRAVITY = [0.0, -9.8, 0.0]
# M-PhyGs config max_stiffness bounds E/rho of the material field AND sets the CFL substep (native confs: 1e5).
# 2e4 m^2/s^2 (E <= 2e7 Pa at 1000 kg/m^3) is >= 5x above every reference E/rho in both scenes and runs
# sqrt(5)=2.2x fewer substeps than 1e5. Same value for every scene/world/episode.
MAX_STIFFNESS = 2e4

# --- block (Figure-1 S2), constants mirrored from scripts/fig1_v01/block.py ---
_MARK_X = [-0.09, -0.045, 0.0, 0.045, 0.09]
_BLOCK_MARKERS = np.array([[x, 0.63, z] for z in (-0.03, 0.0, 0.03) for x in _MARK_X] + [[x, 0.60, 0.06] for x in _MARK_X])
# --- stem (S1/S3), constants mirrored from scripts/stem_trial_v01/{fem,ours}.py ---
_MX = [-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.18]
_STEM_MARKERS = np.array([[x, .60, .04] for x in _MX] + [[x, .64, 0.] for x in _MX])
STATIONS = [(-.11, -.09), (-.01, .01), (.09, .11), (.18, .2001)]

SCENES = {
    'block': dict(lo=(-0.12, 0.53, -0.06), hi=(0.12, 0.63, 0.06), fps=48, markers=_BLOCK_MARKERS,
                  worlds=['Bleft', 'Bcentre', 'Bright'], fit_episodes=['D0', 'D1'], heldout=['D2'],
                  forecast_frames=25,
                  obs=lambda ep, w: R/f'runs/fig1-v01/block/rgb/obs-{ep}-{w}.npz',
                  force=lambda ep: R/f'runs/fig1-v01/block/forces/force-{ep}.json'),
    'stem': dict(lo=(-0.24, 0.56, -0.04), hi=(0.20, 0.64, 0.04), fps=48, markers=_STEM_MARKERS, clamp_x=-0.20,
                 worlds=['W500', 'W1000', 'W2000'], fit_episodes=['D0', 'D1'], heldout=['D2c', 'D3'],
                 forecast_frames=37,
                 obs=lambda ep, w: R/f'runs/stem-trial-v01/rgb/obs-{ep}-{w}.npz',
                 force=lambda ep: (R/'runs/stem-trial-v01/gic/striker-D3.json' if ep == 'D3' else
                                   R/f'runs/stem-trial-v01/gic/force-{ep}.json')),
}
FORBIDDEN = ('runs/stem-trial-v01/truth', 'runs/fig1-v01/block/truth')


def check_path(p):
    s = str(p)
    assert not any(f in s for f in FORBIDDEN), f'refusing to touch truth path {s}'
    return p


def sample_particles(scene):
    """Uniform grid sample of the rest geometry. Returns x (N,3), volume (N,), sdf (N,), pinned (N,) bool,
    body (N,) bool. Block: two extra pinned layers below the glued bottom face (a rigid base, like the pot in
    the native M-PhyGs scenes). Stem: body particles in the clamp (x <= -0.20) are pinned."""
    S = SCENES[scene]; lo, hi = np.array(S['lo']), np.array(S['hi'])
    n = np.round((hi-lo)/H).astype(int)
    ax = [lo[d]+H*(np.arange(n[d])+0.5) for d in range(3)]
    x = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    body = np.ones(len(x), bool)
    if scene == 'block':
        base = np.stack(np.meshgrid(ax[0], lo[1]-H*(np.arange(2)+0.5), ax[2], indexing='ij'), -1).reshape(-1, 3)
        x = np.concatenate([x, base]); body = np.concatenate([body, np.zeros(len(base), bool)])
        pinned = ~body
    else:
        pinned = x[:, 0] <= S['clamp_x']
    d = np.minimum(x-lo, hi-x).min(1)  # box SDF (negative inside); base particles get the same sign convention
    sdf = -np.abs(d)
    vol = np.full(len(x), H**3)
    return x, vol, sdf, pinned, body


def kmeans_labels(x, body, k=K_CLUSTERS, seed=0):
    """Geometry-only clusters on rest positions of the body (blind to any inclusion); pinned base gets its own
    label slot appended (it is overridden by the modifier anyway)."""
    from sklearn.cluster import KMeans
    lab = np.full(len(x), k, int)
    lab[body] = KMeans(k, n_init=10, random_state=seed).fit_predict(x[body])
    return lab


def load_force(scene, ep):
    return json.loads(check_path(SCENES[scene]['force'](ep)).read_text())


def force_distribution(scene, ep, x, vol, body, rho=RHO_FIXED):
    """Per-particle force (N,3) at full amplitude, same selection rule as gic_force_adapter._prepare, mass-weighted."""
    spec = load_force(scene, ep)
    if 'window' in spec:
        (x0, z0), (x1, z1) = spec['window']
        inwin = body & (x[:, 0] >= x0) & (x[:, 0] < x1) & (x[:, 2] >= z0) & (x[:, 2] < z1)
        sel = inwin & (x[:, 1] >= x[inwin, 1].max()-spec['top'])
    else:
        sel = body & (x[:, 0] >= spec['lo']) & (x[:, 0] < spec['hi'])
    assert sel.sum() > 0
    m = rho*vol
    f = np.zeros((len(x), 3))
    f[sel] = np.asarray(spec['force'])[None]*(m[sel]/m[sel].sum())[:, None]
    return f, float(spec['t_ramp']), dict(particles_loaded=int(sel.sum()), total_force=f.sum(0).tolist())


def ramp(t, t_ramp):
    return 0.5*(1-np.cos(np.pi*np.clip(t/t_ramp, 0, 1)))


def anchors(scene):
    """Zero-displacement points on the known fixed boundary (glued bottom face / clamp), used only to
    constrain the marker interpolation (every method knows the boundary condition)."""
    S = SCENES[scene]; lo, hi = S['lo'], S['hi']
    if scene == 'block':
        g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 9), [lo[1]], np.linspace(lo[2], hi[2], 5), indexing='ij'), -1)
    else:
        g = np.stack(np.meshgrid([lo[0], S['clamp_x']], np.linspace(lo[1], hi[1], 3), np.linspace(lo[2], hi[2], 3), indexing='ij'), -1)
    return g.reshape(-1, 3)


def dense_tracks(scene, ep, world, x_rest):
    """Thin-plate RBF interpolation (rest coordinates) of RGB-triangulated marker displacements to every
    particle, per frame. Returns (F, N, 3) absolute positions, frame 0 = rest."""
    from scipy.interpolate import RBFInterpolator
    z = np.load(check_path(SCENES[scene]['obs'](ep, world)))
    y = z['y']; M = SCENES[scene]['markers']
    assert np.abs(y[0]-M).max() < 1e-4, 'marker order / rest mismatch'
    A = anchors(scene); P = np.concatenate([M, A])
    out = np.empty((len(y), len(x_rest), 3))
    for k in range(len(y)):
        d = np.concatenate([y[k]-M, np.zeros_like(A)])
        out[k] = x_rest+RBFInterpolator(P, d, kernel='thin_plate_spline', smoothing=1e-6)(x_rest)
    out[0] = x_rest
    return out, y
