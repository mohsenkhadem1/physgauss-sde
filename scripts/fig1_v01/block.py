"""Figure-1 scenario 2: block with a hidden stiff region, pressed off-centre.
truth EPISODE WORLD  -> nonlinear neo-Hookean T10 reference (implicit Newmark), bottom face glued
scene EPISODE WORLD  -> PAC-NeRF-format multi-view renders (same 11 cameras as the stem)
The block exterior and texture are identical in every world; only the hidden inclusion moves."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; R = HERE.parents[1]
sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import fem

LO, HI = (-0.12, 0.53, -0.06), (0.12, 0.63, 0.06)
TOP = HI[1]
RHO, NU, E_SOFT, CONTRAST = 1000.0, 0.4, 6.0e4, 8.0
WORLDS = {'Bleft': -0.06, 'Bcentre': 0.0, 'Bright': 0.06}  # inclusion centre x
INCL_HALF_X, INCL_Y0 = 0.04, 0.555  # inclusion: |x-xc|<0.04, y>0.555 (upper 75%), full depth
PATCH = 0.02
PRESS = {'D0': 0.0, 'D1': 0.05, 'D2': -0.07}  # press centre x on the top face
FORCE = 30.0
EPISODE = dict(duration_s=0.5, fps=48, frames=25, ramp_s=0.25)
TRUTH_CELLS = (18, 8, 9)
OUT = R/'runs/fig1-v01/block'


def mesh(cells=TRUTH_CELLS):
    return fem.Mesh(cells, bounds=(LO, HI), fixed_fn=lambda X: X[:, 1] <= LO[1]+1e-9)


def modulus(m, xc, contrast=CONTRAST, e_soft=E_SOFT):
    c = m.element_centroids()
    inc = (np.abs(c[:, 0]-xc) < INCL_HALF_X) & (c[:, 1] > INCL_Y0) if xc is not None else np.zeros(len(c), bool)
    return np.where(inc, e_soft*contrast, e_soft)


def press_box(episode):
    """Top 1 cm of the body under a 4x4 cm window (adapters select the same layer from their own top surface)."""
    x = PRESS[episode]
    return [[x-PATCH, TOP-0.01, -PATCH], [x+PATCH, TOP+1e-6, PATCH]]


def force_fn(m, episode, rho=RHO):
    g = m.gravity(rho); b = press_box(episode)
    f = m.body_force_box(b[0], b[1], [0, -FORCE, 0])
    return lambda t: g+fem.smooth_ramp(t, EPISODE['ramp_s'])*f


# Observation markers (RGB-tracked) and scoring probes, all on the visible surface.
MARK_X = [-0.09, -0.045, 0.0, 0.045, 0.09]
MARKERS = np.array([[x, TOP, z] for z in (-0.03, 0.0, 0.03) for x in MARK_X]+[[x, 0.60, HI[2]] for x in MARK_X])
NORMALS = np.array([[0, 1., 0]]*15+[[0, 0, 1.]]*5)
PROBE_X = np.linspace(-0.10, 0.10, 9)
PROBES = np.array([[x, TOP, z] for z in (-0.04, 0.0, 0.04) for x in PROBE_X])


def truth(episode, world, cells=TRUTH_CELLS, tag=''):
    if episode == 'D2' and not (R/'reports/fig1-v01/prediction-freeze-block.json').exists():
        raise SystemExit('block D2 truth sealed until predictions are frozen')
    m = mesh(cells); model = fem.Elastic(m, modulus(m, WORLDS[world]), NU, RHO)
    dt = 1/480; t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    U, st = fem.newmark(model, force_fn(m, episode), EPISODE['duration_s'], dt, round(1/(EPISODE['fps']*dt)),
                        log=lambda s: print(episode, world, s, flush=True))
    assert len(U) == EPISODE['frames']
    np.savez_compressed(OUT/f'truth/{episode}-{world}{tag}.npz', X=m.X, U=U, elem=m.elem, cells=np.array(cells))
    print(dict(episode=episode, world=world, seconds=time.time()-t0, newton_mean=float(np.mean(st)),
               max_disp_mm=float(np.abs(U).max()*1e3)))


def scene(episode, world):
    sys.path.insert(0, str(R/'scripts/stem_trial_v01')); import render
    z = np.load(OUT/f'truth/{episode}-{world}.npz'); m = mesh(tuple(z['cells']))
    out = OUT/f'scenes/{episode}-{world}'
    render.write_scene(out, z['X'][None]+z['U'], z['X'], m.surface_triangles(), EPISODE['fps'])
    (out/'render-meta.json').write_text(json.dumps(dict(episode=episode, world=world, gic_camera_deviation=render.gic_camera_check())))
    print('scene', out)


if __name__ == '__main__':
    (OUT/'truth').mkdir(parents=True, exist_ok=True)
    {'truth': truth, 'scene': scene}[sys.argv[1]](*sys.argv[2:4])
