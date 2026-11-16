"""Step-2 truths for the hidden-command variants of Figure-1 S2 (post-hoc namespace).

Same pipeline as scripts/fig1_v01/block.py (imported, not modified): nonlinear neo-Hookean
T10 on the (18,8,9) truth mesh, implicit Newmark dt=1/480, 25 frames at 48 fps, bottom glued,
gravity on from t=0, press = 4x4 cm x 1 cm body-load box ramped over 0.25 s.  The only change
is the press magnitude (FORCE * scale).  Optional --linear gives the small-strain reference on
the same mesh (diagnostic of the force-linearity assumption only).
Usage: truth.py EPISODE WORLD SCALE [--linear] [--nopress]   (EPISODE D0/D1/D2 or P<x>, e.g. P+0.06)
Writes runs/patch-roles-v02/truth/{EPISODE}-{WORLD}-a{SCALE}[-lin][-grav].npz
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'scripts/fig1_v01')); sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import fem, block as B

OUT = R/'runs/patch-roles-v02/truth'


def tag(episode, world, scale, linear=False, nopress=False):
    return f'{episode}-{world}-a{scale:g}' + ('-lin' if linear else '') + ('-grav' if nopress else '')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('episode'); ap.add_argument('world'); ap.add_argument('scale', type=float)
    ap.add_argument('--linear', action='store_true'); ap.add_argument('--nopress', action='store_true')
    a = ap.parse_args()
    m = B.mesh(B.TRUTH_CELLS); model = fem.Elastic(m, B.modulus(m, B.WORLDS[a.world]), B.NU, B.RHO)
    g = m.gravity(B.RHO)
    if a.episode.startswith('P'):   # calibrated probe press at x = float(episode[1:]) (same box as the protocol presses)
        x = float(a.episode[1:]); b = [[x-B.PATCH, B.TOP-0.01, -B.PATCH], [x+B.PATCH, B.TOP+1e-6, B.PATCH]]
    else:
        b = B.press_box(a.episode)
    f = 0*g if a.nopress else m.body_force_box(b[0], b[1], [0, -B.FORCE*a.scale, 0])
    dt = 1/480; t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    U, st = fem.newmark(model, lambda t: g+fem.smooth_ramp(t, B.EPISODE['ramp_s'])*f, B.EPISODE['duration_s'], dt,
                        round(1/(B.EPISODE['fps']*dt)), nonlinear=not a.linear, log=lambda s: print(a.episode, a.world, s, flush=True))
    name = tag(a.episode, a.world, a.scale, a.linear, a.nopress)
    np.savez_compressed(OUT/f'{name}.npz', X=m.X, U=U, elem=m.elem, cells=np.array(B.TRUTH_CELLS), scale=a.scale)
    info = dict(name=name, seconds=time.time()-t0, newton_mean=float(np.mean(st)) if st else 0., max_disp_mm=float(np.abs(U).max()*1e3))
    (OUT/f'{name}.json').write_text(json.dumps(info)); print(info)


if __name__ == '__main__':
    main()
