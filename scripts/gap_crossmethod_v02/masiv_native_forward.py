"""Native MASIV solver/constitutive identifiability diagnostic, NOT a fitted MASIV benchmark.

Stress and density scale together; gravity-only response should be invariant.
A known uniform total force is converted to acceleration through the native
gravity input. This is a declared force-input adapter, not an upstream force API.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'runs/gap-crossmethod-v02/vendor/masiv'
sys.path.insert(0, str(UPSTREAM))
from nclaw.sim import MPMModelBuilder
from nclaw.material import InvariantFullMetaElasticity, InvariantFullMetaPlasticity


def geometry(name):
    d = .018
    q = np.stack(np.meshgrid(*[np.arange(-.15, .151, d)]*3, indexing='ij'), -1).reshape(-1, 3)
    if name == 'torus':
        keep = (np.sqrt(q[:, 0]**2+q[:, 2]**2)-.095)**2+q[:, 1]**2 < .045**2
    elif name == 'ellipsoid':
        keep = ((q/np.array([.13, .07, .09]))**2).sum(-1) < 1
    else:
        raise ValueError(name)
    x = q[keep] + [0.5, .16, .5]
    return x.astype(np.float32), d**3


def run(args):
    torch.set_num_threads(2)
    wp.init()
    cfg = OmegaConf.load(UPSTREAM/'config/default.yaml')
    cfg.sim.dt = args.dt
    cfg.sim.num_grids = args.grid
    cfg.sim.bc = args.boundary
    model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(args.device, requires_grad=False)
    x0, volume = geometry(args.shape)
    n = len(x0)
    statics = model.statics(n)
    statics.update_vol([n], [volume])
    statics.update_rho([n], [1000.*args.scale])
    statics.update_clip_bound([n], [.5])
    elastic = InvariantFullMetaElasticity(cfg.meta.elasticity).to(args.device)
    plastic = InvariantFullMetaPlasticity(cfg.meta.plasticity).to(args.device)
    weights = torch.load(UPSTREAM/'pretrained/jelly_0300.pt', map_location=args.device, weights_only=True)
    elastic.load_state_dict(weights['elasticity'], strict=True)
    plastic.load_state_dict(weights['plasticity'], strict=True)
    mass = n*volume*1000.*args.scale
    states = [model.state(n), model.state(n)]
    x = torch.tensor(x0, device=args.device)
    v = torch.zeros_like(x)
    C = torch.zeros(n, 3, 3, device=args.device)
    F = torch.eye(3, device=args.device).repeat(n, 1, 1)
    center = torch.tensor([.60, .16, .50], device=args.device)
    force_weights = torch.exp(-((x-center)**2).sum(-1)/(2*.045**2))
    force_weights /= force_weights.sum()
    frames, determinants = [x0], [1.]
    times = [0.]
    with torch.no_grad():
        for step in range(args.steps):
            t = step*args.dt
            active_force = args.force if args.force_start <= t < args.force_end else 0.
            accel = [0., -9.8, 0.]
            axis = {'x':0, 'y':1, 'z':2}[args.force_axis]
            if args.force_profile == 'uniform':
                accel[axis] += active_force/mass
            else:
                # Declared localized force adapter: explicit known impulse before P2G.
                v = v.clone()
                v[:, axis] += args.dt*active_force*force_weights/(1000.*args.scale*volume)
            model.constant.gravity = wp.vec3(*accel)
            stress = args.scale*args.constitutive_scale*elastic(F)
            current, nxt = states[step % 2], states[(step+1) % 2]
            current.from_torch(x=x, v=v, C=C, F=F, stress=stress)
            model.forward(statics, current, nxt, None)
            x, v, C, trial, _ = nxt.to_torch()
            F = plastic(trial)
            if (step+1) % args.save_every == 0:
                arr = x.cpu().numpy().copy()
                det = float(torch.linalg.det(F).min())
                if not np.isfinite(arr).all() or not np.isfinite(det) or det <= 0:
                    raise RuntimeError(f'Invalid state step {step}: min J={det}')
                frames.append(arr)
                determinants.append(det)
                times.append((step+1)*args.dt)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.array(frames), time=np.array(times), min_J=np.array(determinants))
    meta = dict(vars(args), particles=n, mass_kg=mass, particle_volume_m3=volume,
                min_J=min(determinants), max_displacement_m=float(np.linalg.norm(frames[-1]-x0, axis=-1).max()),
                classification='native MPM and neural constitutive forward diagnostic; no scene fitting or native appearance training',
                force_adapter='uniform F/M native gravity or normalized rest-coordinate Gaussian impulse before native P2G, as declared by force_profile',
                density_assumption='density uncertain; this tests an extended setting, not failure under known-density native assumptions')
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shape', choices=['torus','ellipsoid'], default='torus')
    p.add_argument('--scale', type=float, default=1.)
    p.add_argument('--constitutive-scale', type=float, default=1.)
    p.add_argument('--boundary', choices=['freeslip','noslip'], default='freeslip')
    p.add_argument('--force', type=float, default=0.)
    p.add_argument('--force-profile', choices=['uniform','gaussian'], default='uniform')
    p.add_argument('--force-axis', choices=['x','y','z'], default='x')
    p.add_argument('--force-start', type=float, default=.12)
    p.add_argument('--force-end', type=float, default=.16)
    p.add_argument('--dt', type=float, default=.0001)
    p.add_argument('--steps', type=int, default=2400)
    p.add_argument('--grid', type=int, default=32)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--output', required=True)
    run(p.parse_args())
