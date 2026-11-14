"""Forecast a command from a fitted M-PhyGs material with the same native simulator.
Usage:
  forecast.py SCENE WORLD FIT_EP --ckpt CKPT --targets D2 [D2c D3 ...] [--frames F]
  forecast.py block WORLD D0 --truth-check            (sanity: true block material, nu=0.3, D0 force vs D0 obs)
  forecast.py SCENE WORLD FIT_EP --ckpt CKPT --consistency   (substep-loop vs native forward(), fitting command)
Writes runs/fig1-v01/mphygs/forecast-{EP}-{FIT_EP}-{WORLD}.npz (+ .json)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import warp as wp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mphygs_lib as L
import common as C
from mphygs.mpm_simple.warp_utils import from_torch_safe

DEV = 'cuda:0'


def setup(scene, world, fit_ep):
    root = C.OUT/scene/f'{fit_ep}-{world}'; exp = f'{scene}_{fit_ep}_{world}'
    cfg = json.loads((root/'config.json').read_text())
    inp = L.load_inputs(root, exp, DEV)
    x0 = inp['x']
    X = x0.cpu().numpy()
    bb = np.stack([X.min(0), X.max(0)]); cen = bb.mean(0); size = 2*np.max(bb[1]-bb[0])
    aabb = torch.tensor(np.stack([cen-size/2, cen+size/2]), dtype=torch.float32, device=DEV)
    sim = L.make_simulator(cfg, x0, aabb, torch.tensor(size), DEV)
    return cfg, inp, sim, aabb


def modifier(inp):
    return inp['modifier_mask'].float(), inp['x'], inp['x']  # static identity contact (native formula reduces to this)


@torch.no_grad()
def rollout_forward(cfg, inp, sim, aabb, mfunc, f_full, t_ramp, F, cpf=10):
    """Same per-frame native simulator.forward calls as the training / dump loops."""
    sim.set_force(f_full, t_ramp, cfg['frame_timestep'], audit=True); L.CLOCK.audit = []
    mv = mfunc(); x0 = inp['x']; m, a0, a1 = modifier(inp)
    st = [x0, torch.zeros_like(x0), torch.eye(3, device=DEV)[None].repeat(len(x0), 1, 1), torch.zeros((len(x0), 3, 3), device=DEV)]
    aabbs = L.FrameList([aabb]*(F-1)); xs = [x0.cpu().numpy()]
    for t in range(F-1):
        st = list(sim.forward(*st, **mv, dt=cfg['frame_timestep'], particle_modifier_mask=m, particle_modifier_x0=a0,
                              particle_modifier_x1=a1, init_particle_acc=torch.zeros_like(x0), sim_aabb=aabbs[t],
                              num_checkpoints_per_frame=cpf))
        xs.append(st[0].cpu().numpy())
    return np.stack(xs)


@torch.no_grad()
def rollout_substeps(cfg, inp, sim, aabb, mfunc, F, ext_fn, post_fn=None):
    """Explicit substep loop over the native kernels (SimplifiedMPMDiff.p2g2p), replicating
    MPMDifferentiableSimulationWCheckpoint.forward for a static modifier; ext_fn(x_world, t) -> (N,3) force is
    re-evaluated every substep (needed for the striker penalty)."""
    mv = mfunc(); x0 = inp['x']; m = inp['modifier_mask'].float(); dt = cfg['frame_timestep']
    nu = mv['particle_nu']; dens = (1-m)*mv['particle_density']+m*1000e4
    E = (1-m)*mv['particle_E']+m*sim.max_stiffness*dens
    mpm = sim.mpm_solver
    c = np.sqrt(torch.max(sim.max_stiffness*(1-nu)/(1+nu)/(1-2*nu)).item())
    h = min(0.04999*dt, sim.substep_margin*(mpm.grid_length/mpm.n_grid)/c); n = int(dt/h)
    mpm.n_particles = len(x0)
    mpm.setup_material_params(from_torch_safe(E.contiguous()), from_torch_safe(nu.contiguous()), from_torch_safe(dens.contiguous()))
    mpm.setup_particle_volume(from_torch_safe(mv['particle_volume'].contiguous()))
    mw = from_torch_safe(m.contiguous())
    rest = (x0-aabb[0]).contiguous()
    x = rest.clone(); v = torch.zeros_like(x); Fm = torch.eye(3, device=DEV)[None].repeat(len(x), 1, 1).contiguous()
    Cm = torch.zeros_like(Fm); xs = [x0.cpu().numpy()]; t = 0.0
    for fr in range(F-1):
        for s in range(n):
            vmod = (rest-x)/(h*1e4)  # compute_particle_modifier_v with x0 == x1 == rest
            ext = ext_fn(x+aabb[0], t).float().contiguous()
            xw, vw, Fw, Cw = mpm.p2g2p(from_torch_safe(x, dtype=wp.vec3), from_torch_safe(v, dtype=wp.vec3),
                                       from_torch_safe(Fm, dtype=wp.mat33), from_torch_safe(Cm, dtype=wp.mat33), dt=h,
                                       particle_boundary_V=from_torch_safe(vmod.contiguous(), dtype=wp.vec3),
                                       particle_boundary_mask=mw, particle_ext_force=from_torch_safe(ext, dtype=wp.vec3),
                                       init_particle_acc=None, residual_gravity=False)
            x, v, Fm, Cm = [wp.to_torch(q).clone() for q in (xw, vw, Fw, Cw)]
            t += h
            if post_fn is not None:
                post_fn(x+aabb[0], t, h)
        xs.append((x+aabb[0]).cpu().numpy())
    return np.stack(xs), dict(substep_dt=h, substeps_per_frame=n)


class Striker:
    """Rigid sphere, volumetric penalty f_p = kappa (r-|x-c|) n V_p (V_p = m_p/rho_config), opposite total force
    on the sphere, symplectic Euler, no gravity on the sphere (same law as striker_adapter.py)."""
    def __init__(self, spec, mass_p):
        self.s = spec; self.c = torch.tensor(spec['c0'], device=DEV, dtype=torch.float64)
        self.v = torch.tensor(spec['v0'], device=DEV, dtype=torch.float64); self.Vp = mass_p/spec['rho_config']
        self.hist = []; self.f = None

    def ext(self, x, t):
        d = x.double()-self.c; dist = d.norm(dim=1); inside = (dist < self.s['r']) & (dist > 1e-9)
        f = torch.zeros_like(d)
        f[inside] = (self.s['kappa']*(self.s['r']-dist[inside])*self.Vp[inside])[:, None]*d[inside]/dist[inside, None]
        self.f = f.sum(0)
        return f

    def post(self, x, t, h):
        self.v = self.v-h*self.f/self.s['m']; self.c = self.c+h*self.v
        self.hist.append([t, *self.c.tolist(), *self.v.tolist()])


def save(scene, ep, fit_ep, world, xyz, inp, extra, info):
    body = inp['ad']['body']
    out = dict(xyz=xyz[:, body].astype(np.float32), rest=inp['ad']['x'][body], labels=inp['ad']['labels'][body],
               fps=C.SCENES[scene]['fps'], **extra)
    if scene == 'stem':
        rx = inp['ad']['x'][body][:, 0]
        out['station_centroid'] = np.stack([xyz[:, body][:, (rx >= lo) & (rx < hi)].mean(1) for lo, hi in C.STATIONS], 1)
    p = C.OUT/f'forecast-{ep}-{fit_ep}-{world}'
    np.savez_compressed(p.with_suffix('.npz'), **out)
    p.with_suffix('.json').write_text(json.dumps(info, indent=1))
    print('wrote', p, json.dumps({k: v for k, v in info.items() if k != 'clusters'})[:600])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scene'); ap.add_argument('world'); ap.add_argument('fit_ep')
    ap.add_argument('--ckpt'); ap.add_argument('--targets', nargs='*', default=[])
    ap.add_argument('--frames', type=int, default=None); ap.add_argument('--truth-check', action='store_true')
    ap.add_argument('--consistency', action='store_true')
    a = ap.parse_args()
    cfg, inp, sim, aabb = setup(a.scene, a.world, a.fit_ep)
    S = C.SCENES[a.scene]; ad = inp['ad']
    if a.truth_check:  # sanity (b): uses the published scene definition of the block, D0 observations only
        assert a.scene == 'block' and a.fit_ep == 'D0'
        sys.path.insert(0, str(C.R/'scripts/fig1_v01')); import block as B
        X = ad['x']; xc = B.WORLDS[a.world]
        inc = (np.abs(X[:, 0]-xc) < B.INCL_HALF_X) & (X[:, 1] > B.INCL_Y0)
        Ef = np.where(inc, B.E_SOFT*B.CONTRAST, B.E_SOFT)
        f_full, t_ramp, audit = C.force_distribution('block', 'D0', X, ad['vol'], ad['body'])
        F = a.frames or 25; t0 = time.time()
        xyz = rollout_forward(cfg, inp, sim, aabb, L.truth_material_func(inp, Ef), f_full, t_ramp, F)
        obs = ad['markers_obs'][:F]; M = S['markers']
        from scipy.interpolate import RBFInterpolator
        pred = np.stack([M+RBFInterpolator(X[ad['body']], (xyz[k]-X)[ad['body']], neighbors=32, kernel='linear')(M) for k in range(F)])
        err = np.linalg.norm(pred-obs, axis=-1)*1e3
        info = dict(check='truth-material forward vs D0 RGB markers', world=a.world, seconds=time.time()-t0,
                    force_audit=audit, injected_total_force_at_end=L.CLOCK.audit[-1][2],
                    marker_err_mm_max=float(err.max()), marker_err_mm_mean=float(err.mean()),
                    obs_disp_mm_max=float(np.linalg.norm(obs-M, axis=-1).max()*1e3),
                    pred_disp_mm_max=float(np.linalg.norm(pred-M, axis=-1).max()*1e3),
                    per_frame_err_mm_max=err.max(1).round(3).tolist(),
                    pred_minus_obs_mean_dy_mm_last=float((pred-obs)[-1, :, 1].mean()*1e3))
        p = C.OUT/f'sanity-truthcheck-D0-{a.world}.json'; p.write_text(json.dumps(info, indent=1)); print(json.dumps(info))
        return
    mf = L.make_material(cfg, inp['onehot'].shape[1], DEV, a.ckpt); mfunc = L.make_material_func(mf, inp)
    clusters = L.cluster_params(mfunc, inp)
    if a.consistency:  # native forward() rollout vs explicit substep loop (used for the striker), soft homogeneous E
        F = a.frames or 4
        soft = L.truth_material_func(inp, np.full(len(ad['x']), 1e5))
        ff = torch.as_tensor(ad['f_full'], dtype=torch.float32, device=DEV); out = {}
        for name, tr in (('constant_force', 1e-9), ('ramped_force', float(ad['t_ramp']))):
            A = rollout_forward(cfg, inp, sim, aabb, soft, ad['f_full'], tr, F)
            B_, _ = rollout_substeps(cfg, inp, sim, aabb, soft, F, lambda x, t: ff*float(C.ramp(t, tr)))
            out[name] = dict(max_diff_mm=float(np.abs(A-B_).max()*1e3), max_disp_mm=float(np.abs(A-A[0]).max()*1e3))
        print(json.dumps(out)); (C.OUT/f'sanity-consistency-{a.scene}.json').write_text(json.dumps(out, indent=1))
        return
    for ep in a.targets:
        F = a.frames or S['forecast_frames']; t0 = time.time()
        info = dict(scene=a.scene, target=ep, fit_episode=a.fit_ep, world=a.world, ckpt=a.ckpt, frames=F,
                    fps=S['fps'], clusters=clusters, nu_fixed=0.3, density_fixed=C.RHO_FIXED)
        if ep == 'D3':
            spec = C.load_force(a.scene, ep)
            mass = (inp['modifier_mask'].float()*0+1)*C.RHO_FIXED*L.make_material_func(mf, inp)()['particle_volume']
            st = Striker(spec, mass.double())
            xyz, meta = rollout_substeps(cfg, inp, sim, aabb, mfunc, F, st.ext, st.post)
            info.update(meta, striker=spec); extra = dict(striker=np.array(st.hist, dtype=np.float64))
        else:
            f_full, t_ramp, audit = C.force_distribution(a.scene, ep, ad['x'], ad['vol'], ad['body'])
            xyz = rollout_forward(cfg, inp, sim, aabb, mfunc, f_full, t_ramp, F)
            info.update(force_audit=audit, injected_total_force_at_end=L.CLOCK.audit[-1][2],
                        substep_dt=sim.last_substep_dt, substeps_per_frame=sim.last_num_substeps); extra = {}
        info['seconds'] = time.time()-t0
        save(a.scene, ep, a.fit_ep, a.world, xyz, inp, extra, info)


if __name__ == '__main__':
    main()
