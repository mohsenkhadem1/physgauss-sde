"""Forecast held-out commands from a PhysTwin fit, plus sanity modes.
  forecast.py SCENE WORLD FIT_EP --targets D2 [D2c D3] [--tag TAG] [--frames F]
      -> runs/fig1-v01/phystwin/forecast-{EP}-{FIT_EP}-{WORLD}{TAG}.npz/.json
  forecast.py SCENE WORLD D0 --homog-scan        sanity (b): homogeneous spring_Y scan, native topology defaults,
                                                  D0 forward vs D0 RGB markers (fitting data only)
  forecast.py SCENE WORLD FIT_EP --consistency   explicit substep loop (used for the striker) vs native forward graph
Force commands use the native forward CUDA graph with the prescribed-force file swapped in.  The striker (D3) needs
the contact force re-evaluated every substep, so it runs an explicit substep loop over the SAME native kernels
(set_control_points, add_prescribed_forces, eval_springs, update_vel_from_force, integrate_ground_collision) plus a
penalty kernel: f_p = kappa (r-|x-c|) n V_p, V_p = m_p/rho_config; opposite total force on the sphere, sphere
integrated symplectically (v -= h F/m, c += h v), no gravity on the sphere (as striker_adapter.py)."""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptw

ap = argparse.ArgumentParser()
ap.add_argument('scene'); ap.add_argument('world'); ap.add_argument('fit_ep')
ap.add_argument('--targets', nargs='*', default=[]); ap.add_argument('--tag', default='')
ap.add_argument('--frames', type=int, default=None)
ap.add_argument('--homog-scan', action='store_true'); ap.add_argument('--set', nargs='*', default=[], help='homog-scan cfg overrides key=val'); ap.add_argument('--consistency', action='store_true')
a = ap.parse_args()
t_all = time.time(); peak = ptw.GpuPeak()
geo = ptw.geometry(a.scene); m0 = geo['m0']
FF = dict(block=25, stem=37)[a.scene]
ptw.import_phystwin()
import torch, warp as wp
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger
import qqtt.model.diff_simulator.spring_mass_warp as K

if a.homog_scan:
    sfx = ''.join('-'+kv for kv in a.set)
    fdir = ptw.OUT/'sanity'/f'homog-{a.scene}-{a.fit_ep}-{a.world}{sfx}'; fdir.mkdir(parents=True, exist_ok=True)
    data = ptw.dataset(a.scene, a.fit_ep, a.world, geo); F = len(data['object_points'])
    with open(fdir/'data.pkl', 'wb') as f:
        pickle.dump(data, f)
    acc, audit = ptw.force_series(a.scene, a.fit_ep, geo, F); np.savez_compressed(fdir/'forces.npz', forces=acc.astype(np.float32))
    cfg_dict = ptw.configure(cfg, a.scene, fdir/'forces.npz', {k: type(getattr(cfg, k))(v) for k, v in (kv.split('=') for kv in a.set)})
    params = None
else:
    fdir = ptw.fit_dir(a.scene, a.fit_ep, a.world, a.tag)
    params = json.loads((fdir/'params.json').read_text())
    with open(fdir/'data.pkl', 'rb') as f:
        data = pickle.load(f)
    cfg_dict = ptw.configure(cfg, a.scene, fdir/'forces.npz')
    with open(fdir/'stageA/optimal_params.pkl', 'rb') as f:
        cfg.set_optimal_params(dict(pickle.load(f)))
logger.set_log_file(path=str(fdir), name='forecast')


class Infer(InvPhyTrainerWarp):
    def visualize_sim(self, *args, **kw):
        pass


tr = Infer(str(fdir/'data.pkl'), str(fdir/'infer'), train_frame=len(data['object_points']), pure_inference_mode=True)
sim = tr.simulator; N = sim.num_object_points; dev = cfg.device
ctrl = torch.tensor(geo['ctrl'], dtype=torch.float32, device=dev)
if params is not None:
    ck = torch.load(params['stageB_checkpoint'], map_location=dev)
    assert len(ck['spring_Y']) == sim.n_springs
    sim.set_spring_Y(torch.log(ck['spring_Y']).detach().clone())


def set_episode(acc, F):
    sim.external_force_trajectory = torch.tensor(acc, dtype=torch.float32, device=dev)
    sim.controller_points = ctrl[None].repeat(F, 1, 1).contiguous()


def rollout_native(F):
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
    xs = [wp.to_torch(sim.wp_init_vertices, requires_grad=False).cpu().numpy().copy()]
    for j in range(1, F):
        sim.set_controller_target(j, pure_inference=True); wp.capture_launch(sim.forward_graph)
        xs.append(wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).cpu().numpy().copy())
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
    return np.stack(xs)


def injected_total(acc):
    """Read back what the native prescribed-force kernel adds at the last frame (alpha=1), minus gravity, in N."""
    set_episode(acc, len(acc)); sim.set_controller_target(len(acc)-1, pure_inference=True)
    f = wp.zeros(N, dtype=wp.vec3)
    wp.launch(K.add_prescribed_forces, dim=N, inputs=[sim.wp_external_start, sim.wp_external_end, 1.0], outputs=[f])
    return ((f.numpy().astype(np.float64).sum(0)-np.array([0, -ptw.G, 0])*N)*m0).tolist()


@wp.kernel
def striker_force(x: wp.array(dtype=wp.vec3), c: wp.array(dtype=wp.vec3d), r: float, kV: float, inv_m0: float,
                  f: wp.array(dtype=wp.vec3), ftot: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    cc = wp.vec3(float(c[0][0]), float(c[0][1]), float(c[0][2]))
    d = x[i]-cc
    dist = wp.length(d)
    if dist < r and dist > 1e-9:
        fp = kV*(r-dist)/dist*d
        f[i] = f[i]+fp*inv_m0
        wp.atomic_add(ftot, 0, wp.vec3d(wp.float64(fp[0]), wp.float64(fp[1]), wp.float64(fp[2])))


@wp.kernel
def sphere_step(c: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d), ftot: wp.array(dtype=wp.vec3d), h: wp.float64,
                M: wp.float64, t: wp.float64, k: int, hist: wp.array2d(dtype=wp.float64)):
    # force on the sphere is -ftot (reaction); symplectic Euler
    v[0] = v[0]-h*ftot[0]/M
    c[0] = c[0]+h*v[0]
    hist[k, 0] = t
    for j in range(3):
        hist[k, 1+j] = c[0][j]
        hist[k, 4+j] = v[0][j]


def rollout_loop(acc, F, striker=None):
    """Explicit substep loop with the native kernels (identical arithmetic to SpringMassSystemWarp.step)."""
    set_episode(acc, F); n = sim.num_substeps; dt = sim.dt
    A, B = K.State(sim.wp_init_vertices, sim.num_control_points), K.State(sim.wp_init_vertices, sim.num_control_points)
    wp.copy(A.wp_x, sim.wp_init_vertices); A.wp_v.zero_()
    cw = wp.from_torch(ctrl.contiguous(), dtype=wp.vec3)
    xs = [sim.wp_init_vertices.numpy().copy()]
    if striker is not None:
        s = striker; c = wp.array([s['c0']], dtype=wp.vec3d); v = wp.array([s['v0']], dtype=wp.vec3d)
        ftot = wp.zeros(1, dtype=wp.vec3d); hist = wp.zeros(((F-1)*n, 7), dtype=wp.float64)
        kV = float(s['kappa'])*m0/float(s['rho_config'])  # kappa * V_p, V_p = m_p / rho_config
    k = 0
    for j in range(1, F):
        sim.set_controller_target(j, pure_inference=True)  # loads external force start/end for this frame
        for i in range(n):
            A.clear_forces()
            wp.launch(K.set_control_points, dim=sim.num_control_points, inputs=[n, cw, cw, i], outputs=[A.wp_control_x])
            wp.launch(K.add_prescribed_forces, dim=N, inputs=[sim.wp_external_start, sim.wp_external_end, float(i+1)/float(n)],
                      outputs=[A.wp_vertice_forces])
            if striker is not None:
                ftot.zero_()
                wp.launch(striker_force, dim=N, inputs=[A.wp_x, c, float(s['r']), kV, 1.0/m0], outputs=[A.wp_vertice_forces, ftot])
            wp.launch(K.eval_springs, dim=sim.n_springs, inputs=[A.wp_x, A.wp_v, A.wp_control_x, A.wp_control_v, N, sim.wp_springs,
                      sim.wp_rest_lengths, sim.wp_spring_Y, sim.dashpot_damping, sim.spring_Y_min, sim.spring_Y_max],
                      outputs=[A.wp_vertice_forces])
            wp.launch(K.update_vel_from_force, dim=N, inputs=[A.wp_v, A.wp_vertice_forces, sim.wp_masses, dt, sim.drag_damping,
                      sim.reverse_factor], outputs=[A.wp_v_before_ground])
            wp.launch(K.integrate_ground_collision, dim=N, inputs=[A.wp_x, A.wp_v_before_ground, sim.wp_collide_elas,
                      sim.wp_collide_fric, dt, sim.reverse_factor], outputs=[B.wp_x, B.wp_v])
            if striker is not None:
                wp.launch(sphere_step, dim=1, inputs=[c, v, ftot, wp.float64(dt), wp.float64(s['m']), wp.float64(((j-1)*n+i+1)*dt), k],
                          outputs=[hist]); k += 1
            A, B = B, A
        xs.append(A.wp_x.numpy().copy())
    return np.stack(xs), (hist.numpy() if striker is not None else None)


def summary_params():
    if params is None:
        return None
    keep = {k: v for k, v in params.items() if k not in ('config',)}
    keep['stageB'] = {k: v for k, v in params['stageB'].items() if k not in ('loss_history', 'marker_rollout_log')}
    return keep


rest = geo['x']; nm = geo['n_markers']
if a.homog_scan:
    obs = data['object_points']; F = len(obs); acc, audit = ptw.force_series(a.scene, a.fit_ep, geo, F); set_episode(acc, F)
    rows = []
    for Y in (np.geomspace(1e2, 1e6, 17) if a.scene == 'block' else np.geomspace(1e3, 1e7, 17)):
        sim.set_spring_Y(torch.full((sim.n_springs,), float(np.log(Y)), device=dev))
        xs = rollout_native(F); err = np.linalg.norm(xs[:, :nm]-obs, axis=-1)
        rows.append(dict(spring_Y_native=float(Y), spring_Y_physical_N=float(Y*m0), finite=bool(np.isfinite(xs).all()),
                         marker_err_mm_mean=float(err.mean()*1e3), marker_err_mm_max=float(err.max()*1e3),
                         pred_marker_disp_mm_max=float(np.linalg.norm(xs[:, :nm]-rest[:nm], axis=-1).max()*1e3)))
        print(rows[-1], flush=True)
    good = [r for r in rows if r['finite']]; best = min(good, key=lambda r: r['marker_err_mm_mean'])
    info = dict(check='homogeneous spring_Y scan, native real.yaml topology/damping, %s forward vs %s RGB markers' % (a.fit_ep, a.fit_ep),
                scene=a.scene, world=a.world, episode=a.fit_ep, obs_disp_mm_max=float(np.linalg.norm(obs-obs[0], axis=-1).max()*1e3),
                best=best, overrides=a.set, scan=rows, injected_total_force_N=injected_total(acc), force_audit=audit,
                n_springs=int(sim.n_springs), seconds=time.time()-t_all, gpu_peak_mib=peak.stop(), **ptw.gpu_info())
    p = ptw.OUT/f'sanity-homog-{a.scene}-{a.fit_ep}-{a.world}{sfx}.json'; p.write_text(json.dumps(info, indent=1)); print('BEST', best)
    sys.exit(0)

if a.consistency:
    F = a.frames or 6; ep = 'D2' if a.scene == 'block' else 'D2c'
    acc, _ = ptw.force_series(a.scene, ep, geo, F)
    set_episode(acc, F); A_ = rollout_native(F); B_, _ = rollout_loop(acc, F)
    info = dict(check='explicit substep loop vs native forward graph', episode_force=ep, frames=F,
                max_diff_mm=float(np.abs(A_-B_).max()*1e3), max_disp_mm=float(np.abs(A_-A_[0]).max()*1e3))
    print(json.dumps(info)); (ptw.OUT/f'sanity-consistency-{a.scene}-{a.world}{a.tag}.json').write_text(json.dumps(info, indent=1))
    sys.exit(0)

for ep in a.targets:
    t0 = time.time(); F = a.frames or FF
    info = dict(method='PhysTwin', scene=a.scene, target=ep, fit_episode=a.fit_ep, world=a.world, frames=F, fps=ptw.FPS,
                substeps_per_frame=sim.num_substeps, substep_dt=sim.dt, fit=summary_params())
    extra = {}
    if ep == 'D3':
        spec = ptw.MC.load_force(a.scene, 'D3')
        acc = np.zeros((F, N, 3)); acc[:, :, 1] = -ptw.G
        xyz, hist = rollout_loop(acc, F, spec)
        extra['striker'] = hist
        info.update(striker=spec, striker_law='f_p = kappa (r-|x-c|) n V_p, V_p = m_p/rho_config, m_p = node mass',
                    impulse_on_sphere_Ns=float(spec['m']*(hist[-1, 6]-spec['v0'][2])), max_contact_nodes=None)
    else:
        acc, audit = ptw.force_series(a.scene, ep, geo, F)
        info.update(force_audit=audit, injected_total_force_at_end_N=injected_total(acc))
        set_episode(acc, F); xyz = rollout_native(F)
    assert np.isfinite(xyz).all(), 'non-finite rollout'
    xyz_all = np.concatenate([xyz, np.broadcast_to(geo['ctrl'], (F,)+geo['ctrl'].shape)], 1).astype(np.float32)
    rest_all = np.concatenate([rest, geo['ctrl']]); fixed = np.r_[np.zeros(len(rest), bool), np.ones(len(geo['ctrl']), bool)]
    out = dict(xyz=xyz_all, rest=rest_all, fixed=fixed, marker_index=np.arange(nm), fps=ptw.FPS, **extra)
    if a.scene == 'stem':
        out['station_centroid'] = ptw.station_centroid(xyz_all, rest_all)
    info['seconds'] = time.time()-t0; info.update(ptw.gpu_info())
    info['max_disp_mm'] = float(np.linalg.norm(xyz-rest, axis=-1).max()*1e3)
    p = ptw.OUT/f'forecast-{ep}-{a.fit_ep}-{a.world}{a.tag}'
    np.savez_compressed(p.with_suffix('.npz'), **out)
    info['gpu_peak_mib_process_so_far'] = peak.peak
    p.with_suffix('.json').write_text(json.dumps(info, indent=1))
    print('wrote', p, ep, 'seconds', round(info['seconds'], 1), 'max_disp_mm', round(info['max_disp_mm'], 3), flush=True)
print('FORECAST_COMPLETE', time.time()-t_all, 'gpu_peak_mib', peak.stop())
