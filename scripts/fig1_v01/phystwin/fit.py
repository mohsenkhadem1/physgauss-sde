"""Fit official PhysTwin to one fitting episode with its native two-stage optimisation.
Usage: fit.py SCENE WORLD EP [--cma-iters 20] [--iters 200] [--frames F] [--tag TAG] [--init-from D0]
  stage A: native OptimizerCMA (12-dim zero-order search: global spring_Y, topology radii/neighbours, collision,
           drag, dashpot), native defaults (real.yaml init, sigma 1/6, seed 42, max_iter 20)
  stage B: native InvPhyTrainerWarp.train (per-spring log-stiffness + collision params, Adam, lr 1e-3, 200 it,
           best-loss checkpoint at vis_interval 20)
EP=D1 (after-calibration stage) is initialised from the D0 fit of the same world: CMA starts at the D0 optimum
(native mechanism: cfg.set_optimal_params), and the per-spring stiffness starts from the final D0 field (springs
matched by nearest midpoint if the topology changed) rescaled by the ratio of the D1/D0 CMA global stiffness.
Writes runs/fig1-v01/phystwin/<scene>/<EP>-<world><tag>/ : params.json, stageA/optimal_params.pkl, stageB/train/best_*.pth"""
import argparse, json, os, pickle, random, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scene', choices=['block', 'stem']); ap.add_argument('world'); ap.add_argument('ep', choices=['D0', 'D1'])
    ap.add_argument('--cma-iters', type=int, default=20); ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--frames', type=int, default=None); ap.add_argument('--tag', default='')
    ap.add_argument('--init-tag', default=None, help='tag of the D0 fit to initialise D1 from (default: same tag)')
    ap.add_argument('--vis-interval', type=int, default=20)
    a = ap.parse_args()
    t_all = time.time(); peak = ptw.GpuPeak()
    out = ptw.fit_dir(a.scene, a.ep, a.world, a.tag)
    import shutil
    for sub in ('stageA', 'stageB'):  # never pick up checkpoints of an earlier run
        shutil.rmtree(out/sub, ignore_errors=True)
    for f in ('params.json', 'springs.npz'):
        (out/f).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    geo = ptw.geometry(a.scene)
    data = ptw.dataset(a.scene, a.ep, a.world, geo, a.frames); F = len(data['object_points'])
    acc, audit = ptw.force_series(a.scene, a.ep, geo, F)
    with open(out/'data.pkl', 'wb') as f:
        pickle.dump(data, f)
    np.savez_compressed(out/'forces.npz', forces=acc.astype(np.float32))
    ptw.import_phystwin()
    import torch, warp as wp
    import qqtt.engine.trainer_warp as TM
    from qqtt import InvPhyTrainerWarp, OptimizerCMA
    from qqtt.utils import cfg, logger
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    cfg_dict = ptw.configure(cfg, a.scene, out/'forces.npz', dict(vis_interval=a.vis_interval, iterations=a.iters))
    init = None
    if a.ep == 'D1':
        d0 = ptw.fit_dir(a.scene, 'D0', a.world, a.tag if a.init_tag is None else a.init_tag)
        init = json.loads((d0/'params.json').read_text())
        with open(d0/'stageA/optimal_params.pkl', 'rb') as f:
            d0_opt = pickle.load(f)
        cfg.set_optimal_params(dict(d0_opt))  # CMA x_init := D0 optimum (native init path)
    logger.set_log_file(path=str(out), name='fit')

    # ---------------- stage A: native CMA-ES
    class CMA(OptimizerCMA):
        evals = []
        def error_func(self, parameters, visualize=False, video_path=None):
            t = time.time(); loss = super().error_func(parameters, visualize, video_path)
            self.evals.append(dict(x=[float(v) for v in parameters], loss=float(loss), seconds=time.time()-t, final=bool(visualize)))
            return loss
    tA = time.time(); (out/'stageA').mkdir(exist_ok=True)
    cwd = os.getcwd(); os.chdir(out/'stageA')
    opt = CMA(data_path=str(out/'data.pkl'), base_dir=str(out/'stageA'), train_frame=F)
    opt.optimize(max_iter=a.cma_iters)
    # diagnostic: share of the native acceleration regulariser (acc_weight) in the stage-A objective at x_init
    w_acc = cfg.acc_weight; cfg.acc_weight = 0.0; x0 = CMA.evals[0]['x']
    data_only = float(OptimizerCMA.error_func(opt, x0)); cfg.acc_weight = w_acc
    acc_share = 1-data_only/CMA.evals[0]['loss']
    os.chdir(cwd)
    with open(out/'stageA/optimal_params.pkl', 'rb') as f:
        optimal = pickle.load(f)
    del opt.simulator, opt
    secA = time.time()-tA
    init_loss_cma = CMA.evals[0]['loss']; best_cma = min(e['loss'] for e in CMA.evals if not e['final'])
    print('STAGE_A', json.dumps(dict(seconds=secA, evals=len(CMA.evals), init_loss=init_loss_cma, best=best_cma,
                                     optimal={k: float(v) for k, v in optimal.items()})), flush=True)

    # ---------------- stage B: native first-order per-spring training
    cfg.set_optimal_params(dict(optimal))
    hist = []

    class Trainer(InvPhyTrainerWarp):
        pred_log = []
        def visualize_sim(self, save_only=True, video_path=None, save_trajectory=False, save_path=None):
            xs = rollout(self.simulator, self.simulator.external_force_trajectory, self.controller_points, F)
            obs = data['object_points']; nm = geo['n_markers']
            err = np.linalg.norm(xs[:, :nm]-obs, axis=-1)
            self.pred_log.append(dict(video=str(video_path), marker_err_mm_mean=float(err.mean()*1e3), marker_err_mm_max=float(err.max()*1e3)))
            if save_path:
                with open(save_path, 'wb') as f:
                    pickle.dump(xs, f)

    def rollout(sim, forces, controls, n):
        sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
        xs = [wp.to_torch(sim.wp_init_vertices, requires_grad=False).cpu().numpy().copy()]
        for j in range(1, n):
            sim.set_controller_target(j, pure_inference=True); wp.capture_launch(sim.forward_graph)
            xs.append(wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).cpu().numpy().copy())
            sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
        return np.stack(xs)

    tB = time.time()
    tr = Trainer(str(out/'data.pkl'), str(out/'stageB'), train_frame=F)
    # wandb (disabled) rebinds wandb.log inside wandb.init -> capture the native per-iteration losses afterwards
    TM.wandb.Video = lambda *a_, **k_: None  # no preview videos are rendered
    TM.wandb.log = lambda d, step=None: hist.append(dict(step=step, **{k: float(v) for k, v in d.items() if isinstance(v, (int, float))}))
    sim = tr.simulator; ns_obj = tr.num_object_springs
    springs = sim.wp_springs.numpy(); V = tr.init_vertices.cpu().numpy()
    if init is not None:  # per-spring field of the D0 fit -> this topology
        ck0 = torch.load(init['stageB_checkpoint'], map_location='cpu')
        Y0 = ck0['spring_Y'].numpy(); mid0 = np.load(Path(init['stageB_checkpoint']).parent.parent.parent/'springs.npz')
        mid = V[springs].mean(1); kind = np.arange(len(springs)) >= ns_obj
        from scipy.spatial import cKDTree
        Yi = np.empty(len(springs))
        for k in (False, True):
            src = mid0['kind'] == k
            Yi[kind == k] = Y0[src][cKDTree(mid0['mid'][src]).query(mid[kind == k])[1]]
        Yi *= float(optimal['global_spring_Y'])/float(d0_opt['global_spring_Y'])
        sim.set_spring_Y(torch.log(torch.tensor(Yi, dtype=torch.float32, device=cfg.device)))
        init_info = dict(from_fit=str(init['out']), topology_identical=bool(len(Y0) == len(Yi) and
                         np.array_equal(springs, mid0['springs'])), rescale=float(optimal['global_spring_Y'])/float(d0_opt['global_spring_Y']))
    else:
        init_info = None
    np.savez_compressed(out/'springs.npz', springs=springs, mid=V[springs].mean(1), kind=np.arange(len(springs)) >= ns_obj,
                        rest=sim.wp_rest_lengths.numpy())
    tr.train()
    ckpts = list((out/'stageB/train').glob('best_*.pth')); assert len(ckpts) == 1, ckpts; ckpt = ckpts[0]
    secB = time.time()-tB
    ck = torch.load(ckpt, map_location='cpu'); Y = ck['spring_Y'].numpy()
    m0 = geo['m0']
    pk = peak.stop()
    info = dict(method='PhysTwin (official, upstream %s, isolated copy %s)' % (ptw.UPSTREAM_COMMIT, ptw.UP),
                scene=a.scene, world=a.world, episode=a.ep, tag=a.tag, out=str(out), frames=F,
                n_object_nodes=len(geo['x']), n_markers=geo['n_markers'], n_fixed_controllers=len(geo['ctrl']),
                spacing_m=geo['spacing'], node_mass_kg=m0, density_fixed=ptw.RHO, force_audit=audit,
                n_springs=int(len(springs)), n_object_springs=int(ns_obj),
                stageA=dict(seconds=secA, cma_iters=a.cma_iters, evaluations=len(CMA.evals), init_loss=init_loss_cma,
                            best_loss=best_cma, acc_regulariser_share_at_init=acc_share, optimal={k: float(v) for k, v in optimal.items()},
                            physical_global_spring_Y_N=float(optimal['global_spring_Y'])*m0),
                stageB=dict(seconds=secB, iterations=cfg.iterations, checkpoint=str(ckpt),
                            loss_history=hist, marker_rollout_log=Trainer.pred_log, init=init_info,
                            spring_Y_native=dict(min=float(Y.min()), median=float(np.median(Y)), max=float(Y.max()),
                                                 p05=float(np.percentile(Y, 5)), p95=float(np.percentile(Y, 95))),
                            spring_Y_physical_N_median=float(np.median(Y))*m0),
                stageB_checkpoint=str(ckpt), wall_seconds=time.time()-t_all, gpu_peak_mib=pk, **ptw.gpu_info(),
                config=cfg_dict)
    (out/'params.json').write_text(json.dumps(info, indent=1))
    print('FIT_COMPLETE', json.dumps({k: info[k] for k in ('scene', 'world', 'episode', 'wall_seconds', 'gpu_peak_mib')}), flush=True)


if __name__ == '__main__':
    main()
