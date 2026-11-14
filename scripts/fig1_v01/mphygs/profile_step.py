"""Time one simulated frame (forward and forward+backward) of the native simulator on the block particle set.
Usage: profile_step.py [--fast-timers] [--max-stiffness S] [--frames F]"""
import argparse, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser(); ap.add_argument('--fast-timers', action='store_true')
ap.add_argument('--max-stiffness', type=float, default=None); ap.add_argument('--frames', type=int, default=1)
a = ap.parse_args()
import mphygs_lib as L
import common as C
if a.fast_timers:
    L.disable_scoped_timers()
dev = 'cuda:0'; root = C.OUT/'block/D0-Bcentre'
cfg = json.loads((root/'config.json').read_text())
if a.max_stiffness:
    cfg['max_stiffness'] = a.max_stiffness
inp = L.load_inputs(root, 'block_D0_Bcentre', dev)
mf = L.make_material(cfg, inp['onehot'].shape[1], dev); mfunc = L.make_material_func(mf, inp)
aabbs, diffs, size = L.sim_aabbs_from(inp['tracks'][:a.frames])
sim = L.make_simulator(cfg, inp['x'], aabbs[0], size, dev)
sim.set_force(inp['ad']['f_full'], float(inp['ad']['t_ramp']), cfg['frame_timestep'])
x0 = inp['x']; m = inp['modifier_mask'].float()
fl = L.FrameList(list(aabbs))
res = {}
for mode in ('fwd', 'fwd+bwd', 'fwd+bwd'):
    torch.cuda.synchronize(); t0 = time.time()
    mv = mfunc()
    st = [x0, torch.zeros_like(x0), torch.eye(3, device=dev)[None].repeat(len(x0), 1, 1), torch.zeros((len(x0), 3, 3), device=dev)]
    loss = 0
    with torch.set_grad_enabled(mode != 'fwd'):
        for t in range(a.frames):
            st = list(sim.forward(*st, **mv, dt=cfg['frame_timestep'], particle_modifier_mask=m, particle_modifier_x0=x0,
                                  particle_modifier_x1=x0, init_particle_acc=torch.zeros_like(x0), sim_aabb=fl[t],
                                  num_checkpoints_per_frame=10))
            loss = loss+((st[0]-inp['tracks'][t])**2).sum(-1).mean()
        if mode != 'fwd':
            loss.backward()
    torch.cuda.synchronize(); res[mode] = time.time()-t0
print(json.dumps(dict(fast_timers=a.fast_timers, max_stiffness=cfg['max_stiffness'], frames=a.frames,
                      substeps_per_frame=sim.last_num_substeps, seconds=res, peak_gb=torch.cuda.max_memory_allocated()/1e9,
                      grad_norm=float(sum(p.grad.norm()**2 for p in mf.parameters() if p.grad is not None)**.5))))
