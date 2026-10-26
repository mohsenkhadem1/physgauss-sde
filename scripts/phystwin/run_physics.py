"""Run official PhysTwin physics with CPU point-cloud previews and explicit split metrics.

No physics kernels, topology, loss, optimizer, or time step are replaced.
This first integration uses author-processed observations and CMA initialization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time


def metrics(vertices, data, split, manual):
    import numpy as np
    from scipy.spatial import cKDTree
    surface_n = data['object_points'].shape[1] + len(data['surface_points'])
    valid0 = ~np.isnan(manual[0]).any(axis=1)
    _, correspondence = cKDTree(vertices[0]).query(manual[0, valid0])
    out = {}
    for label, start, end in [('train', 1, split['train'][1]),
                              ('test', split['train'][1], split['test'][1])]:
        track, chamfer = [], []
        for t in range(start, end):
            target = manual[t, valid0]
            valid = np.isfinite(target).all(axis=1)
            if valid.any():
                track.append(float(np.linalg.norm(vertices[t, correspondence[valid]]-target[valid], axis=1).mean()))
            observed = data['object_points'][t, data['object_visibilities'][t].astype(bool)]
            if len(observed):
                # Match official single-direction Chamfer with norm=1, not squared L2.
                dist, _ = cKDTree(vertices[t, :surface_n]).query(observed, p=1)
                chamfer.append(float(dist.mean()))
        out[label] = {'manual_track_mean_m': float(np.mean(track)),
                      'chamfer_l1_mean_m': float(np.mean(chamfer)),
                      'track_frames': len(track), 'chamfer_frames': len(chamfer)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('runs/third_party/PhysTwin'))
    p.add_argument('--case', default='single_lift_sloth')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['reference','train','initialization'], default='reference')
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--initialization', type=Path,
                   help='CMA optimal_params.pkl; defaults to the authors\' initialization')
    a = p.parse_args()
    root, out = a.root.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['PHYSTWIN_HEADLESS'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ.setdefault('OMP_NUM_THREADS','4')
    os.environ.setdefault('MPLBACKEND','Agg')
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    import warp as wp
    import cv2
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import cfg, logger
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data_dir = root/'data/different_types'/a.case
    def read(name):
        with (data_dir/name).open('rb') as f: return pickle.load(f)
    data = read('final_data.pkl')
    split = json.loads((data_dir/'split.json').read_text())
    manual = read('gt_track_3d.pkl')
    cfg.load_from_yaml(str(root/'configs'/('cloth.yaml' if 'cloth' in a.case or 'package' in a.case else 'real.yaml')))
    initialization = (a.initialization.resolve() if a.initialization else
                      root/'experiments_optimization'/a.case/'optimal_params.pkl')
    with initialization.open('rb') as f:
        cfg.set_optimal_params(pickle.load(f))
    cfg.iterations = a.iterations
    cfg.vis_interval = 20
    cfg.c2ws = np.asarray(read('calibrate.pkl'))
    cfg.w2cs = np.linalg.inv(cfg.c2ws)
    meta = json.loads((data_dir/'metadata.json').read_text())
    cfg.intrinsics = np.asarray(meta['intrinsics']); cfg.WH = meta['WH']
    cfg.overlay_path = None
    logger.set_log_file(path=str(out), name='physics')
    # Fixed camera and shared axis limits: no deformation exaggeration or framewise scaling.
    all_points = data['object_points'].reshape(-1,3)
    center = np.median(all_points,axis=0)
    _,_,axes = np.linalg.svd(all_points[::20]-center,full_matrices=False)
    basis = axes[:2].T
    projected = (all_points-center)@basis
    lo,hi = np.percentile(projected,[0,100],axis=0)
    scale = 430/max(hi-lo)
    midpoint = (hi+lo)/2
    def draw(points):
        xy = ((points-center)@basis-midpoint)*scale
        return np.column_stack([xy[:,0]+260, 285-xy[:,1]]).astype(int)
    def rollout(self, save_only=True, video_path=None, save_trajectory=False, save_path=None):
        sim = self.simulator
        sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
        verts = [wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).cpu().clone().numpy()]
        for t in range(1,self.dataset.frame_len):
            sim.set_controller_target(t,pure_inference=True)
            if sim.object_collision_flag: sim.update_collision_graph()
            if cfg.use_graph: wp.capture_launch(sim.forward_graph)
            else: sim.step()
            verts.append(wp.to_torch(sim.wp_states[-1].wp_x,requires_grad=False).cpu().clone().numpy())
            sim.set_init_state(sim.wp_states[-1].wp_x,sim.wp_states[-1].wp_v)
        vertices = np.stack(verts)
        if not np.isfinite(vertices).all(): raise RuntimeError('Non-finite PhysTwin rollout')
        if save_path:
            with open(save_path,'wb') as f: pickle.dump(vertices,f)
        if video_path:
            path = Path(video_path); path.parent.mkdir(parents=True,exist_ok=True)
            writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),cfg.FPS,(1040,560))
            if not writer.isOpened(): raise RuntimeError('Video writer failed')
            for t in range(len(vertices)):
                image = np.full((560,1040,3),245,np.uint8)
                visible = data['object_visibilities'][t].astype(bool)
                for offset, points, color in [(0,data['object_points'][t,visible],(30,145,40)),
                        (520,vertices[t,:self.num_surface_points],(200,85,30))]:
                    for x,y in draw(points):
                        if 0 <= x < 520 and 50 <= y < 530: cv2.circle(image,(x+offset,y),2,color,-1)
                for x,y in draw(data['controller_points'][t]):
                    for off in (0,520): cv2.circle(image,(x+off,y),4,(30,30,220),-1)
                state = 'TRAIN' if t < split['train'][1] else 'HELD-OUT FUTURE'
                cv2.putText(image,'Observed visible surface',(15,30),0,.65,(0,0,0),1)
                cv2.putText(image,'PhysTwin predicted surface',(535,30),0,.65,(0,0,0),1)
                cv2.putText(image,f'{state} | frame {t} | fixed projection; red = prescribed controls',(20,548),0,.55,(0,0,0),1)
                writer.write(image)
            writer.release()
            # Browser-compatible H.264 while keeping the renderer independent of CUDA extensions.
            import imageio_ffmpeg
            tmp = path.with_suffix('.h264.mp4')
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(path),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],check=True)
            tmp.replace(path)
        label = Path(video_path).stem if video_path else 'rollout'
        result = metrics(vertices,data,split,manual)
        (out/f'{label}-metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        print('METRICS',label,json.dumps(result),flush=True)
        self.last_vertices = vertices
    InvPhyTrainerWarp.visualize_sim = rollout
    start = time.time()
    trainer = InvPhyTrainerWarp(data_path=str(data_dir/'final_data.pkl'),base_dir=str(out),
        train_frame=split['train'][1],pure_inference_mode=a.mode!='train')
    if a.mode == 'train':
        trainer.train()
        checkpoint = next((out/'train').glob('best_*.pth'))
    elif a.mode == 'reference':
        checkpoint = next((root/'reference/experiments'/a.case/'train').glob('best_*.pth'))
    else:
        checkpoint = None
    trainer.test(str(checkpoint) if checkpoint else None)
    reference_path = root/'reference/experiments'/a.case/'inference.pkl'
    with reference_path.open('rb') as f: reference = pickle.load(f)
    frozen = np.broadcast_to(trainer.last_vertices[0],trainer.last_vertices.shape)
    result = dict(case=a.case, mode=a.mode, split=split, elapsed_seconds=time.time()-start,
        initialization=str(initialization),
        initialization_sha256=hashlib.sha256(initialization.read_bytes()).hexdigest(),
        upstream_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        headless_patch_sha256=hashlib.sha256(subprocess.check_output(['git','-C',str(root),'diff'])).hexdigest(),
        checkpoint=str(checkpoint) if checkpoint else None,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
        device=torch.cuda.get_device_name(0),torch=torch.__version__,warp=wp.__version__,
        frame_count=len(trainer.last_vertices), particle_count=trainer.num_all_points,
        spring_count=trainer.simulator.n_springs, iterations=a.iterations if a.mode=='train' else 0,
        result=metrics(trainer.last_vertices,data,split,manual),
        official_saved_trajectory=metrics(reference,data,split,manual),
        frozen_first_frame=metrics(frozen,data,split,manual),
        reference_trajectory_rmse_m=float(np.sqrt(np.mean((trainer.last_vertices-reference)**2))),
        scope='Official processed observations; CMA initialization recorded separately. No raw RGB-D preprocessing or Gaussian appearance training. Future control trajectories are supplied. Test metrics are diagnostic only, never used for optimization or checkpoint selection.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',json.dumps(result),flush=True)

if __name__ == '__main__': main()
