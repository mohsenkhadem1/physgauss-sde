"""Controlled PhysTwin transfer gate, known geometry and controller motion.

Uses the upstream dense training loop; adapters are declared in protocol.json.
"""
import argparse,hashlib,json,os,pickle,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--object',required=True,choices=['soft_firm','stiff_loose']);p.add_argument('--iterations',type=int,default=80)
p.add_argument('--evaluate-only',action='store_true');p.add_argument('--step-refinement',type=int,default=1)
a=p.parse_args();root=Path(__file__).resolve().parents[2];base=root/'runs/phystwin-transfer-v01';up=root/'runs/third_party/PhysTwin'
assert a.step_refinement in (1,2)
out=base/(a.object+(f'-dt{a.step_refinement}' if a.evaluate_only else ''));out.mkdir(parents=True,exist_ok=True)
assert 'getattr(cfg, "gravity_scale", 1.0)' in (up/'qqtt/model/diff_simulator/spring_mass_warp.py').read_text(), 'Apply controlled-physics.patch first'
assert json.loads((base/'reference/qualification.json').read_text())['passed'],'Reference qualification failed'
os.environ['PHYSTWIN_HEADLESS']='1';os.environ['WANDB_MODE']='disabled';sys.path.insert(0,str(up));sys.path.insert(0,str(root))
import numpy as np
import torch
import warp as wp
import cv2
import imageio_ffmpeg
import subprocess
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg,logger
from scripts.phystwin.controlled_reference import ACTIONS
from experiments.boundary_block import lumped_masses
np.random.seed(42);torch.manual_seed(42)
z=np.load(base/'reference'/f'{a.object}-axial-n2.npz');rest=z['rest'];mass=z['mass'];N=len(rest)
left=np.isclose(rest[:,0],0);right=np.isclose(rest[:,0],.24)
times=np.arange(61)/10;alpha=np.where(times<3,.5*(1-np.cos(np.pi*times/3)),1.)
controls=np.concatenate([rest[left]+[-.02,0,0],rest[right]+[.02,0,0]])

def dataset(action):
    z=np.load(base/'reference'/f'{a.object}-{action}-n8.npz')
    fields=z['sampled_positions'];idx=np.minimum((alpha*8).astype(int),7);fraction=alpha*8-idx
    points=(1-fraction[:,None,None])*fields[idx]+fraction[:,None,None]*fields[idx+1]
    targets=np.broadcast_to(controls,(61,len(controls),3)).copy()
    targets[:,left.sum():]+=alpha[:,None,None]*ACTIONS[action]
    return dict(object_points=points.astype('float32'),object_colors=np.ones((61,N,3),dtype='float32')*.5,
        object_visibilities=np.broadcast_to(right,(61,N)).copy(),object_motions_valid=np.broadcast_to(right,(60,N)).copy(),
        controller_points=targets.astype('float32'),surface_points=np.empty((0,3),dtype='float32'),interior_points=np.empty((0,3),dtype='float32'))
train_data=dataset('axial')
with (out/'train-input.pkl').open('wb') as f:pickle.dump(train_data,f)
cfg.load_from_yaml(str(up/'configs/real.yaml'))
cfg.FPS=10;cfg.dt=.0001/a.step_refinement;cfg.num_substeps=1000*a.step_refinement;cfg.gravity_scale=0.;cfg.reverse_z=False
cfg.base_lr=.01;cfg.iterations=a.iterations;cfg.vis_interval=20;cfg.init_spring_Y=.5
cfg.object_radius=.07;cfg.object_max_neighbours=30;cfg.controller_radius=.06;cfg.controller_max_neighbours=16
cfg.spring_Y_min=.0001;cfg.spring_Y_max=20.;cfg.dashpot_damping=.01;cfg.drag_damping=3.;cfg.collision_learn=False;cfg.self_collision=False
logger.set_log_file(path=str(out),name='fit')

class ControlledTrainer(InvPhyTrainerWarp):
    def _init_start(self,*args,**kwargs):
        vertices,springs,lengths,masses,count=super()._init_start(*args,**kwargs)
        masses[:N]=torch.tensor(mass,dtype=torch.float32,device=cfg.device)
        return vertices,springs,lengths,masses,count
    def rollout(self,control):
        sim=self.simulator;old=sim.controller_points
        sim.controller_points=torch.tensor(control,dtype=torch.float32,device=cfg.device)
        sim.set_init_state(sim.wp_init_vertices,sim.wp_init_velocities)
        states=[rest.copy()]
        for j in range(1,len(control)):
            sim.set_controller_target(j,pure_inference=True);wp.capture_launch(sim.forward_graph)
            states.append(wp.to_torch(sim.wp_states[-1].wp_x,requires_grad=False).cpu().clone().numpy())
            sim.set_init_state(sim.wp_states[-1].wp_x,sim.wp_states[-1].wp_v)
        sim.controller_points=old
        return np.stack(states)
    def video(self,vertices,truth,path,label):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(1000,440))
        assert writer.isOpened()
        # Oblique fixed projection makes y and z motion visible; same scale in both panels.
        def project(X):return np.c_[70+X[:,0]*1400+X[:,1]*400,240-X[:,2]*1800-X[:,1]*800].astype(int)
        for t in range(len(vertices)):
            img=np.full((440,1000,3),245,np.uint8)
            for off,pts,color,title in [(0,truth[t],(40,150,40),'Independent FEM reference'),(500,vertices[t],(190,75,30),'PhysTwin prediction')]:
                for x,y in project(pts):
                    if 0<x<490 and 50<y<400:cv2.circle(img,(x+off,y),3,color,-1)
                cv2.putText(img,title,(off+15,30),0,.65,(0,0,0),1)
            cv2.putText(img,f'{a.object} | {label} | t={t/10:.1f}s | fixed scale, no amplification',(15,425),0,.55,(0,0,0),1)
            writer.write(img)
        writer.release();tmp=Path(path).with_suffix('.h264.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(path),'-c:v','libx264','-pix_fmt','yuv420p',str(tmp)],check=True);tmp.replace(path)
    def visualize_sim(self,save_only=True,video_path=None,save_trajectory=False,save_path=None):
        vertices=self.rollout(train_data['controller_points'])
        assert np.isfinite(vertices).all()
        if video_path:self.video(vertices,train_data['object_points'],video_path,'TRAIN axial')
        if save_path:
            with open(save_path,'wb') as f:pickle.dump(vertices,f)

start=time.time();trainer=ControlledTrainer(str(out/'train-input.pkl'),str(out),train_frame=61,pure_inference_mode=a.evaluate_only)
if not a.evaluate_only:trainer.train()
checkpoint=next((base/a.object/'train').glob('best_*.pth'));trainer.test(str(checkpoint))
results={}
# Prediction receives only controller trajectories. Held object positions enter scoring below.
for action in ACTIONS:
    d=dataset(action);pred=trainer.rollout(d['controller_points']);truth=d['object_points']
    assert np.isfinite(pred).all()
    steady=pred[-10:]-truth[-10:]
    error=float(np.sqrt(np.mean(steady**2)));stationary=float(np.sqrt(np.mean((rest-truth[-10:])**2)))
    drift=float(np.sqrt(np.mean((pred[-1]-pred[-10])**2)))
    visible_error=float(np.sqrt(np.mean((pred[-10:,right]-truth[-10:,right])**2)))
    results[action]=dict(final_hold_coordinate_rmse_m=error,observed_region_rmse_m=visible_error,
        stationary_rmse_m=stationary,hold_drift_m=drift,passed=bool(error<.0015 and error<stationary))
    np.savez_compressed(out/f'{action}.npz',predicted=pred,reference=truth,controls=d['controller_points'],times=times)
    trainer.video(pred,truth,out/f'{action}.mp4','TRAIN' if action=='axial' else 'HELD INTERACTION '+action)
summary=dict(object=a.object,results=results,gate_passed=all(results[k]['passed'] for k in ['sideways','oblique']),
    iterations=0 if a.evaluate_only else a.iterations,step_refinement=a.step_refinement,elapsed_seconds=time.time()-start,checkpoint=str(checkpoint),
    physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),particle_count=N,spring_count=trainer.simulator.n_springs,
    scope='Known geometry and masses, no gravity/contact, partial exact axial observations, supplied held controller motion, original PhysTwin dense optimizer. Quasi-static FEM reference versus dynamic spring model; hold drift reported explicitly.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE',json.dumps(summary),flush=True)
