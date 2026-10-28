"""Matched 91-frame, 30-epoch PhysTwin action comparison on fresh v07 objects."""
from pathlib import Path
import argparse,json,os,pickle,sys,time,hashlib
p=argparse.ArgumentParser();p.add_argument('--object',type=int,choices=list(range(6)),required=True);p.add_argument('--budget',choices=['repeat_initial','fixed','random','adaptive_pokes'],required=True);p.add_argument('--iterations',type=int,default=30);p.add_argument('--evaluate-only',action='store_true');p.add_argument('--step-refinement',type=int,default=1);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'runs/third_party/PhysTwin-v06';BASE=ROOT/'runs/phystwin-cost-aware-v07';OUT=BASE/'phystwin'/f'object-{a.object}-{a.budget}';OUT.mkdir(parents=True,exist_ok=True)
# Multiple resumable workers may meet at the same fit; serialize per output.
import fcntl
_run_lock=(OUT/'run.lock').open('a')
fcntl.flock(_run_lock,fcntl.LOCK_EX)
if (OUT/f'summary-dt{a.step_refinement}.json').exists():
 print('ALREADY_COMPLETE',OUT,a.step_refinement,flush=True);sys.exit(0)
os.environ['PHYSTWIN_HEADLESS']='1';os.environ['WANDB_MODE']='disabled';sys.path.insert(0,str(UP));sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import warp as wp
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg,logger
from scripts.phystwin import v06_physbridge as bridge
bridge.BASE=BASE
from scripts.phystwin.v06_physbridge import dataset,masses
from scripts.phystwin.sequential_fixture import points,ACTIONS
np.random.seed(42);torch.manual_seed(42)
actions=['initial']
if a.budget=='repeat_initial':actions+=['initial','initial']
else:
 chosen=json.loads((BASE/'plans'/f'test-{a.budget}.json').read_text())[a.object*3]['history']
 assert len(chosen)==2 and all(i<6 for i in chosen)
 actions += [ACTIONS[int(i)+1] for i in chosen]
train,load,truth,hold=dataset(a.object,actions)
with (OUT/'train.pkl').open('wb') as f:pickle.dump(train,f)
np.savez_compressed(OUT/'forces.npz',forces=load)
cfg.load_from_yaml(str(UP/'configs/real.yaml'));cfg.FPS=5;cfg.dt=.0002/a.step_refinement;cfg.num_substeps=1000*a.step_refinement;cfg.gravity_scale=0.;cfg.reverse_z=False
cfg.external_force_path=str(OUT/'forces.npz');cfg.base_lr=.01;cfg.iterations=a.iterations;cfg.vis_interval=1000;cfg.init_spring_Y=.1
cfg.object_radius=.07;cfg.object_max_neighbours=30;cfg.controller_radius=.06;cfg.controller_max_neighbours=16;cfg.spring_Y_min=.0001;cfg.spring_Y_max=20.;cfg.dashpot_damping=.01;cfg.drag_damping=3.;cfg.collision_learn=False;cfg.self_collision=False
logger.set_log_file(path=str(OUT),name='fit');rest=points()
class Trainer(InvPhyTrainerWarp):
 def _init_start(self,*args,**kwargs):
  vertices,springs,lengths,mass,count=super()._init_start(*args,**kwargs);mass[:len(rest)]=torch.tensor(masses(),dtype=torch.float32,device=cfg.device);return vertices,springs,lengths,mass,count
 def rollout(self,forces,controls):
  sim=self.simulator;old=sim.external_force_trajectory;oldcontrol=sim.controller_points;sim.external_force_trajectory=torch.tensor(forces,dtype=torch.float32,device=cfg.device);sim.controller_points=torch.tensor(controls,dtype=torch.float32,device=cfg.device)
  sim.set_init_state(sim.wp_init_vertices,sim.wp_init_velocities);states=[rest.copy()]
  for i in range(1,len(forces)):
   sim.set_controller_target(i,pure_inference=True);wp.capture_launch(sim.forward_graph);states.append(wp.to_torch(sim.wp_states[-1].wp_x,requires_grad=False).cpu().clone().numpy());sim.set_init_state(sim.wp_states[-1].wp_x,sim.wp_states[-1].wp_v)
  sim.external_force_trajectory=old;sim.controller_points=oldcontrol;return np.stack(states)
 def visualize_sim(self,save_only=True,video_path=None,save_trajectory=False,save_path=None):
  if video_path is None:return
  import cv2
  pred=self.rollout(load,train['controller_points']);Path(video_path).parent.mkdir(parents=True,exist_ok=True)
  writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),5,(1000,440));assert writer.isOpened()
  for frame,(actual,estimated) in enumerate(zip(truth,pred)):
   image=np.full((440,1000,3),245,np.uint8)
   for offset,positions,color,label in [(0,actual,(40,150,40),'Quasistatic reference'),(500,estimated,(190,75,30),'PhysTwin')]:
    pixels=np.c_[70+positions[:,0]*1400+positions[:,1]*400,240-positions[:,2]*1800-positions[:,1]*800].astype(int)
    for x,y in pixels:
     if 0<x<490 and 50<y<400:cv2.circle(image,(x+offset,y),3,color,-1)
    cv2.putText(image,label,(offset+15,30),0,.65,(0,0,0),1)
   cv2.putText(image,f'Training interactions | object {a.object} | t={frame/5:.1f}s | scale x1',(15,425),0,.55,(0,0,0),1);writer.write(image)
  writer.release()
started=time.time();trainer=Trainer(str(OUT/'train.pkl'),str(OUT),train_frame=len(train['object_points']),pure_inference_mode=a.evaluate_only)
if not a.evaluate_only:trainer.train()
checkpoint=next((OUT/'train').glob('best_*.pth'));trainer.test(str(checkpoint));rows={}
for name in ['initial','held_oblique']:
 data,forces,reference,hold=dataset(a.object,[name],observe=False);pred=trainer.rollout(forces,data['controller_points']);assert np.isfinite(pred).all()
 rows[name]=dict(hold_rmse_mm=float(1000*np.sqrt(np.mean((pred[hold]-reference[hold])**2))),stationary_rmse_mm=float(1000*np.sqrt(np.mean((rest-reference[hold])**2))),hold_drift_mm=float(1000*np.sqrt(np.mean((pred[hold[-1]]-pred[hold[0]])**2))))
 np.savez_compressed(OUT/f'{name}-dt{a.step_refinement}.npz',predicted=pred,reference=reference,times=np.arange(len(pred))/5,hold=hold)
summary=dict(object=a.object,budget=a.budget,fit_actions=actions,iterations=0 if a.evaluate_only else a.iterations,results=rows,elapsed_seconds=time.time()-started,checkpoint=str(checkpoint),step_refinement=a.step_refinement,training_frames=len(train['object_points']), optimizer_updates=(len(train['object_points'])-1)*a.iterations, visible_markers=int(train['object_visibilities'][0].sum()), seed=42, scope='Matched three interaction cycles and 30 epochs for each policy; identical 27 visible markers and optimizer settings. Repeat-initial control repeats the same load three times. Policy uses sparse observations, fitter gets the same richer observations across policies. Six fresh objects, one training seed. Official PhysTwin dense fitter and spring physics with opt-in prescribed-force kernel, known rest geometry/masses/cameras, tracked synthetic RGB-D fiducials. Candidate anchors cover both end faces; true patch labels/stiffnesses are not supplied. 27 visible markers per frame, matched across PhysTwin action policies, but not a head-to-head model comparison with sparse Bayesian inference. Ramp targets interpolate static FEM endpoints; scores use holds, not a validated dynamics trajectory.')
(OUT/f'summary-dt{a.step_refinement}.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE',json.dumps(summary),flush=True)
