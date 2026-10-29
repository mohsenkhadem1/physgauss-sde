"""Official PhysTwin spring-mass dynamics, fitted to shared settled positions.

No interpolated deformation targets and no force labels. Native simulator and
per-spring Adam updates; only the observation schedule is adapted to this task.
"""
from pathlib import Path
import argparse,fcntl,json,os,pickle,sys,time
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'runs/third_party/PhysTwin-v06'
os.environ['PHYSTWIN_HEADLESS']='1';os.environ['WANDB_MODE']='disabled'
sys.path.insert(0,str(UP));sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg,logger
from scripts.position_ood.study import BASE,within_deadline
from scripts.position_ood.observations import make
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.v08_physbridge import masses
from scripts.phystwin.sequential_reference import write


SHIFT=np.array([0.,0.,.3])

def controller(pose):
    rest=points();ends=rest[np.isclose(rest[:,0],0)|np.isclose(rest[:,0],.24)].copy()
    ends[:,0]+=np.where(ends[:,0]<.12,-.02,.02)
    out=ends.copy()
    for side in range(2):
        ids=(ends[:,0]<.12) if side==0 else (ends[:,0]>.12);center=np.array([.24*side,0,0]);R=Rotation.from_rotvec(pose[side,3:]).as_matrix()
        out[ids]=(ends[ids]-center)@R.T+center+pose[side,:3]
    return out


def schedule(poses,measurements=None,visible_ids=None,frames=25):
    rest=points();xyz=[rest.copy()];ctrl=[controller(np.zeros((2,6)))];mask=[np.zeros(81,bool)];ends=[];starts=[]
    for k,pose in enumerate(poses):
        starts.append(len(xyz))
        for j in range(1,frames+1):
            ctrl.append(controller(pose*min(j/5.,1.)));row=rest.copy();m=np.zeros(81,bool)
            if measurements is not None and j>=frames-4:
                row[visible_ids]+=measurements[k];m[visible_ids]=True
            xyz.append(row);mask.append(m)
        ends.append(len(xyz)-1)
    count=len(xyz)
    return dict(object_points=np.array(xyz,dtype='float32')+SHIFT.astype('float32'),object_colors=np.full((count,81,3),.5,dtype='float32'),object_visibilities=np.array(mask),object_motions_valid=np.array(mask[1:]),controller_points=np.array(ctrl,dtype='float32')+SHIFT.astype('float32'),surface_points=np.empty((0,3),dtype='float32'),interior_points=np.empty((0,3),dtype='float32')),starts,ends


class Trainer(InvPhyTrainerWarp):
    def _init_start(self,*args,**kwargs):
        vertices,springs,lengths,mass,count=super()._init_start(*args,**kwargs);mass[:81]=torch.tensor(masses(),dtype=torch.float32,device=cfg.device);return vertices,springs,lengths,mass,count
    def visualize_sim(self,*args,**kwargs):pass
    def capture_observed_loss(self):
        # Counts are host constants captured by CUDA graphs. Every acquired view
        # has 27 markers; transient frames have zero masks and are forward-only.
        sim=self.simulator;sim.num_valid_visibilities=27;sim.num_valid_motions=27
        with wp.ScopedCapture() as capture:
            sim.tape=wp.Tape()
            with sim.tape:sim.step();sim.calculate_loss()
            sim.tape.backward(sim.loss)
        sim.graph=capture.graph;sim.tape.zero();sim.clear_loss()
    def rollout(self,data,starts,ends,training=False):
        sim=self.simulator;sim.controller_points=torch.tensor(data['controller_points'],device=cfg.device)
        states=[];drift=[];total=0.;updates=0;previous=None
        for j in range(1,len(data['object_points'])):
            if j in starts:sim.set_init_state(sim.wp_init_vertices,sim.wp_init_velocities)
            sim.set_controller_target(j,pure_inference=not training)
            if j in starts:wp.copy(sim.wp_original_control_point,wp.from_numpy((controller(np.zeros((2,6)))+SHIFT).astype('float32'),dtype=wp.vec3,device=cfg.device))
            observed=training and bool(data['object_visibilities'][j].any())
            wp.capture_launch(sim.graph if observed else sim.forward_graph)
            if observed:
                value=float(wp.to_torch(sim.loss,requires_grad=False).item());total+=value
                if not np.isfinite(value):raise RuntimeError('Nonfinite PhysTwin training loss')
                self.optimizer.step()
                if getattr(self,'bulk_regularization',0.):
                    # Our benchmark adapter regularizes bulk springs in three
                    # material-coordinate bins; controller springs remain free.
                    with torch.no_grad():
                        weights=wp.to_torch(sim.wp_spring_Y)
                        for ids in self.bulk_groups:
                            if ids.numel():weights[ids]=(1-self.bulk_regularization)*weights[ids]+self.bulk_regularization*weights[ids].mean()
                updates+=1;sim.tape.zero();sim.clear_loss()
            if j in ends or j+1 in ends:
                state=wp.to_torch(sim.wp_states[-1].wp_x,requires_grad=False).cpu().clone().numpy()
                if j in ends:states.append(state);drift.append(float(np.sqrt(np.mean((state-previous)**2))))
                previous=state
            sim.set_init_state(sim.wp_states[-1].wp_x,sim.wp_states[-1].wp_v)
        return np.array(states),np.array(drift),total/max(updates,1),updates


def run(a):
    if ('test' in a.split.split('_')):
        from scripts.position_ood.seal import verify
        verify()
    within_deadline();obs=np.load(make(a.split,a.object));prefix=[0,1,2];destination=BASE/f'phystwin/{a.split}-{a.object}/y{a.initial:g}-dt{a.refinement}'
    if a.bulk_regularization:destination=destination.with_name(destination.name+f'-bulk{a.bulk_regularization:g}')
    destination.mkdir(parents=True,exist_ok=True);lock=(destination/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (destination/'summary.json').exists():return
    train,starts,ends=schedule(obs['poses'][prefix],obs['measurements'][prefix],obs['visible_ids'],a.frames)
    with (destination/'train.pkl').open('wb') as f:pickle.dump(train,f)
    cfg.load_from_yaml(str(UP/'configs/real.yaml'));cfg.FPS=5;cfg.dt=.0002/a.refinement;cfg.num_substeps=1000*a.refinement;cfg.gravity_scale=9.81/9.8;cfg.reverse_z=False;cfg.external_force_path=None
    cfg.base_lr=.02;cfg.init_spring_Y=a.initial;cfg.object_radius=.07;cfg.object_max_neighbours=30;cfg.controller_radius=.06;cfg.controller_max_neighbours=16;cfg.spring_Y_min=.001;cfg.spring_Y_max=300.;cfg.dashpot_damping=.1;cfg.drag_damping=5.;cfg.collision_learn=False;cfg.self_collision=False;cfg.acc_weight=0.;cfg.chamfer_weight=1.;cfg.track_weight=1.;cfg.use_graph=True
    np.random.seed(42);torch.manual_seed(42);logger.set_log_file(path=str(destination),name='fit');trainer=Trainer(str(destination/'train.pkl'),str(destination),train_frame=len(train['object_points']));trainer.capture_observed_loss();sim=trainer.simulator
    trainer.bulk_regularization=a.bulk_regularization
    if a.bulk_regularization:
        edges=wp.to_torch(sim.wp_springs).detach().cpu().numpy();bulk=np.all(edges<81,axis=1);midpoint=points()[np.minimum(edges,80),0].mean(1);groups=np.clip((midpoint/.08).astype(int),0,2);trainer.bulk_groups=[torch.tensor(np.flatnonzero(bulk&(groups==k)),device=cfg.device) for k in range(3)]
    history=[];best=np.inf;best_weights=None;started=time.time();updates=0
    for epoch in range(a.epochs):
        within_deadline();_,_,loss,count=trainer.rollout(train,starts,ends,True);updates+=count
        # Select using a fresh forward rollout of the acquired observations only.
        pred,drift,_,_=trainer.rollout(train,starts,ends);mse=float(np.mean((pred[:,obs['visible_ids']]-SHIFT-points()[obs['visible_ids']]-obs['measurements'][prefix])**2))
        history.append(dict(epoch=epoch,loss=loss,observed_mse=mse,drift_m=drift.tolist()))
        if mse<best:
            best=mse;best_weights=wp.to_torch(sim.wp_spring_Y,requires_grad=False).detach().clone();torch.save(best_weights,destination/'best.pt')
        write(destination/'progress.json',dict(epoch=epoch,best_observed_rmse_mm=1000*np.sqrt(best),history=history));print('PHYSTWIN_EPOCH',a.split,a.object,a.initial,epoch,1000*np.sqrt(mse),flush=True)
    with torch.no_grad():wp.to_torch(sim.wp_spring_Y).copy_(best_weights)
    data,starts,ends=schedule(obs['poses'],frames=a.frames);pred,drift,_,_=trainer.rollout(data,starts,ends)
    assert np.isfinite(pred).all();np.savez_compressed(destination/'predictions.npz',predicted=pred-SHIFT-points(),drift_m=drift,poses=obs['poses'])
    write(destination/'summary.json',dict(state='complete',split=a.split,object=a.object,initial_Y=a.initial,bulk_regularization=a.bulk_regularization,epochs=a.epochs,updates=updates,frames=a.frames,dt=cfg.dt,best_observed_rmse_mm=1000*np.sqrt(best),max_hold_drift_mm=1000*float(max(drift)),elapsed_seconds=time.time()-started,history=history,scope='Official PhysTwin nonlinear spring-mass core and per-spring Adam; prescribed tool positions, nominal gravity, known rest shape and mass; same three noisy settled RGB-D observations, repeated in final one-second hold. No transient deformation labels or true force inputs. Dynamics used to approximate static endpoints.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--object',type=int,default=0);p.add_argument('--initial',type=float,default=3.);p.add_argument('--epochs',type=int,default=80);p.add_argument('--frames',type=int,default=25);p.add_argument('--refinement',type=int,default=1);p.add_argument('--bulk-regularization',type=float,default=0.);a=p.parse_args();run(a)
