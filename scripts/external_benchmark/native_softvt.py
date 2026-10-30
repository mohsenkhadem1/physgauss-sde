"""Official PhysTwin physical core on SoftVTBench's positional observation interface.

Native spring-mass dynamics, losses, per-spring optimizer. Adapter supplies initial
geometry, sparse observed prefix, and positional gripper constraints; no force or
material labels. This is an adapted baseline, not an official SoftVTBench entry.
"""
import os,sys,argparse,time,pickle
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'runs/third_party/PhysTwin'
os.environ['PHYSTWIN_HEADLESS']='1';os.environ['WANDB_MODE']='disabled'
sys.path.insert(0,str(UP));sys.path.insert(0,str(ROOT))
import numpy as np,torch,warp as wp
import wandb
# Disable the upstream training-video logging hook in this headless adapter.
# Quantitative predictions and genuine videos are generated separately.
wandb.Video=lambda *args,**kwargs:None
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg,logger
from scripts.external_benchmark.common import BASE,write_json,check_deadline,sha256,save_provenance
from scripts.external_benchmark.data import softvt,fps

def make_native(data,fit_end):
    rest=data['nodes'];T=len(data['times']);N=len(rest)
    xyz=np.broadcast_to(rest,(T,N,3)).copy();mask=np.zeros((T,N),bool)
    observed_ids=np.argmax(data['observation_matrix'],axis=1)
    xyz[:fit_end,observed_ids]=data['observed'][:fit_end]
    mask[:fit_end,observed_ids]=data['observed_mask'][:fit_end]
    controls=[]
    for g,weights in enumerate(data['attachment_weights']):
        near=np.flatnonzero(weights>weights.max()*.5);ids=near[fps(rest[near],min(4,len(near)))]
        axis=rest[ids].mean(0)-rest.mean(0);axis=axis/max(np.linalg.norm(axis),1e-9)
        offset=axis*.004
        ctrl=np.broadcast_to(rest[ids]+offset,(T,len(ids),3)).copy()
        active=data['attachment_active'][:,g]>0
        ctrl[active]=data['controller_targets'][active,g][:,ids]+offset
        controls.append(ctrl)
    # The native floor is z=0. Shift all positions by the same known initial floor.
    shift=np.array([0.,0.,-data['floor']]);xyz+=shift
    return dict(object_points=xyz.astype('float32'),object_colors=np.full((T,N,3),.5,dtype='float32'),
        object_visibilities=mask,object_motions_valid=mask[1:],controller_points=(np.concatenate(controls,axis=1)+shift).astype('float32'),
        surface_points=np.empty((0,3),dtype='float32'),interior_points=np.empty((0,3),dtype='float32')),shift

class Trainer(InvPhyTrainerWarp):
    def visualize_sim(self,*args,**kwargs):pass
    def rollout(self,end):
        sim=self.simulator;sim.set_init_state(sim.wp_init_vertices,sim.wp_init_velocities)
        out=[wp.to_torch(sim.wp_init_vertices,requires_grad=False).cpu().clone().numpy()]
        for frame in range(1,end):
            if hasattr(self,'release_command') and frame>self.grasp and self.release_command[frame]>0:
                # Remove kinematic gripper springs on an explicit release command.
                # This adapter operation leaves native object springs/dynamics intact.
                with torch.no_grad():wp.to_torch(sim.wp_spring_Y)[self.controller_springs]=0
            check_deadline();sim.set_controller_target(frame,pure_inference=True);wp.capture_launch(sim.forward_graph)
            out.append(wp.to_torch(sim.wp_states[-1].wp_x,requires_grad=False).cpu().clone().numpy())
            sim.set_init_state(sim.wp_states[-1].wp_x,sim.wp_states[-1].wp_v)
        return np.stack(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subset',default='object-soft');p.add_argument('--task',type=int,default=0);p.add_argument('--episode',type=int,default=0)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--development',action='store_true');p.add_argument('--initial',type=float,default=30000);p.add_argument('--tag',default='native-pilot')
    a=p.parse_args();check_deadline();data=softvt(a.subset,a.task,a.episode);fit_end=int(data['train_end']*.7) if a.development else data['train_end']
    destination=BASE/'native'/a.tag/data['name'];destination.mkdir(parents=True,exist_ok=True)
    save_provenance(destination,data)
    train,shift=make_native(data,fit_end);path=destination/'train.pkl'
    with path.open('wb') as f:pickle.dump(train,f)
    cfg.load_from_yaml(str(UP/'configs/real.yaml'));cfg.FPS=20;cfg.dt=.00005;cfg.num_substeps=1000
    cfg.reverse_z=False;cfg.gravity_scale=1.;cfg.iterations=a.epochs;cfg.vis_interval=10000;cfg.init_spring_Y=a.initial
    cfg.object_radius=.014;cfg.object_max_neighbours=24;cfg.controller_radius=.009;cfg.controller_max_neighbours=12
    cfg.dashpot_damping=30.;cfg.drag_damping=3.;cfg.spring_Y_min=1.;cfg.spring_Y_max=1e6
    cfg.collision_learn=True;cfg.self_collision=False;cfg.base_lr=.003
    np.random.seed(42);torch.manual_seed(42);logger.set_log_file(path=str(destination),name='fit')
    start=time.time();trainer=Trainer(str(path),str(destination),train_frame=fit_end);trainer.train()
    files=list((destination/'train').glob('best_*.pth'))
    checkpoint=max(files,key=lambda p:int(p.stem.split('_')[-1]));trainer.test(str(checkpoint))
    trainer.release_command=data['gripper_command'];trainer.grasp=data['grasp']
    edges=wp.to_torch(trainer.simulator.wp_springs).detach()
    trainer.controller_springs=torch.any(edges>=len(data['nodes']),dim=1)
    end=data['train_end'] if a.development else len(data['times']);pred=trainer.rollout(end)[:,:len(data['nodes'])]-shift
    if not np.isfinite(pred).all():raise RuntimeError('Nonfinite native baseline prediction')
    np.savez_compressed(destination/'prediction.npz',predicted=pred,nodes=data['nodes'])
    write_json(destination/'fit.json',dict(name=data['name'],fit_end=fit_end,epochs=a.epochs,initial=a.initial,
        elapsed_seconds=time.time()-start,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        scope='Adapted official PhysTwin physical core; identical sparse observed prefix; control points attach to initial geometric gripper neighborhoods. Unit particle mass convention, learned spring stiffness, native ground collisions; no force labels.'))
