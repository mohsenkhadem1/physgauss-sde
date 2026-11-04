"""Frozen released PhysTwin on new controls, retaining source object springs.

Target object observations after frame zero are not passed to the simulator.
Controller springs are reconstructed geometrically and receive the median
source controller stiffness. This adapter policy is disclosed, not an official
author cross-interaction result. Self-replay retains exact original springs.
"""
import os,sys,argparse,json,pickle,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'runs/third_party/PhysTwin'
os.environ['PHYSTWIN_HEADLESS']='1';os.environ['WANDB_MODE']='disabled'
sys.path.insert(0,str(UP))
import numpy as np,torch,warp as wp
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg,logger
from .development_run import write,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--target',required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.set_num_threads(1);np.random.seed(42);torch.manual_seed(42)
    out=ROOT/'runs/real-transfer-v20/native-transfer'/f'{args.source}--{args.target}'
    out.mkdir(parents=True,exist_ok=False)
    source_path=UP/'data/different_types'/args.source/'final_data.pkl'
    target_path=UP/'data/different_types'/args.target/'final_data.pkl'
    with source_path.open('rb') as f:source=pickle.load(f)
    with target_path.open('rb') as f:target=pickle.load(f)
    flip=np.diag([1.,-1.,-1.]);R=np.eye(3);shift=np.zeros(3);registration=None
    if args.source!=args.target:
        report=json.loads((ROOT/'reports/real-transfer-v20/initial-registration.json').read_text())
        registration=next(r for r in report['pairs'] if r['source']==args.source and r['target']==args.target)
        R=flip@np.array(registration['rotation'])@flip;shift=np.array(registration['translation_m'])@flip
    canonical=np.concatenate([source['object_points'][0],source['surface_points'],source['interior_points']])
    controls=np.asarray(target['controller_points']);T=len(controls);N=len(source['object_points'][0])
    bundle=dict(object_points=np.broadcast_to(source['object_points'][0]@R+shift,(T,N,3)).copy(),
        object_colors=np.broadcast_to(source['object_colors'][0],(T,N,3)).copy(),
        object_visibilities=np.zeros((T,N),bool),object_motions_valid=np.zeros((T-1,N),bool),
        surface_points=source['surface_points']@R+shift,interior_points=source['interior_points']@R+shift,
        controller_points=controls)
    del target,source
    path=out/'input.pkl'
    with path.open('wb') as f:pickle.dump(bundle,f)
    cfg.load_from_yaml(str(UP/'configs/real.yaml'))
    optimal=UP/'experiments_optimization'/args.source/'optimal_params.pkl'
    with optimal.open('rb') as f:cfg.set_optimal_params(pickle.load(f))
    checkpoint=UP/'reference/experiments'/args.source/'train/best_199.pth'
    state=torch.load(checkpoint,map_location=args.device,weights_only=False)
    class FrozenTrainer(InvPhyTrainerWarp):
        def visualize_sim(self,*a,**k):pass
        def _init_start(self,object_points,controller_points,**kwargs):
            target_graph=super()._init_start(object_points,controller_points,**kwargs)
            if args.source==args.target:return target_graph
            original_graph=super()._init_start(torch.tensor(canonical,dtype=torch.float32,device=args.device),None,**kwargs)
            vertices,edges,lengths,masses,n_target=target_graph
            _,object_edges,object_lengths,_,n_source=original_graph
            return vertices,torch.cat([object_edges,edges[n_target:]]),torch.cat([object_lengths,lengths[n_target:]]),masses,n_source
    logger.set_log_file(path=str(out),name='transfer')
    trainer=FrozenTrainer(str(path),str(out),pure_inference_mode=True,device=args.device)
    n=int(state['num_object_springs']);n_new=trainer.num_object_springs
    if n!=n_new:raise ValueError(f'Source topology mismatch: {n} vs {n_new}')
    stiffness=state['spring_Y'].detach().clone()
    if args.source!=args.target:
        control_value=stiffness[n:].median()
        stiffness=torch.cat([stiffness[:n],control_value.expand(trainer.simulator.n_springs-n)])
    if len(stiffness)!=trainer.simulator.n_springs:raise ValueError('Spring count mismatch')
    sim=trainer.simulator;sim.set_spring_Y(torch.log(stiffness))
    sim.set_collide(state['collide_elas'],state['collide_fric'])
    sim.set_collide_object(state['collide_object_elas'],state['collide_object_fric'])
    sim.set_init_state(sim.wp_init_vertices,sim.wp_init_velocities)
    result=[wp.to_torch(sim.wp_init_vertices,requires_grad=False).cpu().clone().numpy()]
    for frame in range(1,T):
        sim.set_controller_target(frame,pure_inference=True);wp.capture_launch(sim.forward_graph)
        result.append(wp.to_torch(sim.wp_states[-1].wp_x,requires_grad=False).cpu().clone().numpy())
        sim.set_init_state(sim.wp_states[-1].wp_x,sim.wp_states[-1].wp_v)
        if frame%20==0:print('frame',frame,flush=True)
    prediction=np.asarray(result)[:,:len(canonical)]@flip
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite native prediction')
    np.savez_compressed(out/'prediction.npz',predicted=prediction)
    replay=None
    if args.source==args.target:
        with (UP/'reference/experiments'/args.source/'inference.pkl').open('rb') as f:reference=np.asarray(pickle.load(f))[:,:len(canonical)]@flip
        replay=float(np.max(np.abs(reference-prediction)))
    write(out/'complete.json',dict(source=args.source,target=args.target,frames=T,
          source_checkpoint_sha256=sha(checkpoint),input_sha256=sha(path),prediction_sha256=sha(out/'prediction.npz'),
          replay_max_abs_m=replay,registration=registration,target_post_initial_geometry_used=False,
          source_object_springs=n,new_controller_springs=sim.n_springs-n,
          boundary_policy='Geometric new controller springs, median source controller stiffness; exact original values for self-replay.',
          scope='Frozen source PhysTwin adapter; no target material fitting. Registration failures remain diagnostic, not qualified OOD evidence.'))
    print('complete',args.source,args.target,'replay',replay,flush=True)

if __name__=='__main__':main()
