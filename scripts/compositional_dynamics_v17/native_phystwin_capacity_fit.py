"""729-particle capacity control: unchanged author CMA/per-spring recipe, denser known geometry.
No evaluator or private scene imports. Headless adapters change presentation only.
"""
from pathlib import Path
import hashlib,json,pickle,sys,os,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
AUTHOR=ROOT/'runs/compositional-dynamics-v17/PhysTwin-author'
PUBLIC=ROOT/'runs/compositional-dynamics-v17/native-phystwin-capacity-public'
OUT=ROOT/'runs/compositional-dynamics-v17/native-phystwin-capacity-fit'
PLAN=ROOT/'reports/compositional-dynamics-v17/native-phystwin-capacity-plan.json'
KEYS={'object_points','object_colors','object_visibilities','object_motions_valid','controller_points','surface_points','interior_points'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def native_data(data,floor_z,forecast_frames=None):
    if set(data)!=KEYS:raise ValueError('Only explicit public input fields allowed')
    out={k:np.array(v,copy=True) for k,v in data.items()}
    if forecast_frames is not None:
        n=int(forecast_frames);x=out['object_points'][0]
        out['object_points']=np.repeat(x[None],n,axis=0)
        out['object_colors']=np.full((n,len(x),3),.5,np.float32)
        out['object_visibilities']=np.zeros((n,len(x)),bool)
        out['object_motions_valid']=np.zeros((n-1,len(x)),bool)
        out['controller_points']=np.empty((n,0,3),np.float32)
    for k in ['object_points','controller_points','surface_points','interior_points']:out[k][...,2]-=floor_z
    return out

def guard():
    p=json.loads(PLAN.read_text())
    for group in ['source_sha256','input_sha256']:
        for name,digest in p[group].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Changed pinned file '+name)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Physical GPU0 only')
    return p

def configure(cfg,metadata):
    cfg.load_from_yaml(str(AUTHOR/'configs/real.yaml'))
    masses=np.asarray(metadata['object_masses_kg'])
    if len(masses)!=729 or not np.allclose(masses,masses[0],rtol=0,atol=1e-12):raise ValueError('Uniform public mass contract')
    alpha=float(masses[0]);cfg.device='cuda:0';cfg.data_type='real';cfg.use_graph=True
    cfg.FPS=100;cfg.dt=.000025;cfg.num_substeps=400;cfg.reverse_z=False;cfg.object_radius=.02
    cfg.init_spring_Y*=alpha;cfg.spring_Y_max*=alpha;cfg.dashpot_damping*=alpha
    return alpha

def runtime_classes(masses,alpha):
    from scripts.compositional_dynamics_v17.native_phystwin_inputs import apply_known_object_masses
    import qqtt.engine.cma_optimize_warp as cm
    import qqtt.engine.trainer_warp as tm
    from qqtt.data import RealData
    class NoVisualData(RealData):
        def visualize_data(self,*args,**kwargs):pass
    class LocalLog:
        @staticmethod
        def init(*args,**kwargs):pass
        @staticmethod
        def log(values,step=None):
            with (OUT/'training-log.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,values=values),default=str)+'\n')
        @staticmethod
        def Video(*args,**kwargs):return 'Headless presentation omitted'
        @staticmethod
        def finish():pass
    cm.RealData=NoVisualData;tm.RealData=NoVisualData;tm.wandb=LocalLog
    class MassMixin:
        def _init_start(self,*args,**kwargs):return apply_known_object_masses(super()._init_start(*args,**kwargs),masses)
    class CMA(MassMixin,cm.OptimizerCMA):
        def normalize(self,value,min,max):
            # Author's only [0,200] parameter is dashpot (kg/s), not drag (1/s).
            if min==0 and max==200:max*=alpha
            return super().normalize(value,min,max)
        def denormalize(self,value,min,max):
            if min==0 and max==200:max*=alpha
            return super().denormalize(value,min,max)
        def error_func(self,parameters,visualize=False,video_path=None):
            value=super().error_func(parameters,visualize=False,video_path=None)
            if not np.isfinite(value):raise FloatingPointError('Native candidate produced nonfinite loss; retained as failed run')
            with (OUT/'cma-evaluations.jsonl').open('a') as f:f.write(json.dumps(dict(parameters=np.asarray(parameters).tolist(),loss=float(value)))+'\n')
            return value
    class Trainer(MassMixin,tm.InvPhyTrainerWarp):
        def visualize_sim(self,*args,**kwargs):pass
    return CMA,Trainer

def run(stage):
    plan=guard();start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    marker=OUT/(stage+'-started.json')
    with marker.open('x') as f:json.dump(dict(plan_sha256=sha(PLAN),stage=stage),f)
    status=dict(stage=stage,completed=False,plan_sha256=sha(PLAN))
    try:
        os.environ['PYNPUT_BACKEND']='dummy';sys.path.insert(0,str(AUTHOR))
        import torch,random,warp as wp
        from qqtt.utils import cfg
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        metadata=json.loads((PUBLIC/'metadata.json').read_text())
        with np.load(PUBLIC/'observations.npz') as f:data={k:f[k] for k in f.files}
        alpha=configure(cfg,metadata);CMA,Trainer=runtime_classes(metadata['object_masses_kg'],alpha)
        trainfile=OUT/'prefix.pkl'
        if stage=='fit':
            with trainfile.open('xb') as f:pickle.dump(native_data(data,metadata['public_floor_z_m']),f)
            cma=CMA(str(trainfile),str(OUT),train_frame=21,device='cuda:0');cma.optimize(max_iter=20)
            with (OUT/'optimal_params.pkl').open('rb') as f:optimal=pickle.load(f)
            cfg.set_optimal_params(optimal)
            trainer=Trainer(str(trainfile),str(OUT),train_frame=21,device='cuda:0');trainer.train()
            best=list((OUT/'train').glob('best_*.pth'))
            if len(best)!=1:raise RuntimeError('Exactly one author prefix-selected checkpoint required')
            np.save(OUT/'fitted-spring-endpoints.npy',trainer.init_springs.detach().cpu().numpy())
            status.update(spring_endpoints_sha256=sha(OUT/'fitted-spring-endpoints.npy'),checkpoint=str(best[0].relative_to(ROOT)),checkpoint_sha256=sha(best[0]),optimal_params_sha256=sha(OUT/'optimal_params.pkl'),prefix_sha256=sha(trainfile),epochs=200,cma_generations=20)
        else:
            fit=json.loads((OUT/'fit-complete.json').read_text())
            if not fit['completed'] or fit['plan_sha256']!=sha(PLAN):raise RuntimeError('Completed same-plan fit required')
            cp=ROOT/fit['checkpoint']
            if sha(cp)!=fit['checkpoint_sha256'] or sha(OUT/'optimal_params.pkl')!=fit['optimal_params_sha256']:raise RuntimeError('Fit artifact changed')
            with (OUT/'optimal_params.pkl').open('rb') as f:cfg.set_optimal_params(pickle.load(f))
            setup=metadata['forecast_public_setup'];floor=float(setup['floor_z_m']);n=26
            forecast=OUT/'forecast-input.pkl'
            with forecast.open('xb') as f:pickle.dump(native_data(data,floor,n),f)
            trainer=Trainer(str(forecast),str(OUT),train_frame=n,pure_inference_mode=True,device='cuda:0')
            checkpoint=torch.load(cp,map_location='cuda:0',weights_only=False);sim=trainer.simulator
            if sha(OUT/'fitted-spring-endpoints.npy')!=fit['spring_endpoints_sha256']:raise ValueError('Fitted graph artifact changed')
            fitted_edges=np.load(OUT/'fitted-spring-endpoints.npy')
            if not np.array_equal(fitted_edges,trainer.init_springs.detach().cpu().numpy()):raise ValueError('Native spring endpoint order changed; no silent per-edge weight reassignment')
            if len(checkpoint['spring_Y'])!=sim.n_springs:raise ValueError('Native spring topology changed')
            sim.set_spring_Y(torch.log(checkpoint['spring_Y']).detach().clone())
            sim.set_collide(checkpoint['collide_elas'].detach().clone(),checkpoint['collide_fric'].detach().clone())
            sim.set_collide_object(checkpoint['collide_object_elas'].detach().clone(),checkpoint['collide_object_fric'].detach().clone())
            sim.set_init_state(sim.wp_init_vertices,sim.wp_init_velocities,pure_inference=True)
            frames=[sim.wp_init_vertices.numpy().copy()]
            for i in range(1,n):
                sim.set_controller_target(i,pure_inference=True)
                if sim.object_collision_flag:sim.update_collision_graph()
                wp.capture_launch(sim.forward_graph);wp.synchronize()
                frame=sim.wp_states[-1].wp_x.numpy().copy()
                if not np.isfinite(frame).all():raise FloatingPointError('Nonfinite native forecast')
                frames.append(frame)
                sim.set_init_state(sim.wp_states[-1].wp_x,sim.wp_states[-1].wp_v,pure_inference=True)
                world=np.asarray(frames);world[...,2]+=floor
                np.savez_compressed(OUT/'forecast.npz',times_s=np.arange(len(world))*.01,positions_m=world,tracked_positions_m=world[:,:98])
            status.update(forecast_sha256=sha(OUT/'forecast.npz'),frames=n,checkpoint_sha256=sha(cp),forecast_input_sha256=sha(forecast))
        guard();status['completed']=True
    except Exception as exc:
        status['error']=repr(exc);raise
    finally:
        status['wall_seconds']=time.monotonic()-start;write(OUT/(stage+'-complete.json'),status)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','predict']);a=p.parse_args();run(a.stage)
