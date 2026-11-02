"""Parameterized, input-only replay of a fitted native graph; never refits parameters."""
import importlib.util,json,pickle,os,sys,time
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_fit import ROOT,AUTHOR,sha,write,native_data
from scripts.compositional_dynamics_v17.native_phystwin_graph_recovery import replace_initial_vertices
SOURCE=ROOT/'scripts/compositional_dynamics_v17/native_phystwin_budget_fit.py'

def validate_setup(setup):
    allowed={'duration_s','frame_dt_s','floor_z_m','initial_velocity_m_s','initial_geometry_same','gravity_vector_m_s2'}
    if set(setup)!=allowed:raise ValueError('Explicit public reset setup only')
    dt=float(setup['frame_dt_s']);duration=float(setup['duration_s'])
    if dt!=.01 or not np.isfinite(duration)or duration<=0 or abs(duration/dt-round(duration/dt))>1e-10:raise ValueError('Native public100Hz clock')
    if setup['initial_velocity_m_s']!=[0,0,0]or setup['gravity_vector_m_s2']!=[0,0,-9.8]or setup['initial_geometry_same']is not True:raise ValueError('Fixed native initial state/gravity recipe')
    if not np.isfinite(setup['floor_z_m']):raise ValueError('Finite floor position')
    return int(round(duration/dt))+1

def fitted_files(fit_directory):
    base=Path(fit_directory);f=json.loads((base/'fit-complete.json').read_text())
    if f.get('completed')is not True:raise ValueError('Completed fit required')
    files={base/'prefix.pkl':f['prefix_sha256'],base/'optimal_params.pkl':f['optimal_params_sha256'],base/'fitted-spring-endpoints.npy':f['spring_endpoints_sha256'],ROOT/f['checkpoint']:f['checkpoint_sha256']}
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Fitted artifact seal changed')
    return f,files

def replay(fit_directory,public_directory,setup,output_directory,plan_sha256):
    n=validate_setup(setup);fit_directory=Path(fit_directory);public_directory=Path(public_directory);output=Path(output_directory)
    if output.exists():raise ValueError('Preserve previous replay')
    fitted,_=fitted_files(fit_directory);output.mkdir(parents=True);started=time.monotonic()
    status=dict(completed=False,plan_sha256=plan_sha256,expected_frames=n,fit_commit_sha256=sha(fit_directory/'fit-complete.json'),setup=setup)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Fixed physicalGPU0 recipe')
        os.environ['PYNPUT_BACKEND']='dummy';sys.path.insert(0,str(AUTHOR))
        import torch,warp as wp
        from qqtt.utils import cfg
        spec=importlib.util.spec_from_file_location('_native_replay_local',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=output
        metadata=json.loads((public_directory/'metadata.json').read_text())
        with np.load(public_directory/'observations.npz')as f:data={key:f[key]for key in f.files}
        alpha=m.configure(cfg,metadata);_,Trainer=m.runtime_classes(metadata['object_masses_kg'],alpha)
        with (fit_directory/'optimal_params.pkl').open('rb')as f:cfg.set_optimal_params(pickle.load(f))
        cfg.drag_damping=0.0;cfg.dashpot_damping=0.0  # Sole post-hoc combined-removal intervention, no refit.
        with (fit_directory/'prefix.pkl').open('rb')as f:prefix=pickle.load(f)
        fixed=np.concatenate([prefix['object_points'][0],prefix['surface_points'],prefix['interior_points']],axis=0)
        class FixedGraph(Trainer):
            def _init_start(self,object_points,controller_points,*args,**kwargs):
                old=torch.as_tensor(fixed,dtype=object_points.dtype,device=object_points.device)
                graph=super()._init_start(old,controller_points,*args,**kwargs)
                if not np.array_equal(graph[1].detach().cpu().numpy(),np.load(fit_directory/'fitted-spring-endpoints.npy')):raise ValueError('Fitted endpoint identity changed')
                return replace_initial_vertices(graph,object_points,controller_points)
        floor=float(setup['floor_z_m']);inputfile=output/'reset-input.pkl'
        with inputfile.open('xb')as f:pickle.dump(native_data(data,floor,n),f)
        trainer=FixedGraph(str(inputfile),str(output),train_frame=n,pure_inference_mode=True,device='cuda:0');sim=trainer.simulator
        checkpoint=torch.load(ROOT/fitted['checkpoint'],map_location='cuda:0',weights_only=False)
        if len(checkpoint['spring_Y'])!=sim.n_springs:raise ValueError('Fitted edge count mismatch')
        sim.set_spring_Y(torch.log(checkpoint['spring_Y']).detach().clone())
        sim.set_collide(checkpoint['collide_elas'].detach().clone(),checkpoint['collide_fric'].detach().clone())
        sim.set_collide_object(checkpoint['collide_object_elas'].detach().clone(),checkpoint['collide_object_fric'].detach().clone())
        sim.set_init_state(sim.wp_init_vertices,sim.wp_init_velocities,pure_inference=True);frames=[sim.wp_init_vertices.numpy().copy()]
        def save_prefix():
            world=np.asarray(frames).copy();world[...,2]+=floor
            temporary=output/'prediction.tmp.npz'
            np.savez_compressed(temporary,times_s=np.arange(len(world))*.01,positions_m=world,tracked_positions_m=world[:,:98]);temporary.replace(output/'prediction.npz')
        save_prefix()
        for i in range(1,n):
            sim.set_controller_target(i,pure_inference=True)
            if sim.object_collision_flag:sim.update_collision_graph()
            wp.capture_launch(sim.forward_graph);wp.synchronize();frame=sim.wp_states[-1].wp_x.numpy().copy()
            if not np.isfinite(frame).all():raise FloatingPointError('Nonfinite native replay')
            frames.append(frame);save_prefix();sim.set_init_state(sim.wp_states[-1].wp_x,sim.wp_states[-1].wp_v,pure_inference=True)
        fitted_files(fit_directory)
        status.update(completed=True,frames=n,prediction_sha256=sha(output/'prediction.npz'),reset_input_sha256=sha(inputfile),checkpoint_sha256=fitted['checkpoint_sha256'])
    except Exception as exc:status['error']=repr(exc);raise
    finally:
        status['wall_seconds']=time.monotonic()-started
        if(output/'prediction.npz').exists():status['retained_prefix_sha256']=sha(output/'prediction.npz')
        write(output/'complete.json',status)
    return status

def main(plan_path):
    plan_path=Path(plan_path);p=json.loads(plan_path.read_text())
    def guard():
        for path,digest in p['sha256'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Replay source/input changed '+path)
        if sha(plan_path)!=plan_hash:raise ValueError('Replay plan changed')
    plan_hash=sha(plan_path);guard()
    result=replay(ROOT/p['fit_directory'],ROOT/p['public_directory'],p['setup'],ROOT/p['output_directory'],plan_hash);guard();return result
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);main(a.parse_args().plan)
