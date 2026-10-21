"""Isolate solver-iteration and world-coordinate sensitivity on one TRAIN case."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import numpy as np

try:
    from campaign_numerical_convergence import atomic,checked_write,sha,command_clock_audit
    from campaign_numerical_validate_rates import adapt_reset_reference
    from campaign_numerical_force_polynomial import finite_strain
except ImportError:
    from experiments.campaign_numerical_convergence import atomic,checked_write,sha,command_clock_audit
    from experiments.campaign_numerical_validate_rates import adapt_reset_reference
    from experiments.campaign_numerical_force_polynomial import finite_strain


def translate_reference(points,spec,pose,translation):
    shift=np.asarray(translation,dtype=np.float64)
    if shift.shape!=(3,) or not np.isfinite(shift).all():raise ValueError('Finite translation required')
    p=np.asarray(points,dtype=float)+shift;s=copy.deepcopy(spec);c=np.asarray(pose,dtype=float).copy()
    s['support_centers_m']=(np.asarray(s['support_centers_m'])+shift).tolist()
    s['camera']['center_m']=(np.asarray(s['camera']['center_m'])+shift).tolist()
    c[...,3,:3]+=shift
    return p,s,c


def install_geometry(config,condition):
    import isaac_dataset
    original=isaac_dataset.geometry;translation=condition['translation_m']
    def geometry(case):
        if case['episode_id']!='t0001' or case['split']!='train' or case['physics_hz']!=960:raise ValueError('Wrong numerical role/rate')
        points,tets,faces,spec=original(case)
        if not np.any(translation):return points,tets,faces,spec
        points,spec,_=translate_reference(points,spec,np.eye(4),translation)
        return points,tets,faces,spec
    isaac_dataset.geometry=geometry
    return isaac_dataset,original


def safe_json(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    try:return [safe_json(x) for x in value]
    except TypeError:return str(value)


def assert_paused_state_unchanged(before,after):
    if before.keys()!=after.keys():raise ValueError('Paused state fields changed')
    for key in before:
        if not np.array_equal(before[key],after[key]):
            raise RuntimeError('Synchronous sensor capture changed physical state: '+key)
    return {key:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key,value in before.items()}


def solver_snapshot(stage,world,count):
    selected={'deformable':'/World/ProbeObject/Block','scene':'/physicsScene',
        'handle0':'/World/ProbeObject/Load_0','handle1':'/World/ProbeObject/Load_1'}
    result={'requested_deformable_iterations':count,'nominal_dt_s':1/960,
        'float32_nominal_dt_s':float(np.float32(1/960)),
        'world_get_physics_dt_s':float(world.get_physics_dt()),'prims':{}}
    for name,path in selected.items():
        prim=stage.GetPrimAtPath(path)
        result['prims'][name]={'path':path,'applied_schemas':list(prim.GetAppliedSchemas()),'attributes':{a.GetName():
            {'value':safe_json(a.Get()),'authored':a.HasAuthoredValueOpinion(),'type':str(a.GetTypeName())}
            for a in prim.GetAttributes() if a.GetName().startswith(('physx','physics:','omniphysics:'))
            and 'Points' not in a.GetName() and 'Indices' not in a.GetName()}}
        prefix='physxDeformableBody:' if name=='deformable' else 'physxRigidBody:'
        result['prims'][name]['requested_sleep_property_validity']={
            prefix+k:bool(prim.GetAttribute(prefix+k)) for k in
            ['sleepThreshold','settlingThreshold','settlingDamping','sleepDamping','disableSleep','stabilizationThreshold']}
    actual=result['prims']['deformable']['attributes']['physxDeformableBody:solverPositionIterationCount']['value']
    if actual!=count:raise ValueError('Requested solver count was not authored')
    return result


def prepare(output,config):
    dataset=Path(config['dataset']);original=dataset/'episodes/t0001'
    case=json.loads((original/'truth/recipe.json').read_text())
    if case['split']!='train' or case['physics_hz']!=960 or case['boundary_type']!='fixed_patch':raise ValueError('Wrong TRAIN case')
    spec=json.loads((original/'truth/boundaries.json').read_text())
    with np.load(original/'reference.npz') as raw:points=raw['rest_points_m']
    with np.load(original/'observations/cameras.npz') as raw:camera={k:raw[k].copy() for k in raw.files}
    for condition in config['coordinate_conditions']:
        name=condition['name'];shift=condition['translation_m'];root=Path(output)/name/'staged-dataset';ep=root/'episodes/t0001'
        p,s,pose=translate_reference(points,spec,camera['camera_to_world_usd_row'],shift)
        checked_write(ep/'truth/recipe.json',(original/'truth/recipe.json').read_bytes())
        actual_spec=spec if not np.any(shift) else s
        checked_write(ep/'truth/boundaries.json',(json.dumps(actual_spec,indent=2)+'\n').encode())
        path=ep/'observations/cameras.npz';path.parent.mkdir(parents=True,exist_ok=True)
        if not np.any(shift):checked_write(path,(original/'observations/cameras.npz').read_bytes())
        elif path.exists():
            with np.load(path) as old:
                for k,v in camera.items():np.testing.assert_array_equal(old[k],pose if k=='camera_to_world_usd_row' else v)
        else:np.savez_compressed(path,**{**camera,'camera_to_world_usd_row':pose})
        # The geometry hook supplies these exact points, therefore shifts all
        # physics nodes, attachment anchors/offsets, rest mesh and support targets.
        checked_write(root/'lineage.json',(json.dumps({'translation_m':shift,
            'original_recipe_sha256':sha(original/'truth/recipe.json'),
            'original_boundaries_sha256':sha(original/'truth/boundaries.json'),
            'staged_boundaries_sha256':sha(ep/'truth/boundaries.json'),
            'staged_cameras_sha256':sha(path),
            'reference_points_float64_sha256':hashlib.sha256(p.tobytes()).hexdigest(),
            'physical_geometry_translated':True,'qualification':config['qualification']},indent=2)+'\n').encode())


def worker(args,config):
    import isaac_calibrated_probes_v03 as base
    if sha(base.__file__)!=config['base_worker_sha256']:raise ValueError('Base changed')
    condition=next(c for c in config['coordinate_conditions'] if c['name']==args.condition)
    if args.iterations not in config['iterations']:raise ValueError('Undeclared count')
    install_geometry(config,condition);original_env=base.create_environment;original_menu=base.calibrated_menu
    original_capture=base.capture_current_state
    state_root=None;warmup_events=[]
    def read_state(world,body,handles):
        from isaacsim.core.simulation_manager import SimulationManager
        p,r=handles.get_world_poses();v,w=handles.get_velocities()
        return {'nodal_q':body.get_nodal_positions()[0].numpy().copy(),
            'nodal_v':body.get_nodal_velocities().numpy().copy(),
            'rigid_p':p.numpy().copy(),'rigid_r':r.numpy().copy(),
            'rigid_v':v.numpy().copy(),'rigid_w':w.numpy().copy(),
            'world_time':np.array(world.current_time),'world_step_index':np.array(world.current_time_step_index),
            'manager_step_count':np.array(SimulationManager.get_num_physics_steps()),
            'world_dt':np.array(world.get_physics_dt()),'manager_dt':np.array(SimulationManager.get_physics_dt())}
    def guarded_state_comparison(before,after,phase):
        try:return assert_paused_state_unchanged(before,after)
        except Exception:
            np.savez_compressed(state_root/'paused-state-violation.npz',
                **{'before_'+k:v for k,v in before.items()},**{'after_'+k:v for k,v in after.items()})
            atomic(state_root/'paused-state-violation.json',{'phase':phase,
                'maximum_absolute_changes':{k:float(np.abs(np.asarray(before[k],dtype=float)-after[k]).max()) for k in before}})
            raise
    def environment(stage,case,root):
        nonlocal state_root
        from isaacsim.core.api import World
        state_root=Path(root)
        env=original_env(stage,case,root)
        attr=stage.GetPrimAtPath('/World/ProbeObject/Block').GetAttribute('physxDeformableBody:solverPositionIterationCount')
        if not attr.Set(args.iterations):raise ValueError('Solver attribute refused')
        atomic(Path(root)/'actual-solver-state.json',solver_snapshot(stage,World.instance(),args.iterations))
        original_render=World.render
        def render(world,*a,**kw):
            frame=inspect.currentframe().f_back
            try:
                if frame.f_code is not base.run_worker.__code__:return original_render(world,*a,**kw)
                body,handles=frame.f_locals['body'],frame.f_locals['handles']
                action=frame.f_locals['action_id']
            finally:del frame
            before=read_state(world,body,handles);result=original_render(world,*a,**kw);after=read_state(world,body,handles)
            hashes=guarded_state_comparison(before,after,'initial_warmup_render')
            warmup_events.append({'action_id':action,'physics_state_sha256':hashes,'bitwise_unchanged':True})
            atomic(state_root/'warmup-state-guards.json',warmup_events)
            if not (state_root/'runtime-getter-availability.json').exists():
                views={'deformable_prim':body,'rigid_prim':handles,
                    'deformable_tensor_view':body._physics_deformable_body_view,
                    'rigid_tensor_view':handles._physics_rigid_body_view}
                atomic(state_root/'runtime-getter-availability.json',{name:{method:hasattr(view,method)
                    for method in ['get_sleep_thresholds','get_stabilization_thresholds','get_settling_thresholds',
                                   'get_settling_dampings','get_solver_position_iteration_counts']}
                    for name,view in views.items()})
            return result
        World.render=render
        return env
    def menu(*a,**kw):
        m=original_menu(*a,**kw);m['candidates']=[{**m['candidates'][0],
            'action_id':'zero_force','kind':'solver_coordinate_audit_zero','peak_force_N':[0.,0.,0.]}]
        m['protocol']=config['protocol'];m['qualification']=config['qualification'];return m
    def capture(world,*a,**kw):
        # These two locals belong to the hash-pinned base worker. Reusing its
        # live views avoids constructing/resetting physics views during capture.
        frame=inspect.currentframe().f_back
        try:
            if frame.f_code is not base.run_worker.__code__:raise RuntimeError('Unexpected capture caller')
            body,handles=frame.f_locals['body'],frame.f_locals['handles']
        finally:del frame
        before=read_state(world,body,handles);sensor,audit=original_capture(world,*a,**kw);after=read_state(world,body,handles)
        audit['paused_physics_state_sha256']=guarded_state_comparison(before,after,'synchronous_capture')
        audit['paused_physics_state_bitwise_unchanged']=True
        return sensor,audit
    base.create_environment=environment;base.calibrated_menu=menu;base.capture_current_state=capture
    return base.run_worker(argparse.Namespace(dataset=Path(args.dataset),output=Path(args.captures),episode='t0001',gpu=condition['gpu']))


def validate_cell(output,config,condition,count):
    import isaac_calibrated_probes_v03 as base
    module,original=install_geometry(config,condition)
    try:
        root=Path(output)/condition['name']/f'i{count}'/'captures'
        source=adapt_reset_reference(inspect.getsource(base.validate_case),'zero_force')
        target=Path(output)/'provenance/adapted_validate_case.py';checked_write(target,source.encode())
        namespace=dict(vars(base));exec(compile(source,str(target),'exec'),namespace)
        result=namespace['validate_case'](root,'t0001')
        result['additional_geometry_solver_wrapper_sha256']=sha(__file__)
        result['reset_adapter_sha256']=sha(target)
        atomic(root/'private/t0001/validation.json',result)
        return result
    finally:module.geometry=original


def analyze(output,config):
    output=Path(output);loaded={};rows=[];hashes={}
    for condition in config['coordinate_conditions']:
        name=condition['name'];shift=np.array(condition['translation_m']);loaded[name]={}
        for count in config['iterations']:
            private=output/name/f'i{count}/captures/private/t0001'
            if not json.loads((private/'validation.json').read_text()).get('passed'):raise ValueError('Unverified cell')
            metadata=json.loads((private/'actual-solver-state.json').read_text())
            with np.load(private/'reference.npz') as raw:rest,tets=raw['rest_points_m'],raw['tetrahedra']
            loaded[name][count]={}
            for action in ['zero_force','evaluation']:
                path=private/action/'trajectory.npz'
                with np.load(path) as raw:data={k:raw[k].copy() for k in raw.files}
                q=data['positions_m'].astype(float);loaded[name][count][action]={**data,'canonical_positions_m':q-shift}
                hashes[str(path)]=sha(path)
                peak=[0,0,0] if action=='zero_force' else [.01,.01,.005]
                rows.append({'condition':name,'iterations':count,'action_id':action,
                    'coordinate_rms_from_rest_m':float(np.sqrt(np.mean((q-rest)**2))),
                    'strain':finite_strain(rest,tets,q),'command_clock':command_clock_audit(data,960,metadata,peak),
                    'actual_solver_state_sha256':sha(private/'actual-solver-state.json')})
    def difference(a,b):
        dq=a['canonical_positions_m']-b['canonical_positions_m'];dv=a['velocities_m_s'].astype(float)-b['velocities_m_s'].astype(float)
        return {'coordinate_rmse_m':float(np.sqrt(np.mean(dq**2))),
            'maximum_nodal_distance_m':float(np.linalg.norm(dq,axis=-1).max()),'velocity_rmse_m_s':float(np.sqrt(np.mean(dv**2)))}
    iteration_pairs=[];translation_pairs=[]
    for name in loaded:
        for low,high in zip(config['iterations'][:-1],config['iterations'][1:]):
            for action in ['zero_force','evaluation']:
                iteration_pairs.append({'condition':name,'iteration_counts':[low,high],'action_id':action,**difference(loaded[name][low][action],loaded[name][high][action])})
    for count in config['iterations']:
        for action in ['zero_force','evaluation']:
            translation_pairs.append({'iterations':count,'action_id':action,**difference(loaded['original'][count][action],loaded['recentered'][count][action])})
    replay={};old=Path('runs/gaussian-campaign-v01/numerical-timestep-v01/hz960/captures/private/t0001')
    for action in ['zero_force','evaluation']:
        with np.load(old/action/'trajectory.npz') as raw:
            replay[action]={k:bool(np.array_equal(raw[k],loaded['original'][32][action][k])) for k in ['positions_m','velocities_m_s','force_step_commands_N']}
    result={'cells':rows,'iteration_pairs':iteration_pairs,'translation_pairs':translation_pairs,
        'unshifted32_replay_bitwise':replay,'input_sha256':hashes,'source_sha256':sha(__file__),
        'qualification':'Predeclared one-layout numerical factorial; no four-BC gate or continuum-accuracy claim. Read actual-solver-state.json for softbody,rigid andscene values; rigid solver counts unchanged.'}
    atomic(output/'analysis.json',result);return result


def driver(args,config):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);prepare(output,config)
    base=Path(__file__).with_name('isaac_calibrated_probes_v03.py')
    if sha(base)!=config['base_worker_sha256']:raise ValueError('Base changed')
    protocol={'config':config,'config_sha256':sha(args.config),'wrapper_sha256':sha(__file__),'base_worker_sha256':sha(base)}
    checked_write(output/'protocol.json',(json.dumps(protocol,indent=2)+'\n').encode())
    for p in [Path(__file__),Path(args.config),Path(__file__).with_name('campaign_numerical_validate_rates.py')]:checked_write(output/'provenance'/p.name,p.read_bytes())
    if not args.launch:return {'prepared':True,'simulation_launched':False}
    lock=(output/'.driver.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'execution.json').exists():raise FileExistsError('Preserve existing factorial')
    ledger={'driver_pid':os.getpid(),'cells':[],'completed':False};mutex=threading.Lock()
    def save():atomic(output/'execution.json',ledger)
    def group(condition):
        for count in config['iterations']:
            if sha(__file__)!=protocol['wrapper_sha256']:raise ValueError('Source changed')
            root=output/condition['name']/f'i{count}';root.mkdir()
            command=[config['isaac_python'],'-u',str(Path(__file__)),'--config',str(args.config),'--worker',
                '--condition',condition['name'],'--iterations',str(count),'--dataset',str(output/condition['name']/'staged-dataset'),'--captures',str(root/'captures')]
            row={'condition':condition['name'],'iterations':count,'gpu':condition['gpu'],'state':'running','command':command,'started_unix_s':time.time()}
            with (root/'worker.log').open('x') as log:
                p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
                with mutex:row['pid']=p.pid;ledger['cells'].append(row);save()
                code=p.wait()
            with mutex:row.update(exit_code=code,state='captured' if code==0 else 'failed',finished_unix_s=time.time());save()
    with ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(group,config['coordinate_conditions']))
    # CPU validation mutates only this process's imported geometry callable;
    # serialize it after all Isaac children exit to prevent cross-cell races.
    for condition in config['coordinate_conditions']:
        for count in config['iterations']:
            row=next(r for r in ledger['cells'] if r['condition']==condition['name'] and r['iterations']==count)
            if row['exit_code']==0:
                try:validate_cell(output,config,condition,count);row['state']='completed_verified'
                except Exception as e:row.update(state='failed_validation',error=repr(e))
            save()
    ledger['completed']=all(r['state']=='completed_verified' for r in ledger['cells']);save()
    if ledger['completed']:analyze(output,config)
    return ledger


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign_numerical_solver_coordinates_config.json'))
    p.add_argument('--output',default='runs/gaussian-campaign-v01/numerical-solver-coordinates-v02');p.add_argument('--launch',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--condition');p.add_argument('--iterations',type=int)
    p.add_argument('--dataset');p.add_argument('--captures')
    args=p.parse_args();config=json.loads(args.config.read_text())
    if args.worker:raise SystemExit(worker(args,config))
    print(json.dumps(driver(args,config),indent=2))
