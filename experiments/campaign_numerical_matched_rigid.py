"""TRAIN t1 native-material and matched rigid/deformable solver-count audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import fcntl
import numpy as np
try:
    import campaign_numerical_solver_coordinates as frozen
    from campaign_numerical_convergence import atomic,checked_write,sha,command_clock_audit
    from campaign_numerical_force_polynomial import finite_strain
except ImportError:
    from experiments import campaign_numerical_solver_coordinates as frozen
    from experiments.campaign_numerical_convergence import atomic,checked_write,sha,command_clock_audit
    from experiments.campaign_numerical_force_polynomial import finite_strain


def verify_native_material(youngs_modulus,poissons_ratio,material):
    result={}
    for name,value,key in [('youngs_modulus_Pa',youngs_modulus,'youngs_modulus_Pa'),('poissons_ratio',poissons_ratio,'poissons_ratio')]:
        a=np.asarray(value)
        if a.shape!=(1,1) or not np.isfinite(a).all():raise ValueError('Native material tensor shape/value invalid: '+name)
        expected=np.float32(material[key]);tolerance=2*float(abs(np.spacing(expected)))
        if abs(float(a[0,0])-float(expected))>tolerance:raise ValueError('Native material differs from authored float32: '+name)
        result[name]={'native':float(a[0,0]),'authored_float32':float(expected),'absolute_tolerance':tolerance}
    return result


def verify_native_rest(native_points,native_tets,points,tets):
    p=np.asarray(native_points);t=np.asarray(native_tets)
    if p.shape!=(1,len(points),3) or t.shape!=(1,len(tets),4):raise ValueError('Unexpected native rest-buffer shape')
    np.testing.assert_array_equal(p[0],np.asarray(points,dtype=np.float32))
    # Element ordering can change in cooking; vertex labels/order per element
    # must preserve the same oriented topology, not only an unoriented set.
    def canonical(a):return np.array(sorted(map(tuple,np.asarray(a).tolist())),dtype=np.int64)
    np.testing.assert_array_equal(canonical(t[0]),canonical(tets))
    return {'points_bitwise_match_authored_float32':True,'oriented_tet_multiset_matches':True,
        'native_nodal_mass_available':False,'native_mass_qualification':'No mass/density getter is exposed by this installed DeformableBodyView.'}


def worker(args,config):
    import isaac_calibrated_probes_v03 as base
    if sha(frozen.__file__)!=config['coordinate_wrapper_sha256']:raise ValueError('Frozen coordinate wrapper changed')
    if args.iterations not in config['iterations']:raise ValueError('Undeclared matched count')
    original_environment=base.create_environment;native_records={};root_path=None
    def environment(stage,case,root):
        nonlocal root_path
        from isaacsim.core.api import World
        root_path=Path(root);env=original_environment(stage,case,root)
        for i in range(2):
            prim=stage.GetPrimAtPath(f'/World/ProbeObject/Load_{i}')
            if not prim.GetAttribute('physxRigidBody:solverPositionIterationCount').Set(args.iterations):raise ValueError('Rigid count refused')
            if not prim.GetAttribute('physxRigidBody:solverVelocityIterationCount').Set(1):raise ValueError('Rigid velocity count refused')
        original_render=World.render
        def render(world,*a,**kw):
            from isaacsim.core.simulation_manager import SimulationManager
            frame=inspect.currentframe().f_back
            try:
                while frame is not None and frame.f_code is not base.run_worker.__code__:frame=frame.f_back
                if frame is not None:
                    action=frame.f_locals.get('action_id');body=frame.f_locals.get('body');handles=frame.f_locals.get('handles')
                    if body is not None and handles is not None and action not in native_records:
                        record={}
                        try:
                            sim=SimulationManager.get_physics_simulation_view()
                            mv=sim.create_deformable_material_view('/World/ProbeObject/Material')
                            if mv.count!=1:raise ValueError('Native material view count must equal1')
                            record={'material':verify_native_material(mv.get_youngs_modulus().numpy(),mv.get_poissons_ratio().numpy(),case['material'])}
                            view=body._physics_deformable_body_view
                            rest=view.get_rest_nodal_positions().numpy().copy();tets=view.get_rest_element_indices().numpy().copy()
                            np.savez_compressed(root_path/f'native-rest-{action}.npz',rest_points=rest,rest_tets=tets)
                            record['rest_buffers']=verify_native_rest(rest,tets,env['points'],env['tets'])
                            rv=handles._physics_rigid_body_view
                            record['rigid_native_masses_kg']=rv.get_masses().numpy().tolist()
                            record['rigid_native_inertias_kg_m2']=rv.get_inertias().numpy().tolist()
                            record['native_read_semantics']='Native tensor backend getters; high-level USD material getters and recomputed stresses are not used.'
                            native_records[action]=record;atomic(root_path/'native-runtime-state.json',native_records)
                        except Exception as error:
                            atomic(root_path/'native-material-or-rest-failure.json',{'action_id':action,'error':repr(error),'completed_native_checks':record,
                                'qualification':'Runtime verification failed or unsupported; stop further declared cells rather than assume native material/rest correctness.'})
                            raise
            finally:del frame
            return original_render(world,*a,**kw)
        World.render=render
        return env
    base.create_environment=environment
    return frozen.worker(argparse.Namespace(condition=args.condition,iterations=args.iterations,
        dataset=args.dataset,captures=args.captures),config)


def analyze(output,config,ledger):
    output=Path(output);rows=[];loaded={}
    for row in ledger['cells']:
        name,count=row['condition'],row['iterations'];private=output/name/f'i{count}'/'captures/private/t0001'
        result={k:row[k] for k in ['condition','iterations','state','exit_code']}
        if row['state']!='completed_verified':rows.append(result);continue
        state=json.loads((private/'native-runtime-state.json').read_text())
        if set(state)!=set(['zero_force','evaluation','reset_replay']):raise ValueError('Missing native-state reset check')
        with np.load(private/'reference.npz') as a:rest,tets=a['rest_points_m'],a['tetrahedra']
        meta=json.loads((private/'actual-solver-state.json').read_text())
        for h in ['handle0','handle1']:
            attrs=meta['prims'][h]['attributes']
            if attrs['physxRigidBody:solverPositionIterationCount']['value']!=count or attrs['physxRigidBody:solverVelocityIterationCount']['value']!=1:raise ValueError('Rigid authoring mismatch')
        old=Path('runs/gaussian-campaign-v01/numerical-solver-coordinates-v02')/name/f'i{count}/captures/private/t0001'
        result.update(native_state_verified=True,native_state_sha256=sha(private/'native-runtime-state.json'),actions=[])
        for action in ['zero_force','evaluation']:
            f=private/action/'trajectory.npz'
            with np.load(f) as a:d={k:a[k].copy() for k in a.files}
            q=d['positions_m'].astype(float);loaded[name,count,action]=q
            record={'action_id':action,'coordinate_rms_from_rest_m':float(np.sqrt(np.mean((q-rest)**2))),
                'strain':finite_strain(rest,tets,q),'trajectory_sha256':sha(f),
                'command_clock':command_clock_audit(d,960,meta,[0,0,0] if action=='zero_force' else [.01,.01,.005])}
            with np.load(old/action/'trajectory.npz') as a:
                delta=q-a['positions_m'].astype(float);record['difference_from_frozen_rigid16_same_deform_count_rmse_m']=float(np.sqrt(np.mean(delta**2)))
            result['actions'].append(record)
        rows.append(result)
    translations=[];iterations=[]
    for count in config['iterations']:
        for action in ['zero_force','evaluation']:
            if all((n,count,action) in loaded for n in ['original','recentered']):
                d=loaded['original',count,action]-(loaded['recentered',count,action]+np.array([0,0,.25]))
                translations.append({'iterations':count,'action_id':action,'coordinate_rmse_m':float(np.sqrt(np.mean(d*d)))})
    for condition in config['coordinate_conditions']:
        n=condition['name']
        for action in ['zero_force','evaluation']:
            if all((n,c,action) in loaded for c in config['iterations']):
                d=loaded[n,32,action]-loaded[n,64,action]
                iterations.append({'condition':n,'iterations':[32,64],'action_id':action,'coordinate_rmse_m':float(np.sqrt(np.mean(d*d)))})
    result={'cells':rows,'translation_pairs':translations,'iteration_pairs':iterations,
        'qualification':config['qualification'],'native_reference_qualified':False,
        'qualification_reason':'Fourcells isolate matched solver settings and native E/nu/rest verification; not timestep/four-BC convergence or calibration.'}
    atomic(output/'analysis.json',result);return result


def driver(args,config):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if config['iterations']!=[32,64] or config['episode_id']!='t0001':raise ValueError('Frozen fourcell design required')
    if sha(frozen.__file__)!=config['coordinate_wrapper_sha256']:raise ValueError('Frozen coordinate wrapper changed')
    frozen.prepare(output,config)
    protocol={'config':config,'config_sha256':sha(args.config),'wrapper_sha256':sha(__file__)}
    checked_write(output/'protocol.json',(json.dumps(protocol,indent=2)+'\n').encode())
    for p in [Path(__file__),Path(args.config),Path(frozen.__file__)]:checked_write(output/'provenance'/p.name,p.read_bytes())
    if not args.launch:return {'prepared':True,'simulation_launched':False}
    lock=(output/'.driver.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'execution.json').exists():raise FileExistsError('Existing execution preserved')
    ledger={'driver_pid':os.getpid(),'cells':[],'terminal':False,'completed':False};mutex=threading.Lock();stop=threading.Event()
    def save():atomic(output/'execution.json',ledger)
    def group(condition):
        for count in config['iterations']:
            if sha(__file__)!=protocol['wrapper_sha256']:raise ValueError('Source changed')
            root=output/condition['name']/f'i{count}';root.mkdir()
            command=[config['isaac_python'],'-u',str(Path(__file__)),'--config',str(args.config),'--worker',
                '--condition',condition['name'],'--iterations',str(count),'--dataset',str(output/condition['name']/'staged-dataset'),'--captures',str(root/'captures')]
            row={'condition':condition['name'],'iterations':count,'gpu':condition['gpu'],'command':command,'started_unix_s':time.time()}
            if stop.is_set():
                with mutex:row.update(state='aborted_after_runtime_verification_failure',exit_code=None);ledger['cells'].append(row);save()
                continue
            with (root/'worker.log').open('x') as log:
                p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
                with mutex:row.update(pid=p.pid,state='running');ledger['cells'].append(row);save()
                code=p.wait()
            if (root/'captures/private/t0001/native-material-or-rest-failure.json').exists():stop.set()
            with mutex:row.update(exit_code=code,state='captured' if code==0 else 'failed',finished_unix_s=time.time());save()
    with ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(group,config['coordinate_conditions']))
    for condition in config['coordinate_conditions']:
        for count in config['iterations']:
            row=next(r for r in ledger['cells'] if r['condition']==condition['name'] and r['iterations']==count)
            if row['exit_code']==0:
                try:frozen.validate_cell(output,config,condition,count);row['state']='completed_verified'
                except Exception as error:row.update(state='failed_validation',error=repr(error))
            save()
    ledger.update(terminal=True,completed=all(r['state']=='completed_verified' for r in ledger['cells']));save()
    analyze(output,config,ledger)
    return {'terminal':True,'completed':ledger['completed'],'cells':[{k:r[k] for k in ['condition','iterations','state','exit_code']} for r in ledger['cells']]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign_numerical_matched_rigid_config.json'))
    p.add_argument('--output',default='runs/gaussian-campaign-v01/numerical-matched-rigid-v01');p.add_argument('--launch',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--condition');p.add_argument('--iterations',type=int)
    p.add_argument('--dataset');p.add_argument('--captures')
    args=p.parse_args();config=json.loads(args.config.read_text())
    if args.worker:raise SystemExit(worker(args,config))
    print(json.dumps(driver(args,config),indent=2))
