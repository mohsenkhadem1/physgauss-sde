"""Predeclared TRAIN t1 low-rate/translation audit; frozen capture code reused."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import inspect
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import numpy as np

try:
    import campaign_numerical_solver_coordinates as frozen
    from campaign_numerical_convergence import atomic,checked_write,sha,digest,command_clock_audit
    from campaign_numerical_validate_rates import adapt_reset_reference
    from campaign_numerical_force_polynomial import finite_strain
except ImportError:
    from experiments import campaign_numerical_solver_coordinates as frozen
    from experiments.campaign_numerical_convergence import atomic,checked_write,sha,digest,command_clock_audit
    from experiments.campaign_numerical_validate_rates import adapt_reset_reference
    from experiments.campaign_numerical_force_polynomial import finite_strain


def check_design(config):
    if config['episode_id']!='t0001' or config['physics_rates_hz']!=[60,120,240] or config['iterations']!=[32]:
        raise ValueError('Frozen low-rate design required')
    if [(c['name'],c['translation_m']) for c in config['coordinate_conditions']] != [('original',[0.,0.,0.]),('recentered',[0.,0.,-.25])]:
        raise ValueError('Both declared coordinate conditions required')
    if sha(frozen.__file__)!=config['coordinate_wrapper_sha256']:
        raise ValueError('Frozen coordinate wrapper changed')
    base=Path(__file__).with_name('isaac_calibrated_probes_v03.py')
    if sha(base)!=config['base_worker_sha256']:raise ValueError('Frozen base changed')


def install_geometry(config,condition,hz):
    import isaac_dataset
    original=isaac_dataset.geometry
    def geometry(case):
        if case['episode_id']!='t0001' or case['split']!='train' or case['physics_hz']!=hz:
            raise ValueError('Wrong numerical role/rate')
        points,tets,faces,spec=original(case)
        if not np.any(condition['translation_m']):return points,tets,faces,spec
        points,spec,_=frozen.translate_reference(points,spec,np.eye(4),condition['translation_m'])
        return points,tets,faces,spec
    isaac_dataset.geometry=geometry
    return isaac_dataset,original


def prepare(output,config):
    check_design(config);original=Path(config['dataset'])/'episodes/t0001'
    case=json.loads((original/'truth/recipe.json').read_text())
    if (case['split'],case['boundary_type'],case['physics_hz'])!=('train','fixed_patch',960):
        raise ValueError('TRAIN source changed')
    spec=json.loads((original/'truth/boundaries.json').read_text())
    with np.load(original/'reference.npz') as raw:points=raw['rest_points_m']
    with np.load(original/'observations/cameras.npz') as raw:camera={k:raw[k].copy() for k in raw.files}
    for condition in config['coordinate_conditions']:
        for hz in config['physics_rates_hz']:
            root=Path(output)/condition['name']/f'hz{hz}'/'staged-dataset';ep=root/'episodes/t0001'
            recipe=copy.deepcopy(case);recipe['physics_hz']=hz;recipe.pop('recipe_sha256');recipe['recipe_sha256']=digest(recipe)
            p,s,pose=frozen.translate_reference(points,spec,camera['camera_to_world_usd_row'],condition['translation_m'])
            checked_write(ep/'truth/recipe.json',(json.dumps(recipe,indent=2)+'\n').encode())
            if not np.any(condition['translation_m']):
                checked_write(ep/'truth/boundaries.json',(original/'truth/boundaries.json').read_bytes())
                checked_write(ep/'observations/cameras.npz',(original/'observations/cameras.npz').read_bytes())
            else:
                checked_write(ep/'truth/boundaries.json',(json.dumps(s,indent=2)+'\n').encode())
                target=ep/'observations/cameras.npz';target.parent.mkdir(parents=True,exist_ok=True)
                arrays={**camera,'camera_to_world_usd_row':pose}
                if target.exists():
                    with np.load(target) as old:
                        for k,v in arrays.items():np.testing.assert_array_equal(old[k],v)
                else:np.savez_compressed(target,**arrays)
            checked_write(root/'lineage.json',(json.dumps({'physics_hz':[960,hz],
                'translation_m':condition['translation_m'],'original_recipe_sha256':sha(original/'truth/recipe.json'),
                'original_boundary_sha256':sha(original/'truth/boundaries.json'),
                'staged_recipe_sha256':sha(ep/'truth/recipe.json'),
                'staged_boundary_sha256':sha(ep/'truth/boundaries.json'),
                'staged_camera_sha256':sha(ep/'observations/cameras.npz')},indent=2)+'\n').encode())


def worker(args,config):
    check_design(config)
    condition=next(c for c in config['coordinate_conditions'] if c['name']==args.condition)
    if args.hz not in config['physics_rates_hz']:raise ValueError('Undeclared rate')
    frozen.install_geometry=lambda c,k:install_geometry(c,k,args.hz)
    original_snapshot=frozen.solver_snapshot
    def snapshot(stage,world,count):
        value=original_snapshot(stage,world,count)
        value.update(requested_physics_hz=args.hz,nominal_dt_s=1/args.hz,
            float32_nominal_dt_s=float(np.float32(1/args.hz)))
        return value
    frozen.solver_snapshot=snapshot
    return frozen.worker(argparse.Namespace(condition=args.condition,iterations=32,
        dataset=args.dataset,captures=args.captures),config)


def validate_cell(output,config,condition,hz):
    import isaac_calibrated_probes_v03 as base
    module,original=install_geometry(config,condition,hz)
    try:
        root=Path(output)/condition['name']/f'hz{hz}'/'captures'
        source=adapt_reset_reference(inspect.getsource(base.validate_case),'zero_force')
        path=Path(output)/'provenance/adapted_validate_case.py';checked_write(path,source.encode())
        scope=dict(vars(base));exec(compile(source,str(path),'exec'),scope)
        result=scope['validate_case'](root,'t0001')
        result['lowrate_wrapper_sha256']=sha(__file__);result['validator_reset_adapter_sha256']=sha(path)
        atomic(root/'private/t0001/validation.json',result)
    finally:module.geometry=original


def analyze(output,config,ledger):
    output=Path(output);loaded={};rows=[]
    for condition in config['coordinate_conditions']:
        name=condition['name'];loaded[name]={}
        for hz in config['physics_rates_hz']:
            row=next(r for r in ledger['cells'] if r['condition']==name and r['physics_hz']==hz)
            if row['state']!='completed_verified':
                rows.append({'condition':name,'physics_hz':hz,'state':row['state'],'exit_code':row['exit_code']});continue
            private=output/name/f'hz{hz}'/'captures/private/t0001'
            with np.load(private/'reference.npz') as raw:rest,tets=raw['rest_points_m'],raw['tetrahedra']
            metadata=json.loads((private/'actual-solver-state.json').read_text());loaded[name][hz]={}
            menu=json.loads((output/name/f'hz{hz}'/'captures/public/t0001/action-menu.json').read_text())
            for action in ['zero_force','evaluation']:
                path=private/action/'trajectory.npz'
                with np.load(path) as raw:a={k:raw[k].copy() for k in raw.files}
                np.testing.assert_allclose(a['times_s'],np.linspace(0,3,31),atol=1e-12,rtol=0)
                q=a['positions_m'].astype(float);v=a['velocities_m_s'].astype(float);d=q-rest
                loaded[name][hz][action]=q-np.asarray(condition['translation_m'])
                peak=[0.,0.,0.] if action=='zero_force' else menu['evaluation_query']['peak_force_N']
                mask=a['times_s']>=1.7
                rows.append({'condition':name,'physics_hz':hz,'action_id':action,'state':row['state'],
                    'coordinate_rms_from_rest_m':float(np.sqrt(np.mean(d*d))),
                    'last_frame_coordinate_rms_from_rest_m':float(np.sqrt(np.mean(d[-1]**2))),
                    'post_pulse_coordinate_rms_from_rest_m':float(np.sqrt(np.mean(d[mask]**2))),
                    'post_pulse_velocity_squared_mean_m2_s2':float(np.mean(v[mask]**2)),
                    'potential_energy_available':False,'energy_proxy_qualification':'Unweighted velocity-squared only; not total kinetic or potential energy.',
                    'rest_float32_max_ulp_m':float(np.abs(np.spacing(rest.astype(np.float32))).max()),
                    'strain':finite_strain(rest,tets,q),'trajectory_sha256':sha(path),
                    'command_clock':command_clock_audit(a,hz,metadata,peak)})
    def difference(a,b):
        d=a-b
        return {'coordinate_rmse_m':float(np.sqrt(np.mean(d*d))),
            'maximum_nodal_distance_m':float(np.linalg.norm(d,axis=-1).max())}
    pairs=[];translations=[];assessments=[]
    for name in loaded:
        for lo,hi in zip(config['physics_rates_hz'][:-1],config['physics_rates_hz'][1:]):
            if lo not in loaded[name] or hi not in loaded[name]:continue
            for action in ['zero_force','evaluation']:
                pairs.append({'condition':name,'rates_hz':[lo,hi],'action_id':action,
                    **difference(loaded[name][lo][action],loaded[name][hi][action])})
        p=[r for r in pairs if r['condition']==name and r['action_id']=='evaluation']
        fine=next((r for r in rows if r.get('condition')==name and r.get('physics_hz')==240 and r.get('action_id')=='evaluation'),None)
        tolerance=max(2e-6,.05*fine['coordinate_rms_from_rest_m']) if fine else None
        passed=len(p)==2 and p[1]['coordinate_rmse_m']<tolerance and (p[1]['coordinate_rmse_m']<p[0]['coordinate_rmse_m'] or max(r['coordinate_rmse_m'] for r in p)<2e-6)
        assessments.append({'condition':name,'all_three_rates_available':len(p)==2,'tolerance_m':tolerance,
            'lowrate_refinement_gate_passed':bool(passed)})
    for hz in config['physics_rates_hz']:
        if all(hz in loaded[n] for n in loaded):
            for action in ['zero_force','evaluation']:
                translations.append({'physics_hz':hz,'action_id':action,
                    **difference(loaded['original'][hz][action],loaded['recentered'][hz][action])})
    highrate=Path('runs/gaussian-campaign-v01/numerical-timestep-v01/analysis.json')
    result={'rows':rows,'refinement_pairs':pairs,'translation_pairs':translations,'assessments':assessments,
        'translation_gate_passed':len(translations)==6 and all(r['coordinate_rmse_m']<config['translation_tolerance_m'] for r in translations),
        'higher_rate_negative_evidence':{'path':str(highrate),'sha256':sha(highrate)},
        'qualification':config['qualification'],'native_reference_qualified':False,
        'qualification_reason':'This six-cell audit alone cannot qualify four-BC material accuracy or erase the retained480/960/1920Hz failure.'}
    atomic(output/'analysis.json',result);return result


def driver(args,config):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);prepare(output,config)
    protocol={'config':config,'config_sha256':sha(args.config),'wrapper_sha256':sha(__file__),
        'coordinate_wrapper_sha256':sha(frozen.__file__),'common_capture_times_s':np.linspace(0,3,31).tolist()}
    checked_write(output/'protocol.json',(json.dumps(protocol,indent=2)+'\n').encode())
    for p in [Path(__file__),Path(args.config),Path(frozen.__file__)]:checked_write(output/'provenance'/p.name,p.read_bytes())
    if not args.launch:return {'prepared':True,'simulation_launched':False}
    lock=(output/'.driver.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'execution.json').exists():raise FileExistsError('Existing execution preserved')
    ledger={'driver_pid':os.getpid(),'cells':[],'terminal':False,'completed':False};mutex=threading.Lock()
    def save():atomic(output/'execution.json',ledger)
    def group(condition):
        for hz in config['physics_rates_hz']:
            if sha(__file__)!=protocol['wrapper_sha256']:raise ValueError('Source changed')
            root=output/condition['name']/f'hz{hz}'
            command=[config['isaac_python'],'-u',str(Path(__file__)),'--config',str(args.config),
                '--worker','--condition',condition['name'],'--hz',str(hz),'--dataset',str(root/'staged-dataset'),'--captures',str(root/'captures')]
            row={'condition':condition['name'],'physics_hz':hz,'gpu':condition['gpu'],'state':'running',
                'command':command,'started_unix_s':time.time()}
            with (root/'worker.log').open('x') as log:
                p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
                with mutex:row['pid']=p.pid;ledger['cells'].append(row);save()
                code=p.wait()
            with mutex:row.update(exit_code=code,state='captured' if code==0 else 'failed',finished_unix_s=time.time());save()
    with ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(group,config['coordinate_conditions']))
    for condition in config['coordinate_conditions']:
        for hz in config['physics_rates_hz']:
            row=next(r for r in ledger['cells'] if r['condition']==condition['name'] and r['physics_hz']==hz)
            if row['exit_code']==0:
                try:validate_cell(output,config,condition,hz);row['state']='completed_verified'
                except Exception as error:row.update(state='failed_validation',error=repr(error))
            save()
    ledger.update(terminal=True,completed=all(r['state']=='completed_verified' for r in ledger['cells']));save()
    analyze(output,config,ledger)
    return {'terminal':True,'completed':ledger['completed'],'cells':[{k:r[k] for k in ['condition','physics_hz','state','exit_code']} for r in ledger['cells']]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign_numerical_lowrate_config.json'))
    p.add_argument('--output',default='runs/gaussian-campaign-v01/numerical-lowrate-v01');p.add_argument('--launch',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--condition');p.add_argument('--hz',type=int)
    p.add_argument('--dataset');p.add_argument('--captures')
    args=p.parse_args();config=json.loads(args.config.read_text())
    if args.worker:raise SystemExit(worker(args,config))
    print(json.dumps(driver(args,config),indent=2))
