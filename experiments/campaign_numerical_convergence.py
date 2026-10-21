"""Standalone timestep audit wrapping, never editing, the verified v0.3 worker."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def atomic(path, value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def checked_write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Frozen staged input changed: '+str(path))
    else:path.write_bytes(data)


def stage_case(dataset, output, episode_id, hz):
    if episode_id!='t0001' or hz not in [480,960,1920]:
        raise ValueError('Only declared TRAIN t0001/three rates allowed')
    original=Path(dataset)/'episodes'/episode_id
    recipe_path=original/'truth/recipe.json'
    case=json.loads(recipe_path.read_text())
    if case['split']!='train' or case['boundary_type']!='fixed_patch' or case['physics_hz']!=960:
        raise ValueError('Original diagnostic case changed')
    modified=copy.deepcopy(case);modified['physics_hz']=hz;modified.pop('recipe_sha256')
    modified['recipe_sha256']=digest(modified)
    root=Path(output)/f'hz{hz}'/'staged-dataset';target=root/'episodes'/episode_id
    checked_write(target/'truth/recipe.json',(json.dumps(modified,indent=2)+'\n').encode())
    unchanged=['truth/boundaries.json','observations/cameras.npz']
    for name in unchanged:checked_write(target/name,(original/name).read_bytes())
    lineage={'original_recipe_sha256':sha(recipe_path),'original_embedded_recipe_sha256':case['recipe_sha256'],
        'staged_recipe_sha256':sha(target/'truth/recipe.json'),'change':{'physics_hz':[960,hz]},
        'unchanged_inputs_sha256':{name:sha(original/name) for name in unchanged},
        'role':'TRAIN numerical audit only; not fitting data'}
    checked_write(root/'lineage.json',(json.dumps(lineage,indent=2)+'\n').encode())
    return root


def worker(args, config):
    import isaac_calibrated_probes_v03 as base
    if sha(base.__file__)!=config['base_worker_sha256']:
        raise ValueError('Verified base worker changed')
    original_menu=base.calibrated_menu
    original_environment=base.create_environment
    def menu(*a,**kw):
        m=original_menu(*a,**kw)
        m['candidates']=[{**m['candidates'][0],'action_id':'zero_force',
            'kind':'numerical_audit_zero','peak_force_N':[0.,0.,0.]}]
        m['protocol']=config['protocol'];m['qualification']=config['qualification']
        return m
    def environment(stage,case,root):
        from isaacsim.core.api import World
        actual=float(World.instance().get_physics_dt())
        atomic(Path(root)/'effective-step.json',{'requested_hz':case['physics_hz'],
            'nominal_dt_s':1/case['physics_hz'],'world_get_physics_dt_s':actual,
            'float32_nominal_dt_s':float(np.float32(1/case['physics_hz'])),
            'qualification':'World getter and float32 representable nominal dt recorded separately; saved simulation clock is audited independently'})
        return original_environment(stage,case,root)
    base.calibrated_menu=menu;base.create_environment=environment
    child=argparse.Namespace(dataset=Path(args.dataset),output=Path(args.rate_output),episode='t0001',gpu=config['gpu'])
    return base.run_worker(child)


def command_clock_audit(data,hz,step_metadata,peak):
    n=round(3*hz);dt=1/hz;dt32=float(np.float32(dt))
    np.testing.assert_allclose(data['force_step_midpoints_s'],(np.arange(n)+.5)/hz,atol=1e-12,rtol=0)
    capture_steps=np.rint(data['times_s']*hz).astype(int)
    # The frozen worker and independent validator retain their original stricter
    # guards. This additional audit explicitly checks the float32-step clock.
    error=data['simulated_times_s']-capture_steps*dt32
    if abs(error).max()>1e-9:
        raise ValueError('Recorded clock differs from expected float32-dt clock')
    f=data['force_step_commands_N'].astype(np.float64)
    impulse_nominal=f.sum(0)*dt;impulse_float32=f.sum(0)*dt32
    analytic=np.zeros((2,3));analytic[0]=np.asarray(peak)*(.45*3/2)
    return {'nominal_dt_s':dt,'float32_dt_s':dt32,
        'world_get_physics_dt_s':step_metadata['world_get_physics_dt_s'],
        'clock_max_error_vs_float32_dt_s':float(abs(error).max()),
        'clock_max_error_vs_nominal_dt_s':float(abs(data['simulated_times_s']-data['times_s']).max()),
        'force_impulse_nominal_dt_Ns':impulse_nominal.tolist(),
        'force_impulse_float32_dt_Ns':impulse_float32.tolist(),
        'analytic_pulse_impulse_Ns':analytic.tolist(),
        'nominal_impulse_max_error_Ns':float(abs(impulse_nominal-analytic).max()),
        'semantics':'World-space force commanded once before each physics advance; envelope sampled at nominal interval midpoint (step-.5)/hz and held for that step. Sum uses every logged float32 command; nominal and representable-float32 dt impulses both reported.'}


def analyze(output,rates):
    output=Path(output);loaded={};rows=[];hashes={}
    from campaign_numerical_force_polynomial import finite_strain
    for hz in rates:
        root=output/f'hz{hz}'/'captures';private=root/'private/t0001'
        validation=json.loads((private/'validation.json').read_text())
        if not validation.get('passed'):raise ValueError('Unvalidated numerical rate')
        metadata=json.loads((private/'effective-step.json').read_text())
        menu=json.loads((root/'public/t0001/action-menu.json').read_text())
        with np.load(private/'reference.npz') as raw:rest,tets=raw['rest_points_m'],raw['tetrahedra']
        loaded[hz]={}
        for name in ['zero_force','evaluation']:
            path=private/name/'trajectory.npz'
            with np.load(path) as raw:data={k:raw[k].copy() for k in raw.files}
            loaded[hz][name]=data;hashes[str(path)]=sha(path)
            peak=[0.,0.,0.] if name=='zero_force' else menu['evaluation_query']['peak_force_N']
            rows.append({'physics_hz':hz,'action_id':name,
                'coordinate_rms_from_rest_m':float(np.sqrt(np.mean((data['positions_m'].astype(float)-rest)**2))),
                'velocity_rms_m_s':float(np.sqrt(np.mean(data['velocities_m_s'].astype(float)**2))),
                'strain':finite_strain(rest,tets,data['positions_m']),
                'command_clock':command_clock_audit(data,hz,metadata,peak)})
    pairs=[]
    for low,high in zip(rates[:-1],rates[1:]):
        for name in ['zero_force','evaluation']:
            a,b=loaded[low][name],loaded[high][name]
            np.testing.assert_array_equal(a['times_s'],b['times_s'])
            q=a['positions_m'].astype(float)-b['positions_m'].astype(float)
            v=a['velocities_m_s'].astype(float)-b['velocities_m_s'].astype(float)
            pairs.append({'rates_hz':[low,high],'action_id':name,
                'coordinate_rmse_m':float(np.sqrt(np.mean(q*q))),'max_nodal_distance_m':float(np.linalg.norm(q,axis=-1).max()),
                'velocity_rmse_m_s':float(np.sqrt(np.mean(v*v)))})
    old=Path('runs/gaussian-campaign-v01/linearity-diagnostic/private/t0001')
    replay={}
    if 960 in loaded and (old/'validation.json').is_file() and json.loads((old/'validation.json').read_text()).get('passed'):
        for name in ['zero_force','evaluation']:
            path=old/name/'trajectory.npz'
            with np.load(path) as raw:
                replay[name]={k:bool(np.array_equal(raw[k],loaded[960][name][k])) for k in ['positions_m','velocities_m_s','force_step_commands_N']}
            hashes[str(path)]=sha(path)
    result={'rates':rows,'refinement_pairs':pairs,'fresh960_vs_original960_bitwise':replay,
        'input_sha256':hashes,'source_sha256':sha(__file__),
        'qualification':'Three timestep levels, one fixed TRAIN layout, same1728tet mesh and32solver iterations. Time refinement alone does not establish mesh/solver convergence or material accuracy. Not fitting data; no reserved responses opened.'}
    atomic(output/'analysis.json',result);return result


def driver(args,config):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    base=Path(__file__).with_name('isaac_calibrated_probes_v03.py')
    if sha(base)!=config['base_worker_sha256']:raise ValueError('Base source changed')
    if config['episode_id']!='t0001' or config['physics_rates_hz']!=[480,960,1920]:raise ValueError('Numerical design changed')
    sources={Path(__file__).name:sha(__file__),Path(args.config).name:sha(args.config)}
    protocol={'config':config,'source_sha256':sources,'base_worker_sha256':sha(base)}
    checked_write(output/'protocol.json',(json.dumps(protocol,indent=2)+'\n').encode())
    for p in [Path(__file__),Path(args.config)]:checked_write(output/'provenance'/p.name,p.read_bytes())
    roots={hz:stage_case(config['dataset'],output,'t0001',hz) for hz in config['physics_rates_hz']}
    if not args.launch:return {'prepared':True,'simulation_launched':False}
    lock=(output/'.driver.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'execution.json').exists():raise FileExistsError('Preserve and inspect existing execution before any resume')
    ledger={'driver_pid':os.getpid(),'gpu':config['gpu'],'cases':[],'completed':False}
    atomic(output/'execution.json',ledger)
    from isaac_calibrated_probes_v03 import validate_case
    for hz,staged in roots.items():
        if sources[Path(__file__).name]!=sha(__file__) or sha(base)!=config['base_worker_sha256']:
            raise ValueError('Source changed during numerical audit')
        rate_output=output/f'hz{hz}'/'captures';log_path=output/f'hz{hz}'/'worker.log'
        command=[config['isaac_python'],'-u',str(Path(__file__)),'--config',str(args.config),'--worker',
            '--dataset',str(staged),'--rate-output',str(rate_output)]
        row={'physics_hz':hz,'command':command,'state':'running','started_unix_s':time.time()};ledger['cases'].append(row)
        with log_path.open('x') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
            row['pid']=process.pid;atomic(output/'execution.json',ledger);code=process.wait()
        row['exit_code']=code
        if code==0:
            try:
                audit=validate_case(rate_output,'t0001')
                atomic(rate_output/'private/t0001/validation.json',audit);row['state']='completed_verified'
            except Exception as error:row.update(state='failed_validation',error=repr(error))
        else:row['state']='failed'
        row['finished_unix_s']=time.time();atomic(output/'execution.json',ledger)
    ledger['completed']=all(r['state']=='completed_verified' for r in ledger['cases'])
    atomic(output/'execution.json',ledger)
    if ledger['completed']:analyze(output,config['physics_rates_hz'])
    return ledger


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign_numerical_convergence_config.json'))
    p.add_argument('--output',default='runs/gaussian-campaign-v01/numerical-timestep-v01')
    p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--dataset');p.add_argument('--rate-output')
    args=p.parse_args();config=json.loads(args.config.read_text())
    if args.worker:raise SystemExit(worker(args,config))
    print(json.dumps(driver(args,config),indent=2))
