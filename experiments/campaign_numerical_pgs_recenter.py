"""One recentered PGS-request cell, paired with the frozen original PGS request."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
try:
    import campaign_numerical_matched_rigid as matched
    from campaign_numerical_convergence import atomic,checked_write,sha
except ImportError:
    from experiments import campaign_numerical_matched_rigid as matched
    from experiments.campaign_numerical_convergence import atomic,checked_write,sha


def solver_log_evidence(text):
    rows=[]
    for line in text.splitlines():
        low=line.lower()
        if any(k in low for k in ['pgs','tgs','solver']) and any(k in low for k in ['not supported','unsupported','fallback','warning','error','failed']):
            rows.append(line[:1000])
    return rows


def worker(args,config):
    import isaac_calibrated_probes_v03 as base
    if sha(matched.__file__)!=config['matched_wrapper_sha256']:raise ValueError('Frozen matched wrapper changed')
    if sha(Path(__file__).with_name('campaign_numerical_pgs_attempt.py'))!=config['original_PGS_wrapper_sha256']:raise ValueError('Frozen original PGS source changed')
    original_environment=base.create_environment
    def environment(stage,case,root):
        attr=stage.GetPrimAtPath('/physicsScene').GetAttribute('physxScene:solverType')
        if not attr.Set('PGS'):raise ValueError('PGS request refused')
        return original_environment(stage,case,root)
    base.create_environment=environment
    return matched.worker(argparse.Namespace(condition='recentered',iterations=32,
        dataset=args.dataset,captures=args.captures),config)


def driver(args,config):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if config['iterations']!=[32] or len(config['coordinate_conditions'])!=1 or config['coordinate_conditions'][0]['name']!='recentered':raise ValueError('Only one declared PGS-request cell')
    if sha(matched.__file__)!=config['matched_wrapper_sha256']:raise ValueError('Frozen matched wrapper changed')
    if sha(Path(__file__).with_name('campaign_numerical_pgs_attempt.py'))!=config['original_PGS_wrapper_sha256']:raise ValueError('Frozen original PGS source changed')
    matched.frozen.prepare(output,config)
    protocol={'config':config,'config_sha256':sha(args.config),'wrapper_sha256':sha(__file__)}
    checked_write(output/'protocol.json',(json.dumps(protocol,indent=2)+'\n').encode())
    for p in [Path(__file__),Path(args.config),Path(matched.__file__)]:checked_write(output/'provenance'/p.name,p.read_bytes())
    if not args.launch:return {'prepared':True,'simulation_launched':False}
    if (output/'execution.json').exists():raise FileExistsError('Preserve existing attempt')
    root=output/'recentered/i32';root.mkdir();command=[config['isaac_python'],'-u',str(Path(__file__)),
        '--config',str(args.config),'--worker','--dataset',str(output/'recentered/staged-dataset'),'--captures',str(root/'captures')]
    row={'driver_pid':os.getpid(),'gpu':config['coordinate_conditions'][0]['gpu'],'state':'running',
        'requested_scene_solver_type':'PGS','command':command,'started_unix_s':time.time()}
    with (root/'worker.log').open('x') as log:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
        row['pid']=p.pid;atomic(output/'execution.json',row);code=p.wait()
    row.update(exit_code=code,state='captured' if code==0 else 'failed',finished_unix_s=time.time())
    if code==0:
        try:
            matched.frozen.validate_cell(output,config,config['coordinate_conditions'][0],32)
            row['state']='completed_verified'
        except Exception as error:row.update(state='failed_validation',error=repr(error))
    row['terminal']=True;atomic(output/'execution.json',row)
    private=root/'captures/private/t0001'
    result={'state':row['state'],'exit_code':code,'requested_scene_solver_type':'PGS',
        'native_solver_type_getter_available':False,
        'actual_solver_type_qualification':'Scene schema request and simulation success do not prove which internal solver handled the deformable. No direct native actual-solver getter is exposed.',
        'solver_related_failure_or_fallback_log_evidence':solver_log_evidence((root/'worker.log').read_text()),
        'native_reference_qualified':False,'qualification':config['qualification']}
    failure=private/'native-material-or-rest-failure.json'
    if failure.exists():result['native_runtime_failure']=json.loads(failure.read_text())
    if row['state']=='completed_verified':
        meta=json.loads((private/'actual-solver-state.json').read_text())
        result['authored_scene_solver_type']=meta['prims']['scene']['attributes']['physxScene:solverType']['value']
        native=json.loads((private/'native-runtime-state.json').read_text())
        if set(native)!=set(['zero_force','evaluation','reset_replay']):raise ValueError('Missing native runtime verification')
        result['native_material_and_rest_verified_all_branches']=True;result['actions']=[]
        reference=Path('runs/gaussian-campaign-v01/numerical-matched-rigid-v02/recentered/i32/captures/private/t0001')
        with np.load(private/'reference.npz') as a:rest=a['rest_points_m']
        for action in ['zero_force','evaluation']:
            f=private/action/'trajectory.npz'
            with np.load(f) as a:q=a['positions_m'].astype(float)
            with np.load(reference/action/'trajectory.npz') as a:r=a['positions_m'].astype(float)
            result['actions'].append({'action_id':action,'coordinate_rms_from_rest_m':float(np.sqrt(np.mean((q-rest)**2))),
                'difference_from_TGS_reference_rmse_m':float(np.sqrt(np.mean((q-r)**2))),
                'positions_bitwise_identical_to_TGS_reference':bool(np.array_equal(q,r)),
                'trajectory_sha256':sha(f)})
    if row['state']=='completed_verified':
        original_pgs=Path('runs/gaussian-campaign-v01/numerical-pgs-attempt-v01/original/i32/captures/private/t0001')
        if not json.loads((original_pgs/'validation.json').read_text()).get('passed'):raise ValueError('Original PGS pair lacks full QC')
        result['translation_pairs']=[]
        with np.load(original_pgs/'reference.npz') as a:original_rest=a['rest_points_m']
        for action in ['zero_force','evaluation']:
            with np.load(private/action/'trajectory.npz') as a:q=a['positions_m'].astype(float);v=a['velocities_m_s'].astype(float)
            with np.load(original_pgs/action/'trajectory.npz') as a:r=a['positions_m'].astype(float);rv=a['velocities_m_s'].astype(float)
            delta=q+np.array([0,0,.25])-r
            signal=max(float(np.sqrt(np.mean((q-rest)**2))),float(np.sqrt(np.mean((r-original_rest)**2))))
            tolerance=max(2e-6,.05*signal)
            error=float(np.sqrt(np.mean(delta**2)))
            result['translation_pairs'].append({'action_id':action,'coordinate_rmse_m':error,
                'maximum_nodal_distance_m':float(np.linalg.norm(delta,axis=-1).max()),
                'velocity_rmse_m_s':float(np.sqrt(np.mean((v-rv)**2))),
                'tolerance_m':tolerance,'passed':bool(error<tolerance),
                'original_pgs_trajectory_sha256':sha(original_pgs/action/'trajectory.npz')})
        result['single_translation_pair_gate_passed']=all(r['passed'] for r in result['translation_pairs'])
    atomic(output/'analysis.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign_numerical_pgs_recenter_config.json'))
    p.add_argument('--output',default='runs/gaussian-campaign-v01/numerical-pgs-recenter-v01');p.add_argument('--launch',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--dataset');p.add_argument('--captures')
    args=p.parse_args();config=json.loads(args.config.read_text())
    if args.worker:raise SystemExit(worker(args,config))
    print(json.dumps(driver(args,config),indent=2))
