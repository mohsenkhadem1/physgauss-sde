"""Two predeclared PGS-request64 cells compared with frozen PGS-request32 pair."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
try:
    import campaign_numerical_matched_rigid as matched
    from campaign_numerical_pgs_attempt import solver_log_evidence
    from campaign_numerical_convergence import atomic,checked_write,sha,command_clock_audit
    from campaign_numerical_force_polynomial import finite_strain
except ImportError:
    from experiments import campaign_numerical_matched_rigid as matched
    from experiments.campaign_numerical_pgs_attempt import solver_log_evidence
    from experiments.campaign_numerical_convergence import atomic,checked_write,sha,command_clock_audit
    from experiments.campaign_numerical_force_polynomial import finite_strain


def reference_private(name):
    run='numerical-pgs-attempt-v01' if name=='original' else 'numerical-pgs-recenter-v01'
    return Path('runs/gaussian-campaign-v01')/run/name/'i32/captures/private/t0001'


def pair_metrics(high,low,rest):
    # Recentered coordinates differ by the known world translation only.
    shift=np.array([0.,0.,.25]);old=low['original']-(low['recentered']+shift)
    new=high['original']-(high['recentered']+shift)
    old_error=float(np.sqrt(np.mean(old**2)));new_error=float(np.sqrt(np.mean(new**2)))
    signal=max(float(np.sqrt(np.mean((high[n]-rest[n])**2))) for n in high)
    tolerance=max(2e-6,.05*signal)
    iterations=[]
    for name in high:
        difference=float(np.sqrt(np.mean((high[name]-low[name])**2)))
        own_tolerance=max(2e-6,.05*float(np.sqrt(np.mean((high[name]-rest[name])**2))))
        iterations.append({'condition':name,'iterations':[32,64],'coordinate_rmse_m':difference,
            'tolerance_m':own_tolerance,'below_tolerance':bool(difference<own_tolerance)})
    return {'translation_32_coordinate_rmse_m':old_error,'translation_64_coordinate_rmse_m':new_error,
        'translation_error_ratio_64_over_32':new_error/old_error if old_error else None,
        'translation_error_decreased':bool(new_error<old_error),'translation_tolerance_m':tolerance,
        'translation_64_gate_passed':bool(new_error<tolerance),'iteration_pairs':iterations,
        'both_iteration_differences_below_tolerance':all(r['below_tolerance'] for r in iterations),
        'qualification':'Only one32-to64 iteration difference per coordinate condition; this cannot establish contraction of iteration differences or convergence.'}


def worker(args,config):
    import isaac_calibrated_probes_v03 as base
    if sha(matched.__file__)!=config['matched_wrapper_sha256']:raise ValueError('Frozen matched wrapper changed')
    original=base.create_environment
    def environment(stage,case,root):
        if not stage.GetPrimAtPath('/physicsScene').GetAttribute('physxScene:solverType').Set('PGS'):raise ValueError('PGS request refused')
        return original(stage,case,root)
    base.create_environment=environment
    return matched.worker(argparse.Namespace(condition=args.condition,iterations=64,dataset=args.dataset,captures=args.captures),config)


def analyze(output,config,ledger):
    output=Path(output);rows=[];high={};low={};rest={}
    for cell in ledger['cells']:
        name=cell['condition'];root=output/name/'i64';private=root/'captures/private/t0001'
        result={k:cell[k] for k in ['condition','state','exit_code']}
        result['solver_related_failure_or_fallback_logs']=solver_log_evidence((root/'worker.log').read_text()) if (root/'worker.log').exists() else []
        failure=private/'native-material-or-rest-failure.json'
        if failure.exists():result['native_runtime_failure']=json.loads(failure.read_text())
        if cell['state']=='completed_verified':
            native=json.loads((private/'native-runtime-state.json').read_text())
            if set(native)!=set(['zero_force','evaluation','reset_replay']):raise ValueError('Missing native runtime checks')
            meta=json.loads((private/'actual-solver-state.json').read_text())
            if meta['prims']['scene']['attributes']['physxScene:solverType']['value']!='PGS':raise ValueError('Wrong scene solver request')
            for prim in ['deformable','handle0','handle1']:
                prefix='physxDeformableBody:' if prim=='deformable' else 'physxRigidBody:'
                if meta['prims'][prim]['attributes'][prefix+'solverPositionIterationCount']['value']!=64:raise ValueError('Wrong solver count')
            with np.load(private/'reference.npz') as a:rest[name]=a['rest_points_m'];tets=a['tetrahedra']
            ref=reference_private(name)
            if not json.loads((ref/'validation.json').read_text()).get('passed'):raise ValueError('Unvalidated old PGS32 reference')
            result.update(native_material_and_rest_verified_all_branches=True,actions=[])
            for action in ['zero_force','evaluation']:
                path=private/action/'trajectory.npz'
                with np.load(path) as a:d={k:a[k].copy() for k in a.files}
                q=d['positions_m'].astype(float)
                with np.load(ref/action/'trajectory.npz') as a:r=a['positions_m'].astype(float)
                if action=='evaluation':high[name]=q;low[name]=r
                result['actions'].append({'action_id':action,'coordinate_rms_from_rest_m':float(np.sqrt(np.mean((q-rest[name])**2))),
                    'difference_from_PGS32_coordinate_rmse_m':float(np.sqrt(np.mean((q-r)**2))),
                    'strain':finite_strain(rest[name],tets,q),'trajectory_sha256':sha(path),
                    'PGS32_reference_sha256':sha(ref/action/'trajectory.npz'),
                    'command_clock':command_clock_audit(d,960,meta,[0,0,0] if action=='zero_force' else [.01,.01,.005])})
        rows.append(result)
    result={'cells':rows,'native_reference_qualified':False,'qualification':config['qualification'],
        'native_solver_type_getter_available':False,'actual_solver_identity':'RequestedPGS and successful capture do not prove internal deformable solver identity.'}
    if set(high)==set(['original','recentered']):result['paired_evaluation_metrics']=pair_metrics(high,low,rest)
    else:result['paired_evaluation_metrics']={'translation_64_gate_passed':False,'qualification':'Both64cells must complete full native/runtime/sensor validation; no subset can qualify.'}
    atomic(output/'analysis.json',result);return result


def driver(args,config):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if config['iterations']!=[64] or [c['name'] for c in config['coordinate_conditions']]!=['original','recentered']:raise ValueError('Frozen two-cell design required')
    if sha(matched.__file__)!=config['matched_wrapper_sha256']:raise ValueError('Matched wrapper changed')
    matched.frozen.prepare(output,config)
    protocol={'config':config,'config_sha256':sha(args.config),'wrapper_sha256':sha(__file__)}
    checked_write(output/'protocol.json',(json.dumps(protocol,indent=2)+'\n').encode())
    for p in [Path(__file__),Path(args.config),Path(matched.__file__)]:checked_write(output/'provenance'/p.name,p.read_bytes())
    if not args.launch:return {'prepared':True,'simulation_launched':False}
    if (output/'execution.json').exists():raise FileExistsError('Preserve existing execution')
    ledger={'driver_pid':os.getpid(),'cells':[],'terminal':False,'completed':False};stop=False
    for condition in config['coordinate_conditions']:
        if sha(__file__)!=protocol['wrapper_sha256']:raise ValueError('Source changed')
        name=condition['name'];root=output/name/'i64';root.mkdir()
        row={'condition':name,'iterations':64,'gpu':condition['gpu'],'started_unix_s':time.time()}
        if stop:
            row.update(state='aborted_after_runtime_verification_failure',exit_code=None);ledger['cells'].append(row);atomic(output/'execution.json',ledger);continue
        command=[config['isaac_python'],'-u',str(Path(__file__)),'--config',str(args.config),'--worker',
            '--condition',name,'--dataset',str(output/name/'staged-dataset'),'--captures',str(root/'captures')]
        row['command']=command
        with (root/'worker.log').open('x') as log:
            p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
            row.update(pid=p.pid,state='running');ledger['cells'].append(row);atomic(output/'execution.json',ledger);code=p.wait()
        row.update(exit_code=code,state='captured' if code==0 else 'failed',finished_unix_s=time.time())
        if (root/'captures/private/t0001/native-material-or-rest-failure.json').exists():stop=True
        atomic(output/'execution.json',ledger)
        if code==0:
            try:matched.frozen.validate_cell(output,config,condition,64);row['state']='completed_verified'
            except Exception as error:row.update(state='failed_validation',error=repr(error))
        atomic(output/'execution.json',ledger)
    ledger.update(terminal=True,completed=all(r['state']=='completed_verified' for r in ledger['cells']));atomic(output/'execution.json',ledger)
    result=analyze(output,config,ledger)
    return {'terminal':True,'completed':ledger['completed'],'paired_evaluation_metrics':result['paired_evaluation_metrics']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign_numerical_pgs_iterations_config.json'))
    p.add_argument('--output',default='runs/gaussian-campaign-v01/numerical-pgs-iterations-v01');p.add_argument('--launch',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--condition');p.add_argument('--dataset');p.add_argument('--captures')
    args=p.parse_args();config=json.loads(args.config.read_text())
    if args.worker:raise SystemExit(worker(args,config))
    print(json.dumps(driver(args,config),indent=2))
