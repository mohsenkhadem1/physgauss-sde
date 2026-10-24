"""Frozen membership and observation replay for ALL original importance failures.

No new observations or action selection. Cases are diagnostic failures, not a
random object population or policy comparison. Old negatives stay immutable.
"""
import numpy as np
from experiments import run_postpoke_refresh_v02 as old


def failed_cases(all_rows,noise_seeds):
    """Require original complete12x6x16 grid before selecting its false flags."""
    if len(all_rows)!=12 or len(noise_seeds)!=12:raise ValueError('All12 original gatefiles')
    cases=[]
    for index,rows in enumerate(all_rows):
        seeds=noise_seeds[index]
        expected={(action,seed)for action in range(6)for seed in seeds}
        actual=[(row['action'],row['noise_seed'])for row in rows]
        if len(seeds)!=16 or len(set(seeds))!=16 or len(rows)!=96 or len(set(actual))!=96 or set(actual)!=expected:raise ValueError('Complete unique original6x16 gategrid required')
        for row in rows:
            passed=row['importance']['importance_gate_passed']
            if not isinstance(passed,bool):raise ValueError('Original boolean importancegate required')
            if not passed:cases.append(dict(object_index=index,action=row['action'],noise_seed=row['noise_seed'],noise_index=seeds.index(row['noise_seed'])))
    cases.sort(key=lambda r:(r['object_index'],r['action'],r['noise_seed']))
    counts={index:sum(r['object_index']==index for r in cases)for index in range(12)}
    if len(cases)!=9 or {i:n for i,n in counts.items()if n}!={2:3,4:1,8:3,10:2}:raise ValueError('Exactlyall9 originalimportance failures required')
    return [dict(case_index=i,**r)for i,r in enumerate(cases)]


def replay_observation(model,prefix_observation,truth_unit,case,old_target,old_pokes,cached_means,cached_variances):
    """Parametric old case replay, including action*16+actualnoiseindex equality."""
    action=case['action'];noise_index=case['noise_index'];seed=case['noise_seed']
    if not 0<=action<6 or not 0<=noise_index<16:raise ValueError('Originalaction andnoiseindex')
    force=list(model.cohort.old.poke_programs().values())[action]
    replay=model.JointObservationForward(force,prefix_observation,np.zeros(12))
    noiseless=replay.predictions(np.asarray(truth_unit)[None])[0,1]
    observed=noiseless+np.random.default_rng(seed).normal(0,model.cohort.old.original.SIGMA,12)
    weights,importance=model.cohort.old.exact_reweight(old_pokes[action],observed)
    means,var,*_=old.previous.old.streamed_updates(old_target,weights[:,None])
    slot=action*16+noise_index;reference_mean=np.asarray(cached_means[slot]);reference_var=np.asarray(cached_variances[slot])
    dm=float(np.max(abs(means[0]-reference_mean)));dv=float(np.max(abs(var[0]-reference_var)))
    if dm>1e-12 or dv>1e-18:raise ValueError('Originalbad-poke replay failed cachedweightedmoments equality')
    if importance['importance_gate_passed']:raise ValueError('Frozen originalfailure must reproduceimportanceFAIL')
    return observed,force,dict(action=action,noise_seed=seed,noise_index=noise_index,replayed_original_observation=True,maximum_old_mean_reproduction_difference_m=dm,maximum_old_variance_reproduction_difference_m2=dv,old_importance_diagnostic=importance),reference_mean,reference_var


def terminal_count_gate(execution,receipt,summary,digest,absent):
    records=execution.get('records',[])
    if not(execution.get('completed')and execution.get('all_owned_workers_reaped')and execution.get('protocol_sha256')==digest and len(records)==12 and {r['object_index']for r in records}==set(range(12))and all(r.get('reaped')and r.get('exit_code')==0 for r in records)):raise ValueError('All12 countworkers must complete/reap successfully')
    pids={r['pid']for r in records}
    if not(receipt.get('completed')and receipt.get('exit_code')==0 and receipt.get('controller_reaped')and receipt.get('controller_PID_absent')and receipt.get('all_observed_owned_worker_PIDs_absent')and receipt.get('controller_pid')==execution.get('pid')and len(receipt.get('observed_owned_worker_PIDs',[]))==12 and set(receipt['observed_owned_worker_PIDs'])==pids):raise ValueError('Exact complete countownership receipt required')
    if not(summary.get('completed')and summary.get('objects')==12 and summary.get('all12_retained')):raise ValueError('Preserveall12 countoutcomes includingnegatives')
    for pid in [receipt['supervisor_pid'],receipt['controller_pid']]+receipt['observed_owned_worker_PIDs']:absent(pid)
    return True
