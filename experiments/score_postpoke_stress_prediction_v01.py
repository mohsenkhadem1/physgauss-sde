"""Post-score forecast errors on all9 known numerical-failure inputs.

Prospective metrics, no action selection, no new inference. Numerical regression
cases are not an independent population/policy comparison. Raw targets staylocal.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from experiments import campaign_diverse_force_cohort_v01 as cohort
from experiments import run_postpoke_stress_regression_v01 as stress
ROOT=Path(__file__).resolve().parents[1]
OUT='runs/operator-dev/postpoke-stress-prediction-v01'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
DESIGN=dict(cases='All9 previously frozen stress-regression cases; no subsets',
            target='Original135coordinates:3held-forcecontinuations xframes30,45,60 x5nodes x3axes, program-major',
            metrics=['Mean135coordinateRMSE, beforepoke cachedposterior','Sameoldimportance-onlyupdatedmean RMSE','NewjointMHposteriorpredictivemean RMSE'],
            aggregate='Simplemean over9knownbadinputs, plus4objectclustermeans; descriptiveonly, no populationCI orsuperiorityclaim',
            truth='IndependentNumPyfinite-model oracle, afterallstressposteriorsandownershipreceiptcommitted',
            scope='Originalsmall15DOF manufacturedmodel, notnewclampedblock/neuralforecasting/realphysicalaccuracy',
            no_new_inference=True,no_policy_comparison=True,all_numerical_failures_retained=True,raw_LOCAL=True)


def target(theta):
    return np.concatenate([cohort.old.independent_numpy_forward(theta,force)[list(cohort.TARGET_FRAMES)].reshape(-1)for force in cohort.old.force_programs().values()])


def prepare():
    p=ROOT/OUT/'protocol.json';p.parent.mkdir(parents=True,exist_ok=True)
    names=['experiments/score_postpoke_stress_prediction_v01.py','tests/test_postpoke_stress_prediction_v01.py',*stress.sources()]
    data=dict(design=DESIGN,source_sha256={n:sha(ROOT/n)for n in sorted(set(names))},status='FROZEN_METRICS_BEFORE_STRESS_OUTCOMES')
    with p.open('x')as f:json.dump(data,f,indent=2);f.write('\n')
    print(sha(p))


def run(digest):
    p=ROOT/OUT/'protocol.json';d=json.loads(p.read_text())
    if (ROOT/OUT/'scalar-results.json').exists():raise FileExistsError('Exclusivepostscore output')
    if sha(p)!=digest or d['design']!=DESIGN or d['source_sha256']!={n:sha(ROOT/n)for n in d['source_sha256']}:raise ValueError('Changed frozenpostscore source/design')
    sp=ROOT/'runs/operator-dev/postpoke-stress-regression-v01/protocol.json';s=json.loads(sp.read_text());run=ROOT/s['output'];summary=json.loads((run/'summary.json').read_text());execution=json.loads((run/'execution.json').read_text());receipt=json.loads(sp.parent.joinpath('actual-supervisor-receipt.json').read_text())
    if not(summary['completed']and summary['cases']==9 and summary['all9_retained']and execution['completed']and execution['all_owned_workers_reaped']and receipt['completed']and receipt['controller_reaped']):raise ValueError('Completeall9stressoutcomes beforetruthforward')
    records=execution['records']
    if len(records)!=9 or {r['object_index']for r in records}!=set(range(9))or not all(r['reaped']and r['exit_code']==0 for r in records)or execution['protocol_sha256']!=sha(sp)or receipt['exit_code']!=0 or receipt['controller_pid']!=execution['pid']or set(receipt['observed_owned_worker_PIDs'])!={r['pid']for r in records}:raise ValueError('Exactclosedninecaseownership')
    for pid in [receipt['supervisor_pid'],receipt['controller_pid']]+receipt['observed_owned_worker_PIDs']:stress.old.previous.old.previous.absent_process_and_group(pid)
    cp=s['original_inputs']['cohort_protocol'];cohort_path=ROOT/cp['path']
    if sha(cohort_path)!=cp['sha256']:raise ValueError('Frozen originaltruthsource')
    c=json.loads(cohort_path.read_text());truths={};rows=[];inputs={str(sp.relative_to(ROOT)):sha(sp),str((run/'summary.json').relative_to(ROOT)):sha(run/'summary.json')}
    for row in summary['per_case']:
        case=row['case'];obj=case['object_index'];idx=case['case_index']
        if obj not in truths:truths[obj]=target(cohort.physical(np.array(c['cohort_unit_coordinates_LOCAL'][obj])[None])[0])
        pp=run/f'case-{idx:02d}/joint-prediction-LOCAL.npz';oldrec=s['original_inputs'][f'object-{obj}/old_updates'];op=ROOT/oldrec['path'];oldsummaryrec=s['original_inputs'][f'object-{obj}/old_summary'];os=ROOT/oldsummaryrec['path']
        if sha(op)!=oldrec['sha256']or sha(os)!=oldsummaryrec['sha256']:raise ValueError('Originalfrozenposterior summary/updates')
        with np.load(pp,allow_pickle=False)as z:new=z['mean'].copy()
        with np.load(op,allow_pickle=False)as z:old=z['means'][case['action']*16+case['noise_index']].copy()
        before=json.loads(os.read_text())['baseline_RMSE_m']['posterior_mean'];truth=truths[obj]
        if new.shape!=(135,)or old.shape!=(135,)or not np.isfinite(new).all()or not np.isfinite(old).all():raise ValueError('Full135predictivemeans')
        olderror=float(np.sqrt(np.mean((old-truth)**2)));newerror=float(np.sqrt(np.mean((new-truth)**2)))
        rows.append(dict(case=case,numerically_passed=row['passed'],before_poke_RMSE_m=before,old_importance_RMSE_m=olderror,refreshed_joint_RMSE_m=newerror,new_minus_old_RMSE_m=newerror-olderror));inputs[str(pp.relative_to(ROOT))]=sha(pp)
    if sorted(r['case']['case_index']for r in rows)!=list(range(9)):raise ValueError('All9fixedcases')
    np.savez(ROOT/OUT/'truth-targets-LOCAL.npz',**{str(k):v for k,v in truths.items()})
    clusters=[dict(object_index=i,cases=sum(r['case']['object_index']==i for r in rows),mean_refreshed_RMSE_m=float(np.mean([r['refreshed_joint_RMSE_m']for r in rows if r['case']['object_index']==i])))for i in sorted(truths)]
    report=dict(completed=True,protocol_sha256=digest,cases=9,objects=4,all9_retained=True,all_numerical_outcomes_retained=True,rows=rows,object_clusters=clusters,
                mean_before_poke_RMSE_m=float(np.mean([r['before_poke_RMSE_m']for r in rows])),mean_old_importance_RMSE_m=float(np.mean([r['old_importance_RMSE_m']for r in rows])),
                mean_refreshed_joint_RMSE_m=float(np.mean([r['refreshed_joint_RMSE_m']for r in rows])),inputs_SHA256=inputs,scope=DESIGN['scope'],policy_advantage_claim=False,physical_calibration_claim=False,raw_targets_LOCAL=True)
    with (ROOT/OUT/'scalar-results.json').open('x')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k]for k in ['cases','mean_before_poke_RMSE_m','mean_old_importance_RMSE_m','mean_refreshed_joint_RMSE_m']}))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run']);a.add_argument('--digest');v=a.parse_args();prepare()if v.mode=='prepare'else run(v.digest)
