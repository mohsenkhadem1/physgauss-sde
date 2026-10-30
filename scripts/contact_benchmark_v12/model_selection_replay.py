"""Prespecified reproducibility replay, not independent algorithm validation."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,sha256,write_json,datetime,timezone,check_deadline
from scripts.contact_benchmark_v12.data import load,predictor_payload,trim_prefix
from scripts.contact_benchmark_v12.feasible_experiment import COHORTS,specifications
from scripts.contact_benchmark_v12.model_selection_experiment import check_freeze
from scripts.contact_benchmark_v12.experiment import base_path,finite_metadata
from scripts.contact_benchmark_v12.model_selection import select_model
from scripts.contact_benchmark_v12.scene import load_scene
from scripts.contact_benchmark_v12.feasibility_verify import equal
PLAN=REPORT/'model-selection-replay-plan.json'
OUTPUT=REPORT/'model-selection-replay-audit.json'


def planned_cases():
    cases=[]
    for cohort in COHORTS:
        for spec in specifications(cohort):
            if 'case' in spec or (cohort=='regression' and spec['subset'] in ['object-soft','spatial-soft'] and spec['task'] in [1,2] and spec['episode']==0):
                name=spec.get('case') or f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"
                cases.append(dict(cohort=cohort,spec=spec,name=name))
    assert len(cases)==12 and sum('case' in c['spec'] for c in cases)==8
    return cases


def declare():
    check_freeze();cases=planned_cases()
    if PLAN.exists():equal(json.loads(PLAN.read_text())['cases'],cases);return
    write_json(PLAN,dict(declared_utc=datetime.now(timezone.utc).isoformat(),cases=cases,
        source_freeze_sha256=sha256(REPORT/'model-selection-freeze.json'),
        rationale='All eight PhysTwin cases; regression object/spatial SoftVT tasks1/2 episode0. Fixed by case identity before replay, no score selection.',
        comparison='Replay all validation configurations/scores and all four final arrays; no suffix score calculation or tuning.',
        interpretation='Same-algorithm reproducibility replay, not independent validation-loss algorithm'))
    print('REPLAY_PLAN',sha256(PLAN),flush=True)


def replay(case):
    check_deadline();check_freeze();start=time.monotonic()
    spec,cohort,name=case['spec'],case['cohort'],case['name']
    folder=BASE/'model-selection-evaluation'/cohort/name
    if not (folder/'metrics.json').exists():return None
    data=load(spec);payload=predictor_payload(data)
    short=predictor_payload(trim_prefix(data,max(3,int(data['train_end']*.7))))
    guarded=BASE/'feasibility-evaluation'/cohort/name
    original=BASE/'evaluation'/cohort/name
    sources={'current_full':base_path(data,cohort),'current_validation':base_path(data,cohort,True),
        **{f'field_free_{tag}':BASE/'model-selection-base'/tag/name/'prediction.npz' for tag in ['full','validation']},
        'default_prediction':guarded/'predictions.npz','default_metadata':original/'prediction-metadata.json'}
    families={family:{tag:np.load(sources[family+'_'+tag])['predicted'] for tag in ['full','validation']} for family in ['current','field_free']}
    default_meta=json.loads(sources['default_metadata'].read_text())['object_frame']
    primary='feasible_contact_scene_state' if data['dataset']=='SoftVTBench' else 'feasible_object_frame'
    with np.load(sources['default_prediction']) as saved:default=saved[primary]
    scene=load_scene(payload) if data['dataset']=='SoftVTBench' else None
    predictions,metadata=select_model(payload,short,families,default_meta,default,scene=scene)
    recorded=json.loads((folder/'prediction-metadata.json').read_text());recorded.pop('topology')
    equal(finite_metadata(metadata),recorded,name+'.metadata')
    max_abs={}
    with np.load(folder/'predictions.npz') as saved:
        for family,prediction in predictions.items():
            prior=saved['model_selection_'+family]
            np.testing.assert_allclose(prediction,prior,rtol=1e-10,atol=1e-12,equal_nan=True)
            finite=np.isfinite(prediction)&np.isfinite(prior)
            max_abs[family]=float(np.max(np.abs(prediction[finite]-prior[finite]))) if finite.any() else None
    return dict(**case,elapsed_seconds=time.monotonic()-start,validation_configurations=len(metadata['validation']),
        metadata_reproduced=True,all_four_predictions_reproduced=True,prediction_max_absolute_difference_m=max_abs,
        sources={str(p.relative_to(ROOT)):sha256(p) for p in sources.values()},
        saved_prediction_sha256=sha256(folder/'predictions.npz'),saved_metadata_sha256=sha256(folder/'prediction-metadata.json'))


def run(watch=False):
    declare();plan=json.loads(PLAN.read_text());equal(plan['cases'],planned_cases())
    assert plan['source_freeze_sha256']==sha256(REPORT/'model-selection-freeze.json')
    results={}
    if OUTPUT.exists():
        prior=json.loads(OUTPUT.read_text());assert prior['plan_sha256']==sha256(PLAN)
        assert prior['replay_source_sha256']==sha256(Path(__file__))
        results={r['name']:r for r in prior['cases']}
        assert set(results)<=set(c['name'] for c in plan['cases'])
        for record in results.values():
            for path,digest in record['sources'].items():assert sha256(ROOT/path)==digest,path
            folder=BASE/'model-selection-evaluation'/record['cohort']/record['name']
            assert sha256(folder/'predictions.npz')==record['saved_prediction_sha256']
            assert sha256(folder/'prediction-metadata.json')==record['saved_metadata_sha256']
    while True:
        for case in plan['cases']:
            if case['name'] in results:continue
            record=replay(case)
            if record is not None:
                results[case['name']]=record
                print('REPLAY_VERIFIED',case['name'],record['elapsed_seconds'],flush=True)
                write_json(OUTPUT,dict(created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha256(PLAN),
                    replay_source_sha256=sha256(Path(__file__)),complete=len(results)==12,cases=list(results.values()),
                    interpretation='Same-algorithm reproducibility: validation losses/configurations and full predictions replayed; no suffix errors scored.'))
        if len(results)==12:return
        if not watch:raise RuntimeError(f'Only {len(results)}/12 declared replay cases available')
        check_deadline();time.sleep(20)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--declare',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.declare:declare()
    else:run(a.watch)
