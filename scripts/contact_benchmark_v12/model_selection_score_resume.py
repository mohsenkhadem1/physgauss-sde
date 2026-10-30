"""Score already persisted exploratory outputs; never rerun any predictor."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,sha256,write_json,datetime,timezone,check_deadline
from scripts.contact_benchmark_v12.model_selection_experiment import check_freeze
from scripts.contact_benchmark_v12.feasible_experiment import COHORTS,specifications
from scripts.contact_benchmark_v12.data import load
from scripts.contact_benchmark_v12.experiment import base_path
from scripts.external_benchmark.evaluate import expand_prediction
from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
AMENDMENT=REPORT/'model-selection-evaluator-amendment.json'


def declare():
    check_freeze()
    sources={p:sha256(ROOT/p) for p in ['scripts/external_benchmark/evaluate.py','scripts/contact_benchmark_v12/model_selection_scoring.py','scripts/contact_benchmark_v12/model_selection_score_resume.py']}
    if AMENDMENT.exists():
        assert json.loads(AMENDMENT.read_text())['source_sha256']==sources
        return
    write_json(AMENDMENT,dict(declared_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,
        source_freeze_sha256=sha256(REPORT/'model-selection-freeze.json'),
        reason='PhysTwin manual scorer constructs a nearest-neighbor tree from frame0, which crashes for explicitly failed all-NaN candidate predictions.',
        rule='Finite initial predictions use exactly original score. Nonfinite initial predictions retain original coordinate/frame-failure scoring and mark manual correspondence metric unavailable. No fabricated initial geometry or predictor changes.',
        scope='Evaluator-only amendment, after saved predictor outputs; no model selection or parameter changes.'))


def run(spec,cohort):
    check_deadline();check_freeze();declare()
    name=spec.get('case') or f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"
    out=BASE/'model-selection-evaluation'/cohort/name
    if (out/'metrics.json').exists():return True
    if not all((out/file).exists() for file in ['predictions.npz','prediction-metadata.json']):return False
    dirs={tag:BASE/'model-selection-base'/tag/name for tag in ['full','validation']}
    if not all((p/'complete.json').exists() for p in dirs.values()):return False
    data=load(spec);guarded=BASE/'feasibility-evaluation'/cohort/name;original=BASE/'evaluation'/cohort/name
    sources={'current_full':base_path(data,cohort),'current_validation':base_path(data,cohort,True),
             'field_free_full':dirs['full']/'prediction.npz','field_free_validation':dirs['validation']/'prediction.npz',
             'default_prediction':guarded/'predictions.npz','default_metadata':original/'prediction-metadata.json'}
    prediction_sha=sha256(out/'predictions.npz');metadata_sha=sha256(out/'prediction-metadata.json')
    fit_records={}
    for tag,directory in dirs.items():
        complete=json.loads((directory/'complete.json').read_text());fit_path=directory/'fit.json'
        fit=json.loads(fit_path.read_text()) if fit_path.exists() else {}
        fit_records[tag]=dict(completion=complete,optimizer=fit.get('optimizer'),fit_sha256=sha256(fit_path) if fit_path.exists() else None)
    old=json.loads((guarded/'metrics.json').read_text());results=old['results'].copy()
    with np.load(out/'predictions.npz') as predictions:
        assert set(predictions.files)=={'model_selection_'+k for k in ['selected','current','field_free','rigid']}
        results.update({k:safe_score(data,expand_prediction(data,predictions[k]),data['train_end'],len(data['times'])) for k in predictions.files})
    assert prediction_sha==sha256(out/'predictions.npz') and metadata_sha==sha256(out/'prediction-metadata.json')
    write_json(out/'metrics.json',dict(name=name,dataset=data['dataset'],cohort=cohort,scope='post-audit exploratory; no new independent confirmation',
        primary_method='model_selection_selected',results=results,interval=[data['train_end'],len(data['times'])],
        sources={str(p.relative_to(ROOT)):sha256(p) for p in sources.values()},prediction_sha256=prediction_sha,metadata_sha256=metadata_sha,
        prior_metrics_sha256=sha256(guarded/'metrics.json'),freeze_sha256=sha256(REPORT/'model-selection-freeze.json'),field_free_fit_diagnostics=fit_records,
        protocol_sha256=sha256(REPORT/'model-selection-protocol.json'),evaluator_amendment_sha256=sha256(AMENDMENT)))
    print('SCORE_ONLY_RESUMED',cohort,name,flush=True);return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--declare',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args();declare()
    if a.declare:return
    while True:
        done=sum(run(spec,cohort) for cohort in COHORTS for spec in specifications(cohort))
        print('SCORE_ONLY_COMPLETE',done,'/148',flush=True)
        if done==148 or not a.watch:return
        check_deadline();time.sleep(20)

if __name__=='__main__':main()
