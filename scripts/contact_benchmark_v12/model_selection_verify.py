"""Recompute all exploratory scores and local volumes independently."""
import gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,sha256,write_json,datetime,timezone
from scripts.contact_benchmark_v12.data import load,predictor_payload
from scripts.contact_benchmark_v12.feasible_experiment import COHORTS,specifications
from scripts.contact_benchmark_v12.model_selection_experiment import check_freeze
from scripts.contact_benchmark_v12.feasibility_verify import equal,topology
from scripts.external_benchmark.fem import ExternalFEM
from scripts.external_benchmark.evaluate import expand_prediction
from scripts.contact_benchmark_v12.model_selection_scoring import safe_score

def verify_decision(meta,predictions,default):
    """Independently reconstruct fixed ordering/2% rule from recorded prefix scores.

    Does not reexecute fits or claim independent recomputation of validation losses.
    """
    from scripts.contact_benchmark_v12.model_selection import CONFIGS
    dc=meta['default_configuration'];default_config=(dc['family'],dc['method'],dc['strength'])
    expected=[default_config]+[(f,m,a) for f in ['current','field_free'] for m,a in CONFIGS if (f,m,a)!=default_config]+[('rigid','none',0.)]
    rows=meta['validation']
    assert [(r['family'],r['method'],r['strength']) for r in rows]==expected
    values=np.array([float(r['validation_rmse_m']) if r['validation_finite'] else np.inf for r in rows])
    for r,value in zip(rows,values):
        if r['validation_finite']:
            assert np.isfinite(value) and value>=0 and not r['failed_validation_frames']
        else:assert r['validation_rmse_m'] is None
    best=int(np.argmin(values));switch=bool(np.isfinite(values[best]) and values[best]<.98*values[0]);chosen=best if switch else 0
    assert meta['switched_from_previous']==switch
    equal(meta['selected_configuration'],rows[chosen])
    for family in ['current','field_free','rigid']:
        indices=[i for i,c in enumerate(expected) if c[0]==family]
        win=min(indices,key=lambda i:values[i]);equal(meta['family_winners'][family],rows[win])
        if not np.isfinite(values[win]):
            assert np.isnan(predictions['model_selection_'+family]).all()
            assert meta['final_predictions'][family]['failed_family_selection']
    selected=predictions['model_selection_selected']
    if switch:
        np.testing.assert_array_equal(selected,predictions['model_selection_'+expected[chosen][0]])
    else:
        np.testing.assert_array_equal(selected,default)
        assert meta['final_predictions']['selected']['exact_previous_guarded_prediction']
    return dict(switched=switch,selected_family=expected[chosen][0],validation_scores_recomputed=False,
                decision_rebuilt_from_recorded_prefix_scores=True,default_exactly_preserved=not switch)


def verify_fits():
    from scripts.contact_benchmark_v12 import model_selection_jobs as jobs
    from scripts.contact_benchmark_v12 import fieldfree_runner as runner
    plan=json.loads(jobs.PLAN.read_text());expected=jobs.planned_jobs()
    assert len(plan['jobs'])==296
    assert plan['protocol_sha256']==sha256(runner.PROTOCOL) and plan['freeze_sha256']==sha256(runner.FREEZE)
    records=[]
    for original,planned in zip(expected,plan['jobs']):
        equal(original,planned)
        job=json.loads((jobs.QUEUE/f"{original['index']:04d}.json").read_text())
        for key in ['index','kind','cohort','spec','validation','command']:equal(job[key],original[key])
        data=predictor_payload(load(job['spec'],job['validation']))
        directory=runner.OUTPUT/('validation' if job['validation'] else 'full')/data['name']
        completed=json.loads((directory/'complete.json').read_text())
        assert completed['fit_end']==data['train_end'] and completed['validation']==job['validation']
        assert completed['family']=='field_free' and completed['settings']==runner.settings(data)
        assert completed['prediction_sha256']==sha256(directory/'prediction.npz')
        assert completed['freeze_sha256']==sha256(runner.FREEZE) and completed['protocol_sha256']==sha256(runner.PROTOCOL)
        terminal=bool(completed.get('terminal_failure',False));optimizer=None
        with np.load(directory/'prediction.npz') as saved:
            assert saved['predicted'].shape==(len(data['times']),len(data['nodes']),3)
            np.testing.assert_array_equal(saved['nodes'],data['nodes'])
            if terminal:
                assert job['state']=='failed' and job.get('terminal_failure_artifact')
                assert np.isnan(saved['predicted']).all() and completed['job_index']==job['index']
                assert completed['failure_reason'] and completed['failed_frames']==len(data['times'])
                equal(completed,json.loads((directory/'terminal-failure.json').read_text()))
            else:
                assert job['state']=='complete'
                theta,fit=runner.validate_checkpoint(data,directory)
                np.testing.assert_array_equal(saved['theta'],theta)
                assert completed['checkpoint_sha256']==sha256(directory/'best.npz')
                optimizer=fit['optimizer']
                if fit.get('state')=='budget_exhausted_valid_checkpoint':
                    assert not optimizer['success'] and job['timeout_repair_attempted']
                    assert fit['timeout_checkpoint_sha256']==sha256(directory/'best.npz')
                if job.get('timeout_repair_attempted'):
                    assert job['repair_returncode']==0 and job['repair_checkpoint_sha256']==sha256(directory/'best.npz')
                    snapshot=BASE/'model-selection-timeout-records'/str(job['index'])
                    assert sha256(snapshot/'best.npz')==sha256(directory/'best.npz')
        records.append(dict(index=job['index'],name=data['name'],validation=job['validation'],fit_end=data['train_end'],terminal_failure=terminal,
                            optimizer=optimizer,timeout_continuation=bool(job.get('timeout_repair_attempted'))))
    assert len(records)==296 and sum(r['validation'] for r in records)==148
    return records

def verify():
    check_freeze();records=[];decisions=[];fits=verify_fits()
    with gzip.open(REPORT/'model-selection-results.json.gz','rt') as f:report=json.load(f)
    for cohort in COHORTS:
        published={r['name']:r for r in report['sequences'][cohort]}
        assert len(published)==len(specifications(cohort))
        for spec in specifications(cohort):
            data=load(spec);name=data['name'];folder=BASE/'model-selection-evaluation'/cohort/name
            row=json.loads((folder/'metrics.json').read_text());equal(row,published[name])
            assert sha256(folder/'predictions.npz')==row['prediction_sha256']
            assert row['freeze_sha256']==sha256(REPORT/'model-selection-freeze.json')
            assert row['protocol_sha256']==sha256(REPORT/'model-selection-protocol.json')
            assert sha256(folder/'prediction-metadata.json')==row['metadata_sha256']
            for path,digest in row['sources'].items():assert sha256(ROOT/path)==digest,path
            if 'evaluator_amendment_sha256' in row:
                amendment=REPORT/'model-selection-evaluator-amendment.json'
                assert sha256(amendment)==row['evaluator_amendment_sha256']
                for path,digest in json.loads(amendment.read_text())['source_sha256'].items():assert sha256(ROOT/path)==digest,path
            old_path=BASE/'feasibility-evaluation'/cohort/name/'metrics.json'
            assert sha256(old_path)==row['prior_metrics_sha256']
            for tag in ['full','validation']:
                fitdir=BASE/'model-selection-base'/tag/name
                fitrecord=row['field_free_fit_diagnostics'][tag]
                equal(fitrecord['completion'],json.loads((fitdir/'complete.json').read_text()))
                fitpath=fitdir/'fit.json'
                if fitpath.exists():
                    assert fitrecord['fit_sha256']==sha256(fitpath)
                    equal(fitrecord['optimizer'],json.loads(fitpath.read_text()).get('optimizer'))
                else:assert fitrecord['fit_sha256'] is None and fitrecord['optimizer'] is None
            old=json.loads(old_path.read_text())
            for method,value in old['results'].items():equal(row['results'][method],value,name+'.preserved.'+method)
            meta=json.loads((folder/'prediction-metadata.json').read_text())
            model=ExternalFEM(predictor_payload(data),device='cpu',nonconvex=data['dataset']=='PhysTwin')
            with np.load(folder/'predictions.npz') as saved:
                assert set(saved.files)=={'model_selection_'+s for s in ['selected','current','field_free','rigid']}
                with np.load(old_path.parent/'predictions.npz') as previous:
                    primary='feasible_contact_scene_state' if data['dataset']=='SoftVTBench' else 'feasible_object_frame'
                    decision=verify_decision(meta,saved,previous[primary])
                decisions.append(dict(name=name,cohort=cohort,**decision))
                for method in saved.files:
                    pred=saved[method];assert pred.shape==(len(data['times']),len(data['nodes']),3)
                    recomputed=safe_score(data,expand_prediction(data,pred),data['train_end'],len(data['times']))
                    equal(recomputed,row['results'][method],name+'.'+method)
                    diag=topology(model,pred,meta['topology'][method])
                    records.append(dict(cohort=cohort,name=name,method=method,**diag))
    assert len(records)==148*4
    write_json(REPORT/'model-selection-fitting-diagnostics.json',dict(fits=fits,completed=len(fits),terminal_failures=sum(r['terminal_failure'] for r in fits)))
    write_json(REPORT/'model-selection-final-audit.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_freeze_intact=True,cases=148,predictions_recomputed=len(records),fits_checked=len(fits),decisions=decisions,
        validation_limitation='Decision independently rebuilt from stored prefix scores; validation losses not independently recomputed',
        checks=['All saved prediction scores recomputed','All finite local volumes independently checked from FEM gradients',
                'All prior guarded metrics preserved','All source prediction and metadata hashes intact'],geometry=records))
    print('MODEL_SELECTION_VERIFIED',len(records),flush=True)

if __name__=='__main__':verify()
