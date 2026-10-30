"""Final independent artifact/protocol checks; no model or score selection."""
import sys,json,gzip,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import cv2
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,sha256,write_json
from scripts.contact_benchmark_v12.data import load
from scripts.external_benchmark.evaluate import score,expand_prediction

def same_scores(actual,recorded):
    if isinstance(actual,dict):
        assert actual.keys()==recorded.keys()
        for key in actual:same_scores(actual[key],recorded[key])
    elif isinstance(actual,(list,tuple)):
        assert len(actual)==len(recorded)
        for a,b in zip(actual,recorded):same_scores(a,b)
    elif isinstance(actual,(int,float)):
        np.testing.assert_allclose(actual,recorded,rtol=1e-10,atol=1e-10,equal_nan=True)
    else:assert actual==recorded

def verify():
    checks=[];freeze=json.loads((REPORT/'method-freeze.json').read_text())
    config=json.loads((ROOT/'configs/contact_benchmark_v12.json').read_text())
    preserved=subprocess.check_output(['git','diff','--name-only',config['frozen_parent'],'--','scripts/external_benchmark','reports/external-benchmark-v11','papers',':(exclude)papers/contact-benchmark-v12'],cwd=ROOT,text=True).strip()
    assert not preserved,('Prior model, evidence or paper changed',preserved)
    checks.append(dict(check='Prior external benchmark source, reports and papers unchanged from frozen parent',passed=True,parent=config['frozen_parent']))
    for name,digest in freeze['source_sha256'].items():
        if sha256(ROOT/name)!=digest:raise AssertionError('Frozen source changed: '+name)
    checks.append(dict(check='Frozen model, input and scoring sources unchanged',passed=True,files=len(freeze['source_sha256'])))
    supplement_path=REPORT/'supplemental-baseline-freeze.json'
    supplement=json.loads(supplement_path.read_text())
    for name,digest in supplement['source_sha256'].items():assert sha256(ROOT/name)==digest,name
    assert supplement['primary_method_freeze_sha256']==sha256(REPORT/'method-freeze.json')
    assert supplement['protocol_sha256']==sha256(REPORT/'benchmark-protocol.json')
    supplement_count=0
    protocol=json.loads((REPORT/'benchmark-protocol.json').read_text());old=json.loads((ROOT/'reports/external-benchmark-v11/results.json').read_text())
    old={r['name']:r for r in old['sequences']};counts={};differences=[];incomplete=[]
    for cohort in ['development','regression','untouched_confirmation']:
        cohort_spec=protocol[cohort];specs=cohort_spec['SoftVTBench']+[dict(case=c) for c in cohort_spec['PhysTwin']]
        counts[cohort]=0
        for spec in specs:
            data=load(spec);name=data['name'];folder=BASE/'evaluation'/cohort/name
            metric=json.loads((folder/'metrics.json').read_text());archive=np.load(folder/'predictions.npz')
            primary='contact_scene_state' if data['dataset']=='SoftVTBench' else 'object_frame'
            pred=archive[primary];assert pred.shape==(len(data['times']),len(data['nodes']),3)
            failed=int((~np.isfinite(pred[data['train_end']:]).all((1,2))).sum())
            assert failed==metric['results'][primary]['summary']['failed_frames']
            same_scores(score(data,expand_prediction(data,pred),data['train_end'],len(data['times'])),metric['results'][primary])
            if failed:incomplete.append(dict(cohort=cohort,name=name,failed_frames=failed))
            validation=json.loads((BASE/'base/validation'/name/'fit.json').read_text());assert validation['fit_end']==int(data['train_end']*.7)
            selection=metric['selection']
            assert selection.get('independent_validation') or selection.get('selection_unavailable')
            if cohort=='regression':
                variant='physical_release_material_world_fem' if data['dataset']=='SoftVTBench' else 'material_world_fem'
                prior=old[name]['results'][variant]['summary'];current=metric['results']['v11_fem']['summary']
                for key in ['coordinate_rmse_mm','aligned_shape_rmse_mm','manual_track_mean_mm']:
                    if key in prior and prior[key] is not None:
                        delta=abs(prior[key]-current[key]);assert delta<1e-7,(name,key,delta);differences.append(delta)
            if cohort!='development' and data['dataset']=='SoftVTBench':
                extra=json.loads((folder/'native-scene-supplement.json').read_text())
                assert extra['decision_sha256']==sha256(supplement_path)
                assert extra['source_native_sha256']==sha256(ROOT/extra['source_native'])
                assert extra['name']==name and extra['cohort']==cohort
                native=np.load(ROOT/extra['source_native'])
                saved=np.load(folder/'native-scene-prediction.npz')
                np.testing.assert_array_equal(native['nodes'],data['nodes'])
                np.testing.assert_array_equal(saved['nodes'],data['nodes'])
                same_scores(score(data,saved['predicted'],data['train_end'],len(data['times'])),extra['result'])
                supplement_count+=1
            counts[cohort]+=1
    checks.append(dict(check='All fixed cohorts accounted for; final failures explicitly reported',passed=True,counts=counts,incomplete=incomplete))
    checks.append(dict(check='Independent shorter-prefix fitting and validation selection',passed=True))
    checks.append(dict(check='Frozen v11 regression scores reproduced without alteration',passed=True,max_absolute_difference_mm=max(differences)))
    assert supplement_count==80
    checks.append(dict(check='Fairness supplement sources, native input identity and recomputed prediction scores verified',passed=True,sequences=supplement_count))
    final_report=json.loads((REPORT/'results.json').read_text())
    assert final_report['comparison_complete']
    for name,digest in final_report['source_files'].items():assert sha256(ROOT/name)==digest,name
    assert final_report['protocol_sha256']==sha256(REPORT/'benchmark-protocol.json')
    assert final_report['method_freeze_sha256']==sha256(REPORT/'method-freeze.json')
    for cohort in ['development','regression','untouched_confirmation']:
        for dataset,summary in final_report['summary'][cohort].items():
            assert summary['comparison_complete']
            for method in summary['required_methods']:
                assert summary['methods'][method]['complete_sequences']==summary['expected_sequences'],(cohort,dataset,method)
    checks.append(dict(check='All required primary and baseline comparisons have complete sequence/frame coverage',passed=True))
    jobs=[json.loads(p.read_text()) for p in (BASE/'jobs').glob('*.json')]
    unfinished=[j['index'] for j in jobs if j['state'] not in ['complete','failed']]
    if unfinished:raise AssertionError(('Unfinished jobs',unfinished))
    checks.append(dict(check='Every planned fitting job has terminal status',passed=True,total=len(jobs),failed=[j['index'] for j in jobs if j['state']=='failed']))
    fitting=[]
    def job_name(j):
        s=j['spec'];return s.get('case') or f"{s['subset']}-task{s['task']}-demo{s['episode']}"
    fit_paths=[BASE/'base'/('validation' if j['validation'] else 'full')/job_name(j)/'fit.json' for j in jobs if j['kind']=='base']
    for path in sorted(fit_paths):
        fit=json.loads(path.read_text());checkpoint=path.parent/'best.npz';saved=np.load(checkpoint)
        np.testing.assert_array_equal(saved['theta'],fit['best']['theta'])
        assert np.isfinite(saved['theta']).all() and (saved['selected_frames']<fit['fit_end']).all()
        fitting.append(dict(path=str(path.relative_to(ROOT)),tag=path.parent.parent.name,name=fit['name'],fit_end=fit['fit_end'],selected_frames=fit['selected_frames'],best=fit['best'],optimizer=fit['optimizer'],
            settings={k:fit[k] for k in ['field','contact','nonconvex','materials']},checkpoint_sha256=sha256(checkpoint),fit_sha256=sha256(path),state=fit['state']))
    native=[json.loads((BASE/'native/confirmation'/job_name(j)/'fit.json').read_text()) for j in jobs if j['kind']=='native']
    write_json(REPORT/'fitting-diagnostics.json',dict(fem=fitting,native=native,timeout_continuations=[j for j in jobs if j.get('timeout_repair_attempted')],interpretation='Fitted ratios and point estimates, not material identification or posterior calibration. Checkpoints/raw data remain local.'))
    checks.append(dict(check='Fitted parameter archive agrees with local checkpoints and observation cuts',passed=True,fem_fits=len(fitting),native_fits=len(native)))
    for video in sorted(REPORT.glob('*.mp4')):
        cap=cv2.VideoCapture(str(video));n=0;fps=cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok,frame=cap.read()
            if not ok:break
            n+=1
        cap.release();assert n>0
        meta=video.with_suffix('.json')
        if meta.exists():
            m=json.loads(meta.read_text())
            if 'frames' in m:assert m['frames']==n
            if 'fps' in m:assert abs(m['fps']-fps)<.01
        checks.append(dict(check='Video decodes completely',passed=True,file=str(video.relative_to(ROOT)),frames=n,fps=fps,sha256=sha256(video)))
    for archive in REPORT.glob('*results.json.gz'):
        unpacked=gzip.decompress(archive.read_bytes());json.loads(unpacked)
        mirror=archive.with_suffix('')
        if mirror.exists():assert json.loads(unpacked)==json.loads(mirror.read_text())
        checks.append(dict(check='Lossless result archive verified',passed=True,file=str(archive.relative_to(ROOT)),sha256=sha256(archive)))
    write_json(REPORT/'final-audit.json',dict(checks=checks,all_passed=all(c['passed'] for c in checks)))
    print(json.dumps(dict(checks=len(checks),counts=counts,primary_incomplete=incomplete)))

if __name__=='__main__':verify()
