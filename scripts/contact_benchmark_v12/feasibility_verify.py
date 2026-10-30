"""Independent final guarded-campaign audit; no model or score selection.

Enumerates protocols and the recorded fresh execution plan, never unrelated
local base/native directories. Positive Jacobians are recomputed from the
original FEM rest gradients, not the guard's determinant implementation.
"""
import datetime,gzip,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,sha256,write_json
from scripts.contact_benchmark_v12.data import load,predictor_payload
from scripts.external_benchmark.fem import ExternalFEM
from scripts.external_benchmark.data import rigid_map
from scripts.external_benchmark.evaluate import score,expand_prediction

COHORTS=['development','regression','untouched_confirmation','fresh_confirmation']
EXPECTED={'development':21,'regression':43,'untouched_confirmation':42,'fresh_confirmation':42}


def read(path):return json.loads(Path(path).read_text())

def name_of(spec):return spec.get('case') or f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"

def equal(a,b,path='root'):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+f'[{i}]')
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert np.isclose(a,b,atol=1e-7,rtol=1e-9,equal_nan=True),(path,a,b)
    else:assert a==b,(path,a,b)


def gradient_jacobians(model,points):
    """Independent J from original grad(F), matching frozen FEM normalization."""
    x=(np.asarray(points)-model.origin)/model.scale
    return np.linalg.det(np.einsum('eai,eaj->eij',x[model.elements.cpu().numpy()],model.grad.cpu().numpy()))


def topology(model,trajectory,record):
    failed=[];minimum=float('inf')
    assert record['frames']==len(trajectory)
    for t,points in enumerate(trajectory):
        row=record['per_frame'][t];assert row['frame']==t
        if not np.isfinite(points).all():
            failed.append(t);assert row['failed'];continue
        J=gradient_jacobians(model,points)
        assert np.isfinite(J).all() and (J>.03).all(),('Final local volume violation',t,float(J.min()))
        assert not row['failed'] and row['inverted_elements']==0 and row['below_solver_margin']==0
        assert np.isclose(float(J.min()),row['min_J'],atol=1e-6,rtol=1e-7)
        minimum=min(minimum,float(J.min()))
    assert record['failed_frames']==len(failed)
    assert record['frames_with_inversions']==record['frames_below_solver_margin']==0
    return dict(failed_frames=failed,min_J=minimum if np.isfinite(minimum) else None)


def verify():
    checks=[];frozen=read(REPORT/'feasibility-freeze.json');protocol=read(REPORT/'feasibility-protocol.json');old_protocol=read(REPORT/'benchmark-protocol.json')
    assert len(frozen['source_sha256'])==17
    assert frozen['protocol_sha256']==sha256(REPORT/'feasibility-protocol.json')
    assert protocol['preserved_protocol_sha256']==sha256(REPORT/'benchmark-protocol.json')
    assert protocol['preserved_method_freeze_sha256']==sha256(REPORT/'method-freeze.json')
    for path,digest in frozen['source_sha256'].items():assert sha256(ROOT/path)==digest,path
    checks.append(dict(check='All17 frozen sources and both protocols intact',passed=True))
    report=read(REPORT/'feasibility-results.json');assert report['cohorts']==COHORTS
    assert report['freeze_sha256']==sha256(REPORT/'feasibility-freeze.json')
    assert report['protocol_sha256']==sha256(REPORT/'feasibility-protocol.json')
    for path,digest in report['sources'].items():assert sha256(ROOT/path)==digest,path
    old_report=read(REPORT/'results.json')
    for path,digest in old_report['source_files'].items():assert sha256(ROOT/path)==digest,path
    old_v11={r['name']:r for r in read(ROOT/'reports/external-benchmark-v11/results.json')['sequences']}
    geometry=[];preservation=[];coverage=[];allnames={};data_cache={}
    for cohort in COHORTS:
        group=(protocol if cohort=='fresh_confirmation' else old_protocol)[cohort]
        specs=group['SoftVTBench']+[dict(case=c) for c in group['PhysTwin']]
        assert len(specs)==EXPECTED[cohort];allnames[cohort]=[name_of(s) for s in specs]
        published={r['name']:r for r in report['sequences'][cohort]};assert set(published)==set(allnames[cohort])
        for spec in specs:
            name=name_of(spec);data=load(spec);data_cache[name]=data;folder=BASE/'feasibility-evaluation'/cohort/name
            metric=read(folder/'metrics.json');equal(metric,published[name],name)
            for artifact in ['metrics.json','prediction-metadata.json']:
                relative=str((folder/artifact).relative_to(ROOT))
                assert report['sources'][relative]==sha256(folder/artifact)
            assert metric['cohort']==cohort and metric['name']==name
            assert metric['freeze_sha256']==sha256(REPORT/'feasibility-freeze.json')
            assert metric['protocol_sha256']==sha256(REPORT/'feasibility-protocol.json')
            original=BASE/'evaluation'/cohort/name
            for key,path in [('prediction_sha256',folder/'predictions.npz'),('guard_metadata_sha256',folder/'prediction-metadata.json'),('original_prediction_sha256',original/'predictions.npz'),('original_metrics_sha256',original/'metrics.json')]:assert metric[key]==sha256(path),(name,key)
            preservation.append(dict(cohort=cohort,name=name,path=str((original/'predictions.npz').relative_to(ROOT)),sha256=metric['original_prediction_sha256'],proof='unchanged since hash recorded by guarded evaluation; not an earlier historical inventory'))
            with np.load(folder/'predictions.npz') as archive,np.load(original/'predictions.npz') as inherited:
                primary='feasible_contact_scene_state' if data['dataset']=='SoftVTBench' else 'feasible_object_frame'
                prediction=archive[primary];assert prediction.shape==(len(data['times']),len(data['nodes']),3)
                start=data['train_end'];end=len(data['times']);assert metric['interval']==[start,end]
                recalculated=score(data,expand_prediction(data,prediction),start,end)
                equal(recalculated,metric['results'][primary],name+'.score')
                meta=read(folder/'prediction-metadata.json');guard=meta['guard']
                assert guard['placement']=='before contact rollout and derived velocities'
                assert guard['cutoff_J']==.03 and guard['numerical_margin']==1e-8 and guard['alpha_schedule']==frozen['alpha_schedule']
                assert len(guard['frames'])==end
                before=archive['feasible_object_frame'];base=inherited['v11_fem'];candidate=inherited['object_frame']
                for t,row in enumerate(guard['frames']):
                    assert row['frame']==t
                    if row['failed']:
                        assert not np.isfinite(before[t]).all() and t in guard['failed_frames'];continue
                    alpha=row['alpha'];assert alpha in frozen['alpha_schedule']
                    if alpha==1:np.testing.assert_array_equal(before[t],candidate[t])
                    else:
                        R,b=rigid_map(base[t],candidate[t]);anchor=base[t]@R+b
                        np.testing.assert_allclose(before[t],anchor+alpha*(candidate[t]-anchor),atol=1e-10,rtol=1e-10)
                assert guard['failed_frames']==np.flatnonzero(~np.isfinite(before).all((1,2))).tolist()
                model=ExternalFEM(predictor_payload(data),device='cpu',nonconvex=data['dataset']=='PhysTwin')
                for key in ['feasible_object_frame',primary]:
                    diag=topology(model,archive[key],meta['topology'][key])
                equal(meta['topology'][primary],metric['topology'])
                summary=metric['results'][primary]['summary']
                assert summary['failed_frames']==sum(t>=start for t in diag['failed_frames'])
                geometry.append(dict(cohort=cohort,name=name,**diag))
            required=['v11_fem','object_frame','pose_reset','prefix_calibrated_rigid',primary]
            required+=['contact_scene_state','rigid_scene_state'] if data['dataset']=='SoftVTBench' else ['official_phystwin']
            if data['dataset']=='SoftVTBench' and cohort!='development':required+=['adapted_phystwin','adapted_phystwin_scene_state']
            for method in required:
                assert method in metric['results'],(name,method)
                s=metric['results'][method]['summary'];assert s['total_frames']==end-start
                assert s['valid_frames']+s['failed_frames']==end-start
                assert s['failed_frames']==0,(name,method,'Required baseline incomplete')
            coverage.append(dict(cohort=cohort,name=name,required=required))
            if cohort=='regression':
                variant='physical_release_material_world_fem' if data['dataset']=='SoftVTBench' else 'material_world_fem'
                equal(metric['results']['v11_fem'],old_v11[name]['results'][variant],name+'.v11')
        print('AUDIT',cohort,len(specs),'sequences',flush=True)
    checks.append(dict(check='All148 fixed cases, report sources, primary score recomputation and complete required baselines',passed=True,counts=EXPECTED))
    checks.append(dict(check='Independent original-FEM gradient Jacobians on every finite guarded frame; failures accounted',passed=True,sequences=geometry))
    checks.append(dict(check='Original106 source predictions unchanged since guarded evaluation; v11 regression scores unchanged',passed=True,old_sequences=sum(c!='fresh_confirmation' for c in [r['cohort'] for r in preservation])))
    original_video_records=[]
    for chosen in read(REPORT/'video-selection.json')['cases']:
        path=REPORT/f"{chosen['cohort']}-{name_of(chosen)}.json";video_meta=read(path)
        assert sha256(ROOT/video_meta['source_prediction'])==video_meta['prediction_sha256']
        original_video_records.append(dict(metadata=str(path.relative_to(ROOT)),source=video_meta['source_prediction'],sha256=video_meta['prediction_sha256']))
    checks.append(dict(check='Four original video provenance records still match original predictions',passed=True,records=original_video_records))
    write_json(REPORT/'feasibility-original-preservation.json',dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Final audit inventory; historical comparison only to hash recorded at respective guarded evaluation',records=preservation))
    plan=read(REPORT/'feasibility-execution-plan.json');assert len(plan['jobs'])==124
    assert plan['protocol_sha256']==sha256(REPORT/'feasibility-protocol.json')
    fits=[];native=[];states=[];expected_jobs=set()
    for planned in plan['jobs']:
        expected_jobs.add(planned['index']);job=read(BASE/'feasibility-jobs'/f"{planned['index']:04d}.json")
        for k in ['index','kind','cohort','spec','command']:assert job[k]==planned[k]
        assert job['state'] in ['complete','failed'];assert job['state']=='complete',(job['index'],job['state'])
        states.append(job);name=name_of(job['spec']);data=data_cache[name]
        if job['kind']=='base':
            assert job['validation']==planned['validation'];tag='validation' if job['validation'] else 'full';folder=BASE/'base'/tag/name
            fitting=read(folder/'fit.json');complete=read(folder/'complete.json');end=int(data['train_end']*.7) if job['validation'] else data['train_end']
            assert fitting['fit_end']==end and complete['fit_end']==end
            saved=np.load(folder/'best.npz');np.testing.assert_array_equal(saved['theta'],fitting['best']['theta'])
            assert np.isfinite(saved['theta']).all() and (saved['selected_frames']<end).all() and (saved['selected_frames']>=0).all()
            np.testing.assert_array_equal(saved['selected_frames'],fitting['selected_frames'])
            fits.append(dict(index=job['index'],name=name,tag=tag,fit_end=end,theta=saved['theta'].tolist(),selected_frames=saved['selected_frames'].tolist(),optimizer=fitting['optimizer'],state=fitting['state'],checkpoint_sha256=sha256(folder/'best.npz'),fit_sha256=sha256(folder/'fit.json'),complete_sha256=sha256(folder/'complete.json')))
        else:
            folder=BASE/'native/confirmation'/name;fitting=read(folder/'fit.json');assert fitting['fit_end']==data['train_end'] and fitting['epochs']==100 and fitting['initial']==30000
            assert sha256(Path(fitting['checkpoint']))==fitting['checkpoint_sha256']
            with open(folder/'train.pkl','rb') as f:train=pickle.load(f)
            end=data['train_end'];assert not train['object_visibilities'][end:].any()
            assert not train['object_motions_valid'][max(0,end-1):].any()
            expected=np.broadcast_to(data['nodes']+np.array([0.,0.,-data['floor']]),train['object_points'][end:].shape).astype('float32')
            np.testing.assert_array_equal(train['object_points'][end:],expected)
            native.append(dict(index=job['index'],**fitting,fit_sha256=sha256(folder/'fit.json'),prediction_sha256=sha256(folder/'prediction.npz'),train_sha256=sha256(folder/'train.pkl')))
    assert len(fits)==84 and len(native)==40
    actual_jobs={int(p.stem) for p in (BASE/'feasibility-jobs').glob('*.json')};assert actual_jobs==expected_jobs
    write_json(REPORT/'feasibility-fitting-diagnostics.json',dict(expected_job_count=124,fem=fits,native=native,job_states=states,interpretation='Expected fresh jobs only; fitted ratios are not calibrated material posteriors'))
    checks.append(dict(check='Exact124 planned fresh jobs,84 prefix-safe FEM checkpoints,40 native checkpoints and hidden future training targets',passed=True))
    selection=read(REPORT/'feasibility-video-selection.json');videos=[]
    for spec in selection['cases']:
        cohort=spec['cohort'];name=name_of(spec);video=REPORT/f'guarded-{cohort}-{name}.mp4';meta=read(video.with_suffix('.json'))
        assert meta['freeze_sha256']==sha256(REPORT/'feasibility-freeze.json')
        for pathkey,hashkey in [('source_prediction','prediction_sha256'),('guarded_prediction','guarded_prediction_sha256')]:assert sha256(ROOT/meta[pathkey])==meta[hashkey]
        cap=cv2.VideoCapture(str(video));count=0;fps=cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok,frame=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==meta['frames'] and count>0 and abs(fps-meta['fps'])<.01
        videos.append(dict(path=str(video.relative_to(ROOT)),frames=count,fps=fps,sha256=sha256(video)))
    assert len(videos)==4;checks.append(dict(check='Four prespecified guarded videos decode at recorded frame rate and match prediction provenance',passed=True,videos=videos))
    archive=REPORT/'feasibility-results.json.gz';unpacked=gzip.decompress(archive.read_bytes());assert unpacked==(REPORT/'feasibility-results.json').read_bytes()
    assert report['complete'] and report['geometry_complete_and_feasible']
    checks.append(dict(check='Published lossless archive, complete report and final geometry agree',passed=True,sha256=sha256(archive)))
    write_json(REPORT/'feasibility-final-audit.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_passed=True,checks=checks,scope='Local saved-frame volume validity only; no global selfcollision or between-frame certification'))
    print(json.dumps(dict(all_passed=True,checks=len(checks),sequences=148,fresh_jobs=124)))

if __name__=='__main__':verify()
