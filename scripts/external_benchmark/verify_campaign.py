"""Final artifact, split, source and video-decode audit without rerunning training."""
import sys,json,subprocess,pickle,gzip
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,cv2
from scripts.external_benchmark.common import *

frozen=json.loads((REPORT/'frozen-protocol.json').read_text());result=json.loads((REPORT/'results.json').read_text())
checks=[]
if (REPORT/'results.json.gz').exists():
    assert gzip.decompress((REPORT/'results.json.gz').read_bytes())==(REPORT/'results.json').read_bytes()
    checks.append(dict(check='Published compressed results exactly reproduce the full JSON',passed=True,sha256=sha256(REPORT/'results.json.gz')))
for path,expected in frozen['source_sha256'].items():
    assert sha256(ROOT/path)==expected,path
checks.append(dict(check='Current prediction/scoring core matches pre-test frozen hashes',passed=True))
missing=[];failed_frames=0;failed_by_method={};complete=0
for row in result['sequences']:
    start,end=row['interval']
    for method,record in row['prediction_files'].items():
        path=ROOT/record['path'];assert sha256(path)==record['sha256'],path
        if method=='official_phystwin':continue
        prediction=np.load(path)['predicted'];assert len(prediction)==end,(path,len(prediction),end)
        folder=path.parent;fit=json.loads((folder/'fit.json').read_text());assert fit['fit_end']==start
        if 'selected_frames' in fit:assert max(fit['selected_frames'])<start
        provenance=json.loads((folder/'provenance.json').read_text())
        for source,expected in frozen['source_sha256'].items():assert provenance['source_files'][source]==expected,(path,source)
        if 'fem' in method:
            actual=set(np.flatnonzero(~np.isfinite(prediction).all((1,2))).tolist())
            diagnostic=json.loads((folder/'prediction.json').read_text());reported=set(v['frame'] for v in diagnostic['failed_frames'])
            assert actual==reported,(path,actual,reported)
            if method.startswith('physical_release_'):
                source='scripts/external_benchmark/dynamic_fem.py'
                assert provenance['source_files'][source]==sha256(ROOT/source)
                assert diagnostic['static_gauge_removed_during_dynamics']
            if method.startswith(('adaptive_','physical_release_')):
                source=Path(diagnostic['source_parameters']);assert sha256(source)==diagnostic['source_parameters_sha256']
                assert np.array_equal(np.load(source)['theta'],np.load(folder/'best.npz')['theta'])
            failed_frames+=len(actual);failed_by_method[method]=failed_by_method.get(method,0)+len(actual)
        else:
            assert np.isfinite(prediction).all(),path
            native_input=pickle.load(open(folder/'train.pkl','rb'))
            assert not native_input['object_visibilities'][start:].any(),path
            assert not native_input['object_motions_valid'][start-1:].any(),path
            assert np.array_equal(native_input['object_points'][start:],np.broadcast_to(native_input['object_points'][0],native_input['object_points'][start:].shape)),path
        complete+=1
    missing.extend([dict(name=row['name'],method=m) for m in row['missing']])
checks.append(dict(check='Prediction lengths, fit-prefix limits, source hashes and explicit failed-frame accounting',passed=True,prediction_files=complete,missing=missing,preserved_failed_frames=failed_frames,failed_frames_by_variant=failed_by_method,note='Fixed-step failures are preserved; adaptive variants have separate counts'))
for name in ['primary-double_stretch_sloth','primary-object-soft-task1-demo0','primary-spatial-soft-task2-demo0','adaptive-object-soft-task1-demo0','adaptive-spatial-soft-task2-demo0','adaptive-spatial-soft-task1-demo2','physical-release-object-soft-task1-demo0','physical-release-spatial-soft-task2-demo0']:
    path=REPORT/f'{name}.mp4';cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path
    frames=0;shape=None
    while True:
        ok,frame=cap.read()
        if not ok:break
        frames+=1;shape=frame.shape
    fps=cap.get(cv2.CAP_PROP_FPS);cap.release();meta=json.loads(path.with_suffix('.json').read_text())
    assert frames==meta['frames'] and abs(fps-meta['fps'])<.01,(path,frames,meta)
    checks.append(dict(check='Video fully decodes',file=str(path.relative_to(ROOT)),passed=True,frames=frames,fps=fps,shape=shape,sha256=sha256(path)))
write_json(REPORT/'final-audit.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks))
print(json.dumps(checks,indent=2))
