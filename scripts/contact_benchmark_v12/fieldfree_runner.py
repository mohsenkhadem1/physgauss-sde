"""Separate refit with only the Gaussian world potential disabled.

Uses unchanged fitter, parameter bounds/starts, mesh, attachments, reversible
support term and release integrator. This is not post-hoc field removal.
"""
import argparse,json,sys,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import minimize
from scripts.contact_benchmark_v12.common import BASE,ROOT,REPORT,check_deadline,write_json,provenance,sha256
from scripts.contact_benchmark_v12.data import load,predictor_payload
from scripts.contact_benchmark_v12.base_runner import feasible,rollout
from scripts.external_benchmark import fit as oldfit

OUTPUT=BASE/'model-selection-base'
FREEZE=REPORT/'model-selection-freeze.json'
PROTOCOL=REPORT/'model-selection-protocol.json'


def check_freeze():
    record=json.loads(FREEZE.read_text())
    assert record['protocol_sha256']==sha256(PROTOCOL),'Model-selection protocol changed'
    for path,digest in record['source_sha256'].items():assert sha256(ROOT/path)==digest,path
    assert 'scripts/contact_benchmark_v12/fieldfree_runner.py' in record['source_sha256']


def settings(data):
    return dict(field=False,contact=True,nonconvex=data['dataset']=='PhysTwin',materials=6 if data['dataset']=='PhysTwin' else 1)


def validate_checkpoint(data,out):
    saved=np.load(out/'best.npz');fit=json.loads((out/'fit.json').read_text())
    np.testing.assert_array_equal(saved['nodes'],data['nodes'])
    np.testing.assert_array_equal(saved['theta'],fit['best']['theta'])
    selected=np.unique(np.linspace(0,data['train_end']-1,36,dtype=int))
    np.testing.assert_array_equal(saved['selected_frames'],selected)
    assert fit['fit_end']==data['train_end'] and np.isfinite(saved['theta']).all()
    for key,value in settings(data).items():assert fit[key]==value,(key,fit[key],value)
    field_index=2+len(data['attachment_weights'])+settings(data)['materials']
    assert saved['theta'][field_index]==0.,'Disabled field coefficient must remain fixed at zero'
    return saved['theta'].copy(),fit


def terminal_failure(spec,validation,reason,job_index=None):
    """Publish explicit unavailable-family arrays; never fabricate a fitted model."""
    data=predictor_payload(load(spec,validation));tag='validation' if validation else 'full';out=OUTPUT/tag/data['name'];out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        existing=json.loads((out/'complete.json').read_text())
        if not existing.get('terminal_failure'):
            return dict(preserved_existing_complete=True,reason=reason)
    if (out/'prediction.npz').exists() and not (out/'prediction.before-terminal-failure.npz').exists():
        shutil.copy2(out/'prediction.npz',out/'prediction.before-terminal-failure.npz')
    count=len(data['times']);theta=np.full(len(data['attachment_weights'])+settings(data)['materials']+4,np.nan)
    if (out/'best.npz').exists():
        try:theta=np.load(out/'best.npz')['theta'].copy()
        except (ValueError,OSError,KeyError):pass
    temporary=out/'prediction.pending.npz';np.savez_compressed(temporary,predicted=np.full((count,len(data['nodes']),3),np.nan),nodes=data['nodes'],theta=theta);temporary.replace(out/'prediction.npz')
    record=dict(name=data['name'],family='field_free',validation=bool(validation),fit_end=data['train_end'],settings=settings(data),terminal_failure=True,failure_reason=str(reason),job_index=job_index,
                failed_frames=count,prediction_sha256=sha256(out/'prediction.npz'),freeze_sha256=sha256(FREEZE),protocol_sha256=sha256(PROTOCOL),
                interpretation='Unavailable candidate; all frames explicitly NaN. No synthetic fit or convergence claim; other families may still be evaluated.')
    write_json(out/'terminal-failure.json',record)
    write_json(out/'prediction.json',dict(**record,failed_frame_records=[dict(frame=t,reason=str(reason)) for t in range(count)]))
    write_json(out/'complete.json',record)
    return record


def run(spec,validation=False,device='cpu'):
    if device!='cpu':raise ValueError('This independent fitting queue is CPU-only')
    check_deadline();check_freeze()
    # The adapter loads evaluator data, then strips it before any optimization
    # or predictor call. Shortened calibration is regenerated inside load.
    data=predictor_payload(load(spec,validation));tag='validation' if validation else 'full';out=OUTPUT/tag/data['name']
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        completed=json.loads((out/'complete.json').read_text())
        if completed.get('terminal_failure'):
            with np.load(out/'prediction.npz') as failed:
                assert failed['predicted'].shape==(len(data['times']),len(data['nodes']),3) and np.isnan(failed['predicted']).all()
            return
        validate_checkpoint(data,out);return
    start=time.monotonic();configuration=settings(data)
    if (out/'best.npz').exists():
        if not (out/'fit.json').exists():raise ValueError('Unfinished checkpoint requires explicit timeout-continuation validation')
        theta,_=validate_checkpoint(data,out)
    else:
        # Scoped in-memory bindings only; no frozen source files are changed.
        prior=(oldfit.check_deadline,oldfit.save_provenance,oldfit.minimize)
        try:
            oldfit.check_deadline=check_deadline;oldfit.save_provenance=provenance
            oldfit.minimize=feasible if data['dataset']=='SoftVTBench' else minimize
            theta=oldfit.fit(data,out,device='cpu',iterations=100,frames=36,**configuration)
        finally:oldfit.check_deadline,oldfit.save_provenance,oldfit.minimize=prior
        validate_checkpoint(data,out)
    prediction,diagnostic=rollout(data,theta,configuration,'cpu')
    temporary=out/'prediction.pending.npz';np.savez_compressed(temporary,predicted=prediction,nodes=data['nodes'],theta=theta);temporary.replace(out/'prediction.npz')
    diagnostic.update(family='field_free',settings=configuration,freeze_sha256=sha256(FREEZE),protocol_sha256=sha256(PROTOCOL))
    write_json(out/'prediction.json',diagnostic)
    write_json(out/'complete.json',dict(name=data['name'],family='field_free',fit_end=data['train_end'],validation=bool(validation),settings=configuration,
        elapsed_seconds=time.monotonic()-start,failed_frames=len(diagnostic['failed_frames']),checkpoint_sha256=sha256(out/'best.npz'),prediction_sha256=sha256(out/'prediction.npz'),freeze_sha256=sha256(FREEZE),protocol_sha256=sha256(PROTOCOL)))
    print('FIELD_FREE',data['name'],tag,'complete',len(diagnostic['failed_frames']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case');parser.add_argument('--subset');parser.add_argument('--task',type=int);parser.add_argument('--episode',type=int);parser.add_argument('--validation',action='store_true');parser.add_argument('--device',choices=['cpu'],default='cpu');args=parser.parse_args()
    spec=dict(case=args.case) if args.case else dict(subset=args.subset,task=args.task,episode=args.episode)
    run(spec,args.validation,args.device)
