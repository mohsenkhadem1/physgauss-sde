"""Frozen PhysTwin forecasts and new-fit provenance, then separate evaluation."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.benchmark_improvement_v14.common import BASE,ROOT,REPORT,read,write,sha,remaining
from scripts.benchmark_improvement_v14.provenance import snapshot_sources,validate_record
from scripts.benchmark_improvement_v14.evaluate import verify_freeze,cases,score_all
from scripts.benchmark_improvement_v14.phystwin_production import directory,validate_fit,select
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.contact_benchmark_v12.experiment import finite_metadata
from scripts.contact_benchmark_v12.feasibility import inherited_elements
from scripts.contact_benchmark_v12.feasible_experiment import topology_audit


def fit_sources(name):
    result={}
    for tag in ['full','short']:
        folder=directory(name,tag)
        if (folder/'complete.json').exists():
            validate_fit(name,tag)
            for path in folder.iterdir():
                if path.is_file() and not path.name.endswith('.tmp'):result[str(path)]=sha(path)
        else:
            job=BASE/'phystwin-jobs'/(name+'-'+tag+'.json')
            if not job.exists() or read(job)['state']!='failed':return None
            result[str(job)]=sha(job)
    return result


def validate_new(case,out):
    record=read(out/'complete.json');validate_record(case,record,out)
    if record['freeze_sha256']!=verify_freeze('PhysTwin') or record['new_fit_sources']!=fit_sources(case['name']):
        raise ValueError('PhysTwin prediction/fit provenance changed')
    return True


def forecast(case):
    remaining();frozen=verify_freeze('PhysTwin');name=case['name'];out=BASE/'evaluation'/case['cohort']/name
    if (out/'complete.json').exists():return validate_new(case,out)
    sources=snapshot_sources(case);new_sources=fit_sources(name)
    if new_sources is None:return False
    folder=ROOT/'runs/prefix-ridge-switch-v13/inputs'/name
    data=load_payload(folder/'full.npz');short_data=load_payload(folder/'short.npz')
    original=ROOT/'runs/contact-benchmark-v12/evaluation'/case['cohort']/name
    with np.load(original/'predictions.npz') as saved:predictions={key:saved[key].copy() for key in saved.files}
    original_meta=read(original/'prediction-metadata.json')['object_frame']
    if all((directory(name,tag)/'complete.json').exists() for tag in ['full','short']):
        full=np.load(directory(name,'full')/'prediction.npz')['predicted']
        short=np.load(directory(name,'short')/'prediction.npz')['predicted']
        oldshort=np.load(ROOT/'runs/contact-benchmark-v12/base/validation'/name/'prediction.npz')['predicted']
        proposal,metadata=select(data,short_data,full,short,predictions['object_frame'],oldshort,original_meta)
        predictions['attachment_width_fem']=full
    else:
        proposal=predictions['object_frame'].copy()
        metadata=dict(switched=False,retained_original=True,reason='Width fit unavailable; explicit original-model fallback')
    predictions['benchmark_v14']=proposal
    for directory_name,key in [('model-selection-evaluation','model_selection_selected'),('feasibility-evaluation','feasible_object_frame')]:
        with np.load(ROOT/'runs/contact-benchmark-v12'/directory_name/case['cohort']/name/'predictions.npz') as saved:
            predictions[key]=saved[key].copy()
    elements=inherited_elements(data)
    metadata['topology']={key:topology_audit(data,predictions[key],elements) for key in ['object_frame','benchmark_v14']+(['attachment_width_fem'] if 'attachment_width_fem' in predictions else [])}
    if snapshot_sources(case)!=sources or fit_sources(name)!=new_sources:raise ValueError('Inputs changed during forecast')
    verify_freeze('PhysTwin');out.mkdir(parents=True,exist_ok=True)
    pending=out/'predictions.pending.npz';np.savez_compressed(pending,**predictions);pending.replace(out/'predictions.npz')
    write(out/'metadata.json',finite_metadata(metadata))
    write(out/'complete.json',dict(case=case,sources=sources,new_fit_sources=new_sources,freeze_sha256=frozen,
        prediction_sha256=sha(out/'predictions.npz'),metadata_sha256=sha(out/'metadata.json')))
    print('PhysTwin forecast',name,'switched',metadata.get('switched'),flush=True);return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--score',action='store_true');a=p.parse_args()
    if a.score:
        for case in cases('PhysTwin'):validate_new(case,BASE/'evaluation'/case['cohort']/case['name'])
        score_all('PhysTwin')
