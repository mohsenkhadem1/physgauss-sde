"""Post-audit model-class comparison, isolated from the frozen primary studies."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,check_deadline,sha256,write_json
from scripts.contact_benchmark_v12.data import load,trim_prefix,predictor_payload
from scripts.contact_benchmark_v12.experiment import base_path,finite_metadata
from scripts.contact_benchmark_v12.feasible_experiment import COHORTS,specifications,topology_audit
from scripts.contact_benchmark_v12.feasibility import inherited_elements
from scripts.contact_benchmark_v12.scene import load_scene
from scripts.external_benchmark.evaluate import score,expand_prediction

FREEZE=REPORT/'model-selection-freeze.json'

def check_freeze():
    frozen=json.loads(FREEZE.read_text())
    assert frozen['protocol_sha256']==sha256(REPORT/'model-selection-protocol.json')
    for path,digest in frozen['source_sha256'].items():assert sha256(ROOT/path)==digest,path

def run(spec,cohort):
    from scripts.contact_benchmark_v12.model_selection import select_model
    check_deadline();check_freeze()
    name=spec.get('case') or f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"
    out=BASE/'model-selection-evaluation'/cohort/name
    if (out/'metrics.json').exists():return True
    guarded=BASE/'feasibility-evaluation'/cohort/name
    original=BASE/'evaluation'/cohort/name
    dirs={tag:BASE/'model-selection-base'/tag/name for tag in ['full','validation']}
    if not (guarded/'metrics.json').exists() or any(not (p/'complete.json').exists() for p in dirs.values()):return False
    data=load(spec);payload=predictor_payload(data)
    short=predictor_payload(trim_prefix(data,max(3,int(data['train_end']*.7))))
    sources={'current_full':base_path(data,cohort),'current_validation':base_path(data,cohort,True),
             'field_free_full':dirs['full']/'prediction.npz','field_free_validation':dirs['validation']/'prediction.npz',
             'default_prediction':guarded/'predictions.npz','default_metadata':original/'prediction-metadata.json'}
    families={family:{tag:np.load(sources[family+'_'+tag])['predicted'] for tag in dirs} for family in ['current','field_free']}
    default_meta=json.loads(sources['default_metadata'].read_text())['object_frame']
    primary='feasible_contact_scene_state' if data['dataset']=='SoftVTBench' else 'feasible_object_frame'
    default=np.load(sources['default_prediction'])[primary]
    scene=load_scene(payload) if data['dataset']=='SoftVTBench' else None
    predictions,metadata=select_model(payload,short,families,default_meta,default,scene=scene)
    predictions={'model_selection_'+k:v for k,v in predictions.items()}
    elements=inherited_elements(payload)
    metadata['topology']={k:topology_audit(payload,v,elements) for k,v in predictions.items()}
    out.mkdir(parents=True,exist_ok=True)
    # Persist every candidate before accessing suffix evaluation targets.
    np.savez_compressed(out/'predictions.npz',**predictions)
    write_json(out/'prediction-metadata.json',finite_metadata(metadata))
    fit_records={}
    for tag,directory in dirs.items():
        complete=json.loads((directory/'complete.json').read_text())
        fit_path=directory/'fit.json'
        fit=json.loads(fit_path.read_text()) if fit_path.exists() else {}
        fit_records[tag]=dict(completion=complete,optimizer=fit.get('optimizer'),fit_sha256=sha256(fit_path) if fit_path.exists() else None)
    old=json.loads((guarded/'metrics.json').read_text())
    results=old['results'].copy()
    results.update({k:score(data,expand_prediction(data,v),data['train_end'],len(data['times'])) for k,v in predictions.items()})
    write_json(out/'metrics.json',dict(name=name,dataset=data['dataset'],cohort=cohort,scope='post-audit exploratory; no new independent confirmation',
        primary_method='model_selection_selected',results=results,interval=[data['train_end'],len(data['times'])],
        sources={str(p.relative_to(ROOT)):sha256(p) for p in sources.values()},
        prediction_sha256=sha256(out/'predictions.npz'),metadata_sha256=sha256(out/'prediction-metadata.json'),
        prior_metrics_sha256=sha256(guarded/'metrics.json'),freeze_sha256=sha256(FREEZE),field_free_fit_diagnostics=fit_records,
        protocol_sha256=sha256(REPORT/'model-selection-protocol.json')))
    print('Model selection',cohort,name,'complete',flush=True);return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',choices=COHORTS,required=True);p.add_argument('--watch',action='store_true');p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1);a=p.parse_args()
    specs=specifications(a.cohort)[a.part::a.parts]
    while True:
        done=sum(run(s,a.cohort) for s in specs)
        print('Model selection cohort',a.cohort,done,'/',len(specs),flush=True)
        if done==len(specs) or not a.watch:break
        check_deadline();time.sleep(20)

if __name__=='__main__':main()
