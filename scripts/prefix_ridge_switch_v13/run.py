"""Isolated input export and forecast persistence; deliberately contains no scorer."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
BASE=ROOT/'runs/prefix-ridge-switch-v13';REPORT=ROOT/'reports/prefix-ridge-switch-v13';PARENT=ROOT/'runs/contact-benchmark-v12'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,data):path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')
def guard():
    p=read(REPORT/'protocol.json');f=read(REPORT/'method-freeze.json');audit=read(REPORT/'prefreeze-audit.json')
    if sha(REPORT/'protocol.json')!=f['protocol_sha256']:raise RuntimeError('Protocol changed')
    if audit.get('passed') is not True or audit.get('errors') or audit.get('pending') or audit.get('method_freeze_sha256')!=sha(REPORT/'method-freeze.json'):raise RuntimeError('Current independent prefreeze approval required')
    for name,h in f['source_sha256'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Source changed: '+name)
    deadline=datetime.datetime.fromisoformat(p['compute_bound']['original_absolute_deadline_utc'].replace('Z','+00:00'))
    remaining=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    if remaining<=0:raise RuntimeError('Original deadline reached')
    return p,remaining

def check_inputs(paths):
    frozen=read(REPORT/'method-freeze.json')['input_sha256']
    for path in paths:
        key=str(Path(path).relative_to(ROOT))
        if key not in frozen or sha(path)!=frozen[key]:raise RuntimeError('Input changed or unpinned: '+key)

def export(case):
    from scripts.prefix_ridge_switch_v13.inputs import load_pair,save_payload,save_scene
    full,short=load_pair(case['spec']);folder=BASE/'inputs'/case['name']
    save_payload(folder/'full.npz',full);save_payload(folder/'short.npz',short)
    scene=save_scene(folder/'scene.npz',full) if full['dataset']=='SoftVTBench' else None
    external=[Path(full['source'])]
    if scene is not None:
        for m in scene.meshes:external.extend([ROOT/m['provenance']['local_path'],ROOT/m['provenance']['config_path']])
    files=list(folder.glob('*.npz'))+list(folder.glob('*.json'))
    write(folder/'export-provenance.json',{'external_input_sha256':{str(f.relative_to(ROOT)):sha(f) for f in external},'source_sha256':sha(full['source']),'source':full['source'],'payload_sha256':{str(f.relative_to(ROOT)):sha(f) for f in files if f.name!='export-provenance.json'},'scope':'Exporter alone accesses raw source; forecast process only reads strict input artifacts. Raw PhysTwin pickle contains withheld arrays; manual file never opened.'})

def forecast(case):
    from scripts.prefix_ridge_switch_v13.inputs import load_payload,load_scene_payload
    from scripts.prefix_ridge_switch_v13.selector import select
    from scripts.contact_benchmark_v12.experiment import base_path
    from scripts.contact_benchmark_v12.feasibility import inherited_elements
    name,cohort=case['name'],case['cohort'];folder=BASE/'inputs'/name
    exported=read(folder/'export-provenance.json')
    for path,h in exported['payload_sha256'].items():
        if sha(ROOT/path)!=h:raise RuntimeError('Safe exported artifact changed: '+path)
    full,short=load_payload(folder/'full.npz'),load_payload(folder/'short.npz')
    saved=PARENT/'model-selection-evaluation'/cohort/name
    paths={'current_short':base_path(full,cohort,True),'current_full':base_path(full,cohort),'field_free_short':PARENT/'model-selection-base/validation'/name/'prediction.npz','old_metadata':saved/'prediction-metadata.json','old_forecast':saved/'predictions.npz'}
    arrays={}
    for key in ('current_short','current_full','field_free_short'):
        with np.load(paths[key],allow_pickle=False) as f:
            np.testing.assert_array_equal(f['nodes'],full['nodes']);arrays[key]=f['predicted'].copy()
    with np.load(paths['old_forecast'],allow_pickle=False) as f:old=f['model_selection_selected'].copy()
    scene=load_scene_payload(folder/'scene.npz') if full['dataset']=='SoftVTBench' else None
    pred,metadata=select(full,short,arrays['current_short'],arrays['current_full'],arrays['field_free_short'],read(paths['old_metadata']),old,scene,inherited_elements(short))
    out=BASE/'forecasts'/cohort/name;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'prediction.npz',predicted=pred)
    metadata.update(case=case,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*paths.values(),folder/'full.npz',folder/'full.json',folder/'short.npz',folder/'short.json',folder/'export-provenance.json',*([folder/'scene.npz',folder/'scene.json'] if scene is not None else [])]},prediction_sha256=sha(out/'prediction.npz'),full_end=full['train_end'],scope='Exploratory guarded nodal ridge switch, not new physical inference. All predictions saved before separate scoring.')
    write(out/'metadata.json',metadata);return metadata

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['export','forecast']);parser.add_argument('--index',type=int,required=True);a=parser.parse_args();p,_=guard();case=p['cases'][a.index]
    if a.stage=='export':
        export(case);record=read(BASE/'inputs'/case['name']/'export-provenance.json');check_inputs([ROOT/name for name in record['external_input_sha256']])
    else:
        record=forecast(case);check_inputs([ROOT/name for name in record['input_sha256'] if not name.startswith('runs/prefix-ridge-switch-v13/')])
    guard()
if __name__=='__main__':main()
