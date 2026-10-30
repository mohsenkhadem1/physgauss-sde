"""Sequential first-case input-only replay, never loads evaluator targets/scores."""
from pathlib import Path
import json,sys,time,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from scripts.prefix_ridge_switch_v13.run import BASE,REPORT,PARENT,read,write,sha,export,forecast
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.contact_benchmark_v12.experiment import base_path
from scripts.contact_benchmark_v12.model_selection import _validation_window,_predict,_score,_topology_failures,rigid_nodal
from scripts.contact_benchmark_v12.feasibility import inherited_elements
from scripts.contact_benchmark_v12.scene import load_scene

def run():
    p=read(REPORT/'protocol-pending.json');cases=[next(c for c in p['cases'] if 'case' not in c['spec']),next(c for c in p['cases'] if 'case' in c['spec'])];rows=[]
    for case in cases:
        start=time.monotonic();export(case);metadata=forecast(case);name=case['name'];folder=BASE/'inputs'/name
        full,short=load_payload(folder/'full.npz'),load_payload(folder/'short.npz');end=full['train_end'];cut=short['train_end'];short=_validation_window(short,end)
        parent=PARENT/'model-selection-evaluation'/case['cohort']/name;old=read(parent/'prediction-metadata.json');chosen=old['selected_configuration'];config=(chosen['family'],chosen['method'],chosen['strength']);elements=inherited_elements(short);scene=load_scene(full) if full['dataset']=='SoftVTBench' else None
        if config[0]=='rigid':anchor=rigid_nodal(short)
        else:
            path=base_path(full,case['cohort'],True) if config[0]=='current' else PARENT/'model-selection-base/validation'/name/'prediction.npz'
            with np.load(path) as f:anchor=f['predicted'][:end].copy()
        pred,_=_predict(short,anchor,config,scene,elements);value,bad=_score(full,pred,short,cut,end)
        if _topology_failures(short,pred,cut,end,elements):value=float('inf')
        expected=chosen['validation_rmse_m'] if chosen['validation_finite'] else float('inf')
        np.testing.assert_allclose(value,expected,rtol=1e-10,atol=1e-12)
        with np.load(parent/'predictions.npz') as f:archived=f['model_selection_selected'].copy()
        if old['switched_from_previous']:
            if config[0]=='rigid':anchor=rigid_nodal(full)
            else:
                path=base_path(full,case['cohort']) if config[0]=='current' else PARENT/'model-selection-base/full'/name/'prediction.npz'
                with np.load(path) as f:anchor=f['predicted'].copy()
            rebuilt,_=_predict(full,anchor,config,scene,elements);bad=_topology_failures(full,rebuilt,0,len(rebuilt),elements)
            if bad:rebuilt[bad]=np.nan
        else:
            path=PARENT/'feasibility-evaluation'/case['cohort']/name/'predictions.npz'
            with np.load(path) as f:rebuilt=f['feasible_contact_scene_state' if full['dataset']=='SoftVTBench' else 'feasible_object_frame'].copy()
        np.testing.assert_allclose(rebuilt,archived,rtol=1e-10,atol=1e-12,equal_nan=True)
        rows.append({'case':case,'seconds':time.monotonic()-start,'old_short_validation_reproduced':True,'old_full_chosen_reproduced':True,'selected_guarded_ridge':metadata['switched'],'prefix_only_input_contract_verified':True,'input_sha256':metadata['input_sha256'],'scope':'No suffix target or score read; forecast arrays are predictions only.'})
        write(REPORT/'input-only-pilot.json',{'passed':len(rows)==2,'protocol_sha256':sha(REPORT/'protocol-pending.json'),'cases':rows,'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'scripts/prefix_ridge_switch_v13').glob('*.py')}})
    return rows
if __name__=='__main__':run()
