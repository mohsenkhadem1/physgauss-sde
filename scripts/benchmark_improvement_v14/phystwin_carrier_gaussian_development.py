"""One fixed development carrier hypothesis; direct-support and nodal audit."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.benchmark_improvement_v14.common import REPORT,BASE,write,sha
from scripts.benchmark_improvement_v14.phystwin_carrier_gaussian import predict,SETTINGS,basis
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.contact_benchmark_v12.data import load
from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
from scripts.contact_benchmark_v12.feasibility import inherited_elements,jacobians

def main():
 name='single_lift_sloth';folder=ROOT/'runs/prefix-ridge-switch-v13/inputs'/name;paths={tag:ROOT/'runs/contact-benchmark-v12/base'/tag/name/'prediction.npz' for tag in ['full','validation']}
 write(REPORT/'phystwin-carrier-gaussian-plan.json',dict(case=name,settings=SETTINGS,scope='One fixed development candidate before execution; directsupport positions; no other7suffix',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('phystwin_carrier_gaussian*.py')},input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths.values()}))
 full=load_payload(folder/'full.npz');short=load_payload(folder/'short.npz');end=full['train_end'];cut=short['train_end']
 for k in ('times','controls','controller_targets','attachment_active'):short[k]=short[k][:end]
 initial=np.load(paths['full'])['predicted'][0];si=np.load(paths['validation'])['predicted'][0]
 p,meta=predict(full,initial,full['support_geometry']);v,vm=predict(short,si,full['observed_initial']);truth=full['observed'][cut:end];mask=full['observed_mask'][cut:end]&np.isfinite(truth).all(2);validation=float(np.sqrt(np.mean((v[cut:end]-truth)[mask]**2))*1000)
 # Evaluator imported/loaded only after this candidate and prefix score are fixed.
 data=load({'case':name});score=safe_score(data,p,end,len(p))['summary'];nodal,nm=predict(full,initial);e=inherited_elements(full);js=np.array([jacobians(full['nodes'],x,e) for x in nodal]);same,_=predict(full,initial,full['nodes']);np.testing.assert_array_equal(nodal,same)
 out=BASE/'phystwin-carrier-gaussian-development';out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'predictions.npz',support=p,nodal=nodal)
 result=dict(validation_rmse_mm=validation,suffix=score,topology=dict(inverted_frames=int(np.any(js<=0,axis=1).sum()),minimum_J=float(js.min())),direct_node_query_exact=True,full_metadata=meta,short_metadata=vm,original_manual_mm=34.07648095788899,original_SHORT_mm=13.938729085992833,prediction_sha256=sha(out/'predictions.npz'))
 write(REPORT/'phystwin-carrier-gaussian-development.json',result);print(json.dumps(dict(validation_mm=validation,suffix=score,topology=result['topology'])),flush=True)
if __name__=='__main__':main()
