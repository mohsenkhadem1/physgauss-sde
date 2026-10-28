"""Post-test diagnosis only; does not change the sealed v05 model or decisions."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.qualified_policy import verify_seal
ROOT=Path(__file__).resolve().parents[2]

def main():
 b=ROOT/'runs/phystwin-qualified-support-v05';out=ROOT/'reports/phystwin-sequential-support-v06';out.mkdir(parents=True,exist_ok=True)
 verify_seal(json.loads((b/'analysis/seal.json').read_text()))
 spec=json.loads((b/'protocol.json').read_text());utility=json.loads((b/'analysis/test-utility.json').read_text());oracle=json.loads((b/'analysis/oracle-audit.json').read_text());rows=[]
 with np.load(b/'analysis/test-predictions.npz') as z:
  for i,action in enumerate(z['adaptive']):
   obj=i//spec['noise_repeats'];r=spec['test'][obj];layout=spec['layouts'].index(r['layout']);prob=z['0_layout_probability'][i]
   with np.load(b/'fields'/f'test-{obj}-qualified.npz') as f:nonrigid=f['nonrigid'][action,0]
   if nonrigid>=1e-9:continue
   sideprob=float(prob[:2].sum()) if layout<2 else float(prob[2:].sum())
   rows.append(dict(object=obj,repeat=i%spec['noise_repeats'],true_layout=r['layout'],chosen_action=int(action),heuristic_action=int(z['heuristic'][i]),true_layout_probability=float(prob[layout]),true_side_probability=sideprob,
     initial_map_side_correct=sideprob>=.5,nonrigid_rms_m=float(nonrigid),predicted_action_risks_m2=utility[i]['risks'][-1],privileged_action_risks_m2=oracle['risks'][i],
     realized_action_rmse_mm=[float(1000*np.sqrt(np.mean((z[f'{a}_mean'][i]-z['truth'][i])**2))) for a in [1,2]]))
 result=dict(scope='Post-evaluation diagnostic; truth and oracle are used only to explain saved decisions, never to retune v05.',uninformative_probes=len(rows),observations=len(utility),incorrect_initial_side_count=sum(not r['initial_map_side_correct'] for r in rows),rows=rows)
 (out/'v05-action-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
