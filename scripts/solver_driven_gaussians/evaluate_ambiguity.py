"""Choose a scale hypothesis from a sealed probe; test on a different force."""
import sys,json,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=1);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
 import torch,numpy as np
 from scripts.solver_driven_gaussians.rendering import MechanicalRenderer
 from scripts.solver_driven_gaussians.score_motion import metrics
 torch.set_num_threads(2);r=MechanicalRenderer();base=ROOT/'runs/solver-driven-gaussians-v02';folder=base/'ambiguity';seal=json.loads((folder/'probe-selection.json').read_text());assert not seal['probe_targets_read'];assert seal['selected_probe']=='force_pulse'
 for name,sha in seal['prediction_hashes'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
 # One predeclared 0.08-s observation, familiar view 0. Selection is not scored on the test action.
 truth=np.load(base/'reference/force_pulse.npz');idx=int(np.argmin(abs(truth['times']-.08)));errors=[]
 with torch.no_grad():
  target=r.image(truth['x'][idx],0);mask=(target-.96).abs().max(0).values>.03
  for factor in [1,2]:
   x=np.load(folder/f'factor{factor}-force_pulse.npz')['x'][idx];pred=r.image(x,0);errors.append(float((pred-target).square()[:,mask].mean()))
 chosen=int(np.argmin(errors))+1
 record=dict(selected_factor=chosen,probe='force_pulse',observation_time_s=.08,observation_view=0,probe_mse=errors,test_target_read_before_selection=False)
 (folder/'posterior-selection.json').write_text(json.dumps(record,indent=2))
 truth=np.load(base/'reference/force_reverse.npz');preds=[np.load(folder/f'factor{f}-force_reverse.npz')['x'] for f in [1,2]];keep=truth['times']>.14;rest=r.model.rest.cpu().numpy()
 results={f'factor{f}':metrics(x[keep],truth['x'][keep],rest) for f,x in zip([1,2],preds)};results['prior_mean']=metrics(np.mean(preds,axis=0)[keep],truth['x'][keep],rest);results['selected']=results[f'factor{chosen}']
 with torch.no_grad():
  x1=np.load(folder/'factor1-calibration.npz')['x'][40];x2=np.load(folder/'factor2-calibration.npz')['x'][40];identity=float((r.image(x1,2)-r.image(x2,2)).abs().max())
 record.update(test_action='force_reverse',results=results,new_view_max_rgb_difference=identity,probe_selection=seal,scope='Constructed two-hypothesis mass/stiffness scale ambiguity, not a calibrated posterior or general policy benchmark. Primary experiments assume known density. Common mass/stiffness/load/contact scaling leaves position-driven videos unchanged. One measured-force probe resolves this constructed ambiguity; subsequent test uses a distinct force.')
 (ROOT/'reports/solver-driven-gaussians-v02/observability.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
