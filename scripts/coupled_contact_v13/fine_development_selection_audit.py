"""Read-only reconstruction of two development FEM calibration minima."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/coupled-friction-v13';REPORT=ROOT/'reports/coupled-friction-v13'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((REPORT/'protocol.json').read_text());evidence={};candidates=[]
 def bind(path):evidence[str(path.relative_to(ROOT))]=sha(path)
 for i,E in enumerate(p['fem_young_grid']):
  for j,mu in enumerate(p['friction_grid']):
   folder=BASE/'library/fem'/f'{i}-{j}';status=json.loads((folder/'complete.json').read_text());bind(folder/'complete.json')
   frames=None
   if status['success']:
    bind(folder/'trajectory.npz')
    with np.load(folder/'trajectory.npz',allow_pickle=False) as a:frames=a['frames'].copy()
   candidates.append((E,mu,frames))
 rows=[]
 for obj in p['objects']:
  if obj['split']!='development':continue
  folder=BASE/'inference'/obj['id'];bind(folder/'selection.json');bind(folder/'observations.npz');selected=json.loads((folder/'selection.json').read_text())['fem']
  with np.load(folder/'observations.npz',allow_pickle=False) as a:observed=a['observed'].copy();mask=a['mask'].copy()
  scores=np.array([np.mean((frames[mask]-observed[mask])**2) if frames is not None else np.inf for _,_,frames in candidates]);index=int(np.argmin(scores))
  np.testing.assert_allclose(scores,selected['scores_m2'],rtol=1e-12,atol=1e-18)
  if index!=selected['index'] or candidates[index][:2]!=(selected['scale'],selected['friction']):raise RuntimeError('Selected minimum identity differs')
  np.testing.assert_allclose(np.sqrt(scores[index]),selected['calibration_rmse_m'],rtol=1e-12,atol=1e-15)
  rows.append(dict(object=obj['id'],selected_young_pa=selected['scale'],selected_friction=selected['friction'],calibration_rmse_mm=1000*np.sqrt(scores[index]),all119_losses_reconstructed=True,first_argmin_reconstructed=True,top3=[dict(young_pa=candidates[k][0],friction=candidates[k][1],rmse_mm=1000*np.sqrt(scores[k])) for k in np.argsort(scores,kind='stable')[:3]]))
 if len(rows)!=2 or len(candidates)!=119:raise RuntimeError('Fixed development coverage differs')
 for name in ('protocol.json','method-freeze.json'):bind(REPORT/name)
 report=dict(passed=True,scope='Two development calibration minima only; no forecast targets, reserved observations or simulation accessed',cases=rows,input_sha256=evidence,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},conclusion='Selection identities and losses confirmed; no forecast improvement inferred from calibration alone')
 (REPORT/'development-selection-independent-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
