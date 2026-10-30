"""Development calibration loss audit; never reads reserved targets or simulates."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/coupled-surface-v13';REPORT=ROOT/'reports/coupled-surface-v13'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 protocol=json.loads((REPORT/'protocol.json').read_text());inputs={'reports/coupled-surface-v13/protocol.json':digest(REPORT/'protocol.json')};candidates=[]
 for i,E in enumerate(protocol['fem_young_grid']):
  for j,mu in enumerate(protocol['friction_grid']):
   path=BASE/'library/fem'/f'{i}-{j}';status=json.loads((path/'complete.json').read_text());frames=None
   inputs[str((path/'complete.json').relative_to(ROOT))]=digest(path/'complete.json')
   if status['success']:
    inputs[str((path/'trajectory.npz').relative_to(ROOT))]=digest(path/'trajectory.npz')
    with np.load(path/'trajectory.npz',allow_pickle=False) as d:frames=d['frames'].copy()
   candidates.append(dict(young=E,friction=mu,frames=frames))
 rows=[]
 for obj in protocol['objects']:
  if obj['split']!='development':continue
  oid=obj['id'];paths=[BASE/'inference'/oid/'selection.json',BASE/'inference'/oid/'observations.npz',BASE/'reference'/oid/'squeeze/trajectory.npz']
  for p in paths:inputs[str(p.relative_to(ROOT))]=digest(p)
  selected=json.loads(paths[0].read_text())['fem']
  with np.load(paths[1],allow_pickle=False) as d:obs=d['observed'].copy();mask=d['mask'].copy();rest=d['rest'].copy();times=d['times'].copy()
  with np.load(paths[2],allow_pickle=False) as d:
   distances,ids=cKDTree(d['rest']).query(rest)
   if distances.max()>1e-10 or not np.allclose(d['times'],times):raise RuntimeError('Calibration mapping mismatch')
   noiseless=d['frames'][:,ids].copy()
  def losses(target,use):return np.array([float(np.mean((c['frames'][use]-target[use])**2)) if c['frames'] is not None else np.inf for c in candidates])
  noisy=losses(obs,mask);clean=losses(noiseless,mask)
  if not np.allclose(noisy,selected['scores_m2'],rtol=1e-12,atol=1e-18):raise RuntimeError('Frozen selection loss reconstruction failed')
  def ranking(scores):
   return [dict(young=candidates[k]['young'],friction=candidates[k]['friction'],rmse_mm=float(np.sqrt(scores[k])*1000)) for k in np.argsort(scores)]
  phases={}
  # Boundaries come from already-frozen commanded compression/shear/release times.
  for name,use_time in [('compression',times<=.16),('shear',(times>.16)&(times<=.35)),('release',times>.35)]:
   use=mask&use_time[:,None];scores=losses(obs,use)
   phases[name]=dict(saved_frames=int(use_time.sum()),best=ranking(scores)[0],frozen_winner_rmse_mm=float(np.sqrt(scores[selected['index']])*1000))
  rows.append(dict(object=oid,true_development_young=obj['young'],true_development_friction=obj['friction'],noisy_ranking=ranking(noisy),noiseless_ranking=ranking(clean),phase_diagnostic=phases,
    best_stiffness_at_each_friction=[ranking(np.where(np.array([c['friction']==mu for c in candidates]),noisy,np.inf))[0] for mu in protocol['friction_grid']],
    selection_reconstructed=True))
 if len(rows)!=2:raise RuntimeError('Expected exactly two fixed development objects')
 source=Path(__file__);output=dict(scope='Read-only frozen development squeeze calibration audit; no reserved data, no forecast target reads, no new simulations',source_sha256={str(source.relative_to(ROOT)):digest(source)},input_sha256=inputs,young_grid=protocol['fem_young_grid'],friction_grid=protocol['friction_grid'],noise_std_mm=1000*protocol['observation_noise_std_m'],cases=rows,
 conclusion='A stiffness/friction tradeoff and unresolved between-grid friction are supported by calibration losses. These diagnostics do not prove that finer friction alone improves future prediction; that requires a separately declared matched development experiment.')
 (REPORT/'development-grid-diagnosis.json').write_text(json.dumps(output,indent=2)+'\n')
 print(json.dumps([{k:r[k] for k in ['object','phase_diagnostic','best_stiffness_at_each_friction']}|dict(noisy_top=r['noisy_ranking'][:3],noiseless_top=r['noiseless_ranking'][:3]) for r in rows],indent=2))
if __name__=='__main__':main()
