"""Rank candidate loads from initial-only PhysTwin prediction disagreement."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.v08_reference import BASE,OUT,ACTIONS,write
from scripts.phystwin.sequential_fixture import points

def select(split,obj):
 # Match the sparse force-response marker locations in the Bayesian policy.
 x=points();probes=[[0,-.04,-.04],[.24,-.04,.04],[0,-.04,.04],[.24,-.04,-.04],[0,-.04,.04],[.24,-.04,0]];ids=[np.argmin(np.linalg.norm(x-p,axis=1)) for p in probes];variance=[]
 for a,name in enumerate(ACTIONS[1:7]):
  fields=[]
  for member in range(3):
   path=BASE/'phystwin'/f'{split}-{obj}-member{member}'
   summary=json.loads((path/'summary-dt1.json').read_text());assert summary['fit_actions']==['initial'] and summary['visible_markers']==3
   with np.load(path/f'{name}-dt1.npz') as z:fields.append(z['predicted'][z['hold']].mean(0)[ids[a]])
  variance.append(float(np.var(fields,axis=0,ddof=1).mean()))
 history=np.argsort(-np.array(variance),kind='stable')[:2].tolist();value=dict(object=obj,split=split,history=history,variance_m2=variance,scope='Three PhysTwin fits initialized at spring Y 0.05/0.1/0.2. Same nine initial RGB-D coordinates as Bayes; no candidate response or held-target data used. Rank disagreement at each candidate observed marker; select two distinct actions as a batch, without retraining between them. Heuristic disagreement, not calibrated uncertainty or expected improvement. 2700 extra selection updates per object.')
 write(BASE/'plans'/f'{split}-ensemble-{obj}.json',value);return value

def diagnose():
 rows=[]
 from scipy.stats import spearmanr
 for obj in range(3):
  plan=json.loads((BASE/'plans'/f'development-ensemble-{obj}.json').read_text());prefix=BASE/'phystwin'
  control=json.loads((prefix/f'development-{obj}-dev_repeat/summary-dt1.json').read_text())['results']['held_oblique']['hold_rmse_mm']
  errors=[json.loads((prefix/f'development-{obj}-dev_action{a}/summary-dt1.json').read_text())['results']['held_oblique']['hold_rmse_mm'] for a in range(6)];gain=control**2-np.square(errors)
  rows.append(dict(object=obj,control_rmse_mm=control,action_rmse_mm=errors,realized_gain_mm2=gain.tolist(),disagreement_mm2=(1e6*np.array(plan['variance_m2'])).tolist(),spearman=float(spearmanr(plan['variance_m2'],gain).statistic),predicted_best=plan['history'][0],realized_best=int(np.argmin(errors))))
 write(OUT/'phystwin-development-action-value.json',dict(scope='Three development objects, one per layout, all six single-force candidates plus repeated-initial control. Each response fit has 61 frames, 27 visible markers and 1800 updates. Selection ensemble sees only the shared sparse prefix. No test-data retuning.',rows=rows,mean_spearman=float(np.mean([r['spearman'] for r in rows]))))
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',default='test');p.add_argument('--object',type=int);p.add_argument('--diagnose',action='store_true');a=p.parse_args()
 if a.diagnose:diagnose()
 else:select(a.split,a.object)
if __name__=='__main__':main()
