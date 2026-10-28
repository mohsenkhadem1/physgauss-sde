"""Development-only expected versus realized value for every candidate action."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.v08_study import BASE,OUT,Experiment,interpolators,fit,protocol,write

def main():
 s=protocol();interp=interpolators();bias,C=fit(interp);ex=Experiment(interp,s['planning_grid'],bias,C);z=np.load(BASE/'development-observations.npz');rows=[]
 for i,(y,truth) in enumerate(zip(z['measurements'],z['truth'])):
  path=BASE/'diagnosis'/f'bayes-{i}.json'
  if path.exists():rows.append(json.loads(path.read_text()));continue
  risks,current=ex.choose([],y,s['utility_samples']);before=ex.predict([],y)['mean'][0];mse0=float(np.mean((before-truth)**2));actual=[]
  for a in range(7):actual.append(float(np.mean((ex.predict([a],y)['mean'][0]-truth)**2)))
  expected=current-risks;realized=mse0-np.array(actual);rank=float(spearmanr(expected,realized).statistic)
  row=dict(object=i//3,repeat=i%3,current_variance_m2=current,initial_mse_m2=mse0,expected_gain_m2=expected.tolist(),realized_gain_m2=realized.tolist(),action_mse_m2=actual,spearman=rank,expected_best=int(expected.argmax()),realized_best=int(realized.argmax()));write(path,row);rows.append(row);print('DIAGNOSIS',i,flush=True)
 out=dict(scope='12 development objects × 3 noise repeats; all six force actions and new end view. Realized value is one noisy observation outcome, not expected value; oracle action uses development held-load truth only. No test action tuning.',mean_spearman=float(np.mean([r['spearman'] for r in rows])),top_action_agreement=float(np.mean([r['expected_best']==r['realized_best'] for r in rows])),mean_expected_gain_mm2=(1e6*np.mean([r['expected_gain_m2'] for r in rows],axis=0)).tolist(),mean_realized_gain_mm2=(1e6*np.mean([r['realized_gain_m2'] for r in rows],axis=0)).tolist(),rows=rows)
 write(OUT/'development-action-value.json',out)
if __name__=='__main__':main()
