"""Post-hoc quadrature audit of saved decisions, never changes the policy history."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.v07_study import BASE,OUT,Experiment,interpolators,fit,protocol,select,write
from scripts.phystwin.sequential_policy import verify
s=protocol();verify(json.loads((BASE/'seal.json').read_text()));interp=interpolators();bias,C=fit(interp);ex=Experiment(interp,65,bias,C);plans=json.loads((BASE/'plans/test-cost_aware.json').read_text());y=np.load(BASE/'test-observations.npz')['measurements'];rows=[]
for obj in range(6):
 i=obj*3;p=plans[i];history=[]
 for step,decision in enumerate(p['decisions']):
  risks,current=ex.choose(history,y[i],s['utility_samples']);choice=select(risks,current,s['costs'],s['cost_penalty_m2']);row=dict(object=obj,step=step,history=history.copy(),saved_choice=decision['chosen'],refined_choice=choice,agreement=choice==decision['chosen'],current_variance_m2=current,expected_variances_m2=[None if not np.isfinite(v) else float(v) for v in risks]);rows.append(row);print('ACTION_AUDIT',obj,step,row['agreement'],flush=True)
  if decision['chosen']<0:break
  history.append(decision['chosen'])
write(OUT/'action-refinement.json',dict(scope='First six test objects, first noise repeat; 33-to-65 cubed per-layout quadrature at identical 128 outcome samples. Evaluates saved histories, never substitutes refined actions. Not a full action or outcome-sampling convergence proof.',agreements=sum(r['agreement'] for r in rows),decisions=len(rows),rows=rows))
