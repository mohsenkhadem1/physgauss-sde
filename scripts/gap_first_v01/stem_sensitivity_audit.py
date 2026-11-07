"""Truth-free pre-outcome numerical sensitivity at declared observations."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
import numpy as np
from .stem_inference import plan

def one(job):
 c,y,condition,grid,bins=job;start=time.time()
 _,_,r=plan(y,condition,.12,c,grid_size=grid,bins=bins)
 return dict(initial_y_m=y,condition=condition,grid=grid,bins=bins,
  current_risk=r['predictions']['posterior_mean_risk'],selected_action=r['selected_action'],
  objectives={k:v['objective'] for k,v in r['action_scores'].items()},seconds=time.time()-start)

def main():
 p=Path(__file__).with_name('stem_protocol.json');c=json.loads(p.read_text());configs=[(129,161),(257,161),(513,161),(513,321),(513,641)]
 jobs=[(c,y,condition,n,bins) for y in [.0032,.0104,.0176] for condition in c['conditions'] for n,bins in configs]
 out=Path('reports/gap-first-v01/stem-development-sensitivity.json')
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  rows=[]
  for r in pool.map(one,jobs):rows.append(r);print(r['initial_y_m'],r['condition'],r['grid'],r['bins'],round(r['seconds'],2),flush=True)
 comparisons=[]
 for y in [.0032,.0104,.0176]:
  for condition in c['conditions']:
   group=[r for r in rows if r['initial_y_m']==y and r['condition']==condition];ref=group[-1]
   for r in group[:-1]:
    values={'proceed':ref['current_risk'],**ref['objectives']};best=min(values.values())
    comparisons.append(dict(initial_y_m=y,condition=condition,grid=r['grid'],bins=r['bins'],reference_grid=513,reference_bins=641,
     decision_agreement=r['selected_action']==ref['selected_action'],chosen_excess_reference_objective=values[r['selected_action']]-best,
     current_risk_difference=r['current_risk']-ref['current_risk'],maximum_action_objective_difference=max(abs(r['objectives'][a]-ref['objectives'][a]) for a in ref['objectives'])))
 doc=dict(scope='Development numerical diagnostic; predetermined observations, no hiddencohorttruth or outcomes accessed. Same current source beforeoutcomeevaluation.',initial_y_m=[.0032,.0104,.0176],rows=rows,comparisons=comparisons,
  source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p,Path(__file__),Path(__file__).with_name('stem_inference.py'),Path(__file__).with_name('quantiles.py')]})
 out.write_text(json.dumps(doc,indent=2)+'\n');print('SAVED',out,flush=True)
if __name__=='__main__':main()
