"""Independent public-calibration loss/selection audit; no forecast targets."""
from pathlib import Path
import json,hashlib,time
import numpy as np
R=Path(__file__).resolve().parents[2];D=R/'reports/compositional-dynamics-v17';B=R/'runs/compositional-dynamics-v17'
def main():
 ev={};errors=[];checks=[]
 def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 def read(p):ev[str(p.relative_to(R))]=h(p);return json.loads(p.read_text())
 def ck(v,s):checks.append(s);errors.extend([]if v else[s])
 planpath=D/'matched-material-fine-fit-plan.json';p=read(planpath);reportpath=D/'matched-material-fine-fit-results.json';d=read(reportpath);raw=B/'matched-material-fine-fit';commit=read(raw/'fitted-parameter-commit.json')
 ck(d['terminal']and d['search_complete'],'terminal complete search')
 for table in ('source_sha256','input_sha256'):
  for n,v in p[table].items():ck(h(R/n)==v,'pin '+n)
 with np.load(B/'calibration-public/observations.npz')as a:obs=a['object_points'].astype(float);mask=a['object_visibilities'][1:]
 losses={};counts={};ids=set()
 for row in d['rows']:
  task=row['task'];key=task['id'];ck(key not in ids,'unique '+key);ids.add(key);folder=raw/key;ck(read(folder/'complete.json')==row,'status '+key);ck(row['plan_sha256']==h(planpath),'plan '+key);counts.setdefault(str(int(task['flag'])),[0,0])[0]+=1
  if row['completed']:
   path=folder/'prediction.npz';ck(h(path)==row['prediction_sha256'],'prediction '+key);ev[str(path.relative_to(R))]=h(path)
   with np.load(path)as a:x=a['positions_m'].astype(float);t=a['times_s']
   ck(x.shape==(21,98,3)and np.isfinite(x).all(),'finite '+key);ck(np.allclose(t,np.arange(21)/100,atol=1e-12,rtol=0),'clock '+key);res=x[1:][mask]-obs[1:][mask];loss=float(np.sum(res*res)/res.size);losses[key]=loss;ck(np.isclose(loss,row['loss_m2'],rtol=1e-10,atol=1e-18),'loss '+key);ck(len(res)==1695==row['observed_point_count'],'denominator '+key);counts[str(int(task['flag']))][1]+=1
 ck(len(ids)==64 and counts=={'-1':[32,32],'1':[32,32]},'64 complete balanced candidates')
 for s in d['refinements']:
  ck(s['evaluations']==32 and len(s['records'])==32,'family budget');ck(set(s['records'])<=ids,'family records');start=next(x for x in p['starts']if x['flag']==s['flag']);ck(s['initial_parameters']==start['parameters'],'declared start')
 best=min(losses,key=losses.get);ck(d['best_training_candidate']['task']['id']==best==commit['task']['id'],'selected minimum public loss');ck(commit['results_sha256']==h(reportpath)and commit['plan_sha256']==h(planpath),'parameter commit binding')
 result=dict(passed=not errors,errors=errors,checks=len(checks),created_unix=time.time(),evidence_sha256=ev,selected_task=commit['task'],coordinate_rmse_mm=1000*np.sqrt(losses[best]),family_counts=counts,optimizer_reports=d['refinements'],scope='Public masked frames1..20 only; independently recomputed all64 candidate losses. Search exhaustion is distinct from optimizer convergence. No private forecast response accessed.')
 (D/'matched-material-fine-fit-final-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ('passed','checks','errors','coordinate_rmse_mm')}),flush=True)
if __name__=='__main__':main()
