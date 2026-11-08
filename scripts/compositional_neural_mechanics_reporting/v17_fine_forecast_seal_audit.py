"""Read-only forecast provenance/public-contract audit, no reference response."""
from pathlib import Path
import json,hashlib,time
import numpy as np
R=Path(__file__).resolve().parents[2];D=R/'reports/compositional-dynamics-v17';B=R/'runs/compositional-dynamics-v17'
def main():
 ev={};errors=[];checks=[]
 def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 def rd(p):ev[str(p.relative_to(R))]=h(p);return json.loads(p.read_text())
 def ck(v,s):checks.append(s);errors.extend([]if v else[s])
 pp=D/'matched-material-fine-forecast-plan.json';p=rd(pp);fit=rd(B/'matched-material-fine-fit/fitted-parameter-commit.json');fit_audit=rd(D/'matched-material-fine-fit-final-independent-audit.json');out=B/'matched-material-fine-forecast';c=rd(out/'prediction-commit.json');s=rd(out/'complete.json');public=rd(B/'calibration-public/metadata.json')
 ck(fit_audit['passed'],'fit independently audited');ck(p['task']==fit['task']and p['n']==3 and p['dt_s']==50e-6,'same committed fit/discretization');ck(p['setup']==public['forecast_public_setup'],'public reset setup');ck(c['plan_sha256']==h(pp)and c['complete_sha256']==h(out/'complete.json'),'plan/complete seals');ck(c['completed']==s['completed'],'completion accounting');ck(s['fitted_parameter_commit_sha256']==h(B/'matched-material-fine-fit/fitted-parameter-commit.json'),'parameter seal')
 for table in ('source_sha256','input_sha256'):
  for n,v in p[table].items():ck(h(R/n)==v,'pin '+n)
 for file,key in [('trajectory.npz','trajectory_sha256'),('surface-prediction.npz','surface_prediction_sha256')]:ck(h(out/file)==c[key],file+'hash');ev[str((out/file).relative_to(R))]=h(out/file)
 with np.load(out/'trajectory.npz')as a,np.load(out/'surface-prediction.npz')as b:
  ck(np.isfinite(a['u']).all()and np.isfinite(a['ledger']).all()and np.isfinite(b['positions_m']).all(),'finite arrays');ck(np.allclose(b['times_s'],a['times_s'][::10],atol=1e-12,rtol=0),'sampled clock');ck(np.array_equal(b['material_queries'],np.asarray(public['material_queries'])),'public queries');ck(len(b['times_s'])==26 if c['completed']else len(b['times_s'])<26,'full/prefix frame census');ck(np.isclose(np.max(abs(a['ledger'][:,1]-s['initial_energy_J']))/s['energy_scale_J'],s['maximum_relative_energy_drift'],rtol=1e-10),'energy ledger')
 r=dict(passed=not errors,checks=len(checks),errors=errors,evidence_sha256=ev,created_unix=time.time(),prediction_completed=c['completed'],scope='Committed forecast source/input/parameter/array identity and public reset only. No private forecast target read; no accuracy or superiority claim.')
 (D/'matched-material-fine-forecast-independent-seal-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in ('passed','checks','errors','prediction_completed')}),flush=True)
if __name__=='__main__':main()
