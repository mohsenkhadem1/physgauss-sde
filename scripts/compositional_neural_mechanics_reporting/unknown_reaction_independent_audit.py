"""Independent terminal-gated audit of fixed-parameter secondary wrench replay."""
import json,itertools,time
import numpy as np
from scripts.compositional_neural_mechanics_reporting.oracle_independent_audit import ROOT,R,Audit,read,sha,rest_points
B=ROOT/'runs/compositional-neural-mechanics-v16/unknown-action-reserved';OUT=ROOT/'runs/compositional-neural-mechanics-v16/unknown-action-secondary-reactions'
MODELS=('analytic_fitted','shared_laws_fitted');ACTIONS=('initial','stretch','right_bend','opposed_twist')
def main():
 a=Audit();pp=R/'unknown-reaction-secondary-plan.json';plan=read(pp);ph=sha(pp);rp=R/'unknown-reaction-secondary-results.json';result=read(rp);terminal=read(OUT/'terminal.json');parentpath=R/'unknown-action-final-independent-audit.json';parent=read(parentpath);primary=read(R/'unknown-action-method-freeze.json')
 for path in (pp,rp,OUT/'terminal.json',parentpath,R/'unknown-action-method-freeze.json'):a.bind(path)
 a.check(parent.get('passed')is True and not parent.get('errors'),'passed primary audit')
 for path,h in parent['evidence_sha256'].items():a.check(sha(ROOT/path)==h,'parent evidence '+path)
 for path,h in plan['source_sha256'].items():a.check(sha(ROOT/path)==h,'frozen replay source '+path)
 a.check(plan['parent_plan_sha256']==sha(R/'unknown-action-method-freeze.json'),'parent plan')
 expectedjobs=set(itertools.product(range(6),(*MODELS,'reference')));a.check(terminal['plan_sha256']==result['plan_sha256']==ph and len(terminal['jobs'])==18 and {(r['world'],r['model'])for r in terminal['jobs']}==expectedjobs,'18terminal')
 predterminal=read(OUT/'predictions-terminal.json');a.bind(OUT/'predictions-terminal.json');a.check(predterminal['plan_sha256']==ph and len(predterminal['jobs'])==12 and {(r['world'],r['model'])for r in predterminal['jobs']}==set(itertools.product(range(6),MODELS)),'12 prediction terminal')
 values={};ledger=[]
 for i,m in sorted(expectedjobs):
  folder=OUT/f'world{i}'/m;cp=folder/'complete.json';states=('reference',)if m=='reference'else ACTIONS
  if not cp.exists():
   ledger.extend(dict(world=i,model=m,state_name=s,command=j,state='missing')for s,j in itertools.product(states,range(2)));continue
  c=read(cp);a.bind(cp);a.check(c['plan_sha256']==ph and c['audit_sha256']==sha(parentpath),'replay commitment')
  keys=[(x['world'],x['model'],x['state_name'],x['command'])for x in c['rows']];a.check(len(keys)==len(states)*2 and set(keys)=={(i,m,s,j)for s,j in itertools.product(states,range(2))},'exact response census')
  publicpath=B/f'world{i}/public.json';public=read(publicpath);a.bind(publicpath)
  for row in c['rows']:
   ledger.append(row);a.check(row['state']in ('complete','failed'),'terminal response')
   if row['state']!='complete':continue
   path=ROOT/row['artifact'];a.check(sha(path)==row['sha256'],'response hash');a.bind(path)
   with np.load(path,allow_pickle=False)as raw:
    a.check(set(raw.files)=={'wrenches','origins_world_m'},'wrench schema');w=raw['wrenches'].copy();origins=raw['origins_world_m'].copy()
   a.check(w.shape==(2,6)and origins.shape==(2,3)and np.isfinite(w).all()and np.isfinite(origins).all(),'finite wrench/origins')
   expected=rest_points(np.array([[0,.5,.5],[1,.5,.5]]),public['geometry'])+np.array(public['held_poses'][row['command']])[:,:3];a.equal(origins,expected,'independent origin mapping')
   values[i,m,row['state_name'],row['command']]=(w,origins)
   if m!='reference':
    fitpath=B/f'world{i}'/m/('initial-fit.json'if row['state_name']=='initial'else 'fits.json');fit=read(fitpath);a.bind(fitpath)
    if row['state_name']!='initial':fit=fit[row['state_name']]
    a.check(row['primary_fit_incomplete']==fit.get('incomplete')and row['primary_optimizer_success']==fit.get('optimizer_success'),'fit status matches pinned parameters')
    a.check(np.shape(fit['parameters'])==(15,)and np.isfinite(fit['parameters']).all(),'fixed finite parameter estimate')
 policies=('initial',*primary['policies']);expectedkeys=set(itertools.product(range(6),MODELS,policies,range(2)));rows={(x['world'],x['model'],x['policy'],x['command']):x for x in result['rows']};a.check(len(rows)==len(result['rows'])==144 and set(rows)==expectedkeys,'144 fixed rows');recomputed=[]
 for i,m,policy,j in sorted(expectedkeys):
  sp=B/f'world{i}'/m/'selection.json';selection=read(sp)['policies']if sp.exists()else {};state='initial'if policy=='initial'else selection.get(policy);old=rows[i,m,policy,j];a.check(old['action']==state,'same primary policy');x=values.get((i,m,state,j));y=values.get((i,'reference','reference',j));row=dict(world=i,model=m,policy=policy,command=j,state='failed')
  if x is not None and y is not None:
   a.equal(x[1],y[1],'matched torque origins');diff=x[0]-y[0];row.update(state='complete',force_coordinate_rmse_N=float(np.linalg.norm(diff[:,:3])/np.sqrt(6)),torque_coordinate_rmse_Nm=float(np.linalg.norm(diff[:,3:])/np.sqrt(6)))
   for k in ('force_coordinate_rmse_N','torque_coordinate_rmse_Nm'):a.equal(row[k],old[k],'recomputed '+k)
  a.check(row['state']==old['state'],'retained failures');recomputed.append(row)
 summaries=[]
 for m,policy in itertools.product(MODELS,policies):
  rr=[r for r in recomputed if r['model']==m and r['policy']==policy];valid=[r for r in rr if r['state']=='complete'];s=dict(model=m,policy=policy,expected_objects=6,expected_commands=12,complete_commands=len(valid))
  if len(valid)==12:
   for k in ('force_coordinate_rmse_N','torque_coordinate_rmse_Nm'):s[k]=float(np.mean([np.mean([r[k]for r in valid if r['world']==i])for i in range(6)]))
  summaries.append(s)
 for path,h in result['evidence_sha256'].items():a.check(sha(ROOT/path)==h,'result evidence '+path)
 out=dict(passed=True,checks=a.checks,created_unix=time.time(),evidence_sha256=a.evidence,expected_replays=108,terminal_replays=len(ledger),complete_replays=sum(x['state']=='complete'for x in ledger),expected_rows=144,complete_rows=sum(x['state']=='complete'for x in recomputed),rows=recomputed,summaries=summaries,scope='Independent saved-wrench metrics and origin/policy/source/fit binding. Parameter replay identity is verified through pinned worker source and primary artifact seals; no expensive solver rerun. Secondary only,0forceobservations; all failures retained.')
 (R/'unknown-reaction-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ('passed','checks','complete_rows','complete_replays')}))
if __name__=='__main__':main()
