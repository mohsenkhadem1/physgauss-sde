"""Read-only terminal-gated independent audit. Never launches or modifies experiments."""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';B=ROOT/'runs/compositional-neural-mechanics-v16/unknown-action-reserved'
MODELS=('analytic_fitted','shared_laws_fitted');ACTIONS=('stretch','right_bend','opposed_twist')
Z=1.6448536269514722

def metrics(mean,sd,target):
 mean,sd,target=map(np.asarray,(mean,sd,target))
 if mean.shape!=target.shape or sd.shape!=target.shape or not all(np.isfinite(x).all()for x in (mean,sd,target)) or np.any(sd<0):raise ValueError('Invalid prediction/target')
 e=mean-target
 return dict(position_coordinate_rmse_m=float(np.linalg.norm(e)/np.sqrt(e.size)),nominal90_coordinate_coverage=float(np.count_nonzero(np.abs(e)<=Z*sd)/e.size),mean_band_halfwidth_m=float(Z*np.sum(sd)/sd.size))

def main():
 evidence={};errors=[];checks=0
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def read(p):evidence[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
 def check(ok,msg):
  nonlocal checks
  checks+=1
  if not ok:errors.append(msg)
 def require(ok,msg):
  check(ok,msg)
  if not ok:raise ValueError(msg)
 planpath=R/'unknown-action-method-freeze.json';plan=read(planpath);ph=sha(planpath)
 # No private arrays are accessed until terminal, source, all commitment and score seals pass.
 terminal=read(B/'fits-terminal.json');result=read(R/'unknown-action-reserved-results.json')
 expected=set(itertools.product(range(6),MODELS))
 require(len(terminal['jobs'])==12 and {(x['world'],x['model'])for x in terminal['jobs']}==expected,'terminal12')
 for table in ('source_sha256','artifact_sha256'):
  for p,h in plan[table].items():require(sha(ROOT/p)==h,'frozen '+p)
 generation=read(B/'generation-complete.json');start=read(B/'run-start.json');clock=read(B/'stage-clock.json')
 require(all(x['plan_sha256']==ph for x in (terminal,result,generation,start,clock)),'plan chain')
 require(len(generation['rows'])==6 and [x['world']for x in generation['rows']]==list(range(6)),'generation6')
 require(terminal['generation_sha256']==start['generation_sha256']==sha(B/'generation-complete.json'),'generation seal')
 require(terminal['run_start_sha256']==sha(B/'run-start.json') and start['stage_clock_sha256']==sha(B/'stage-clock.json'),'runtime seals')
 for p,h in result['evidence_sha256'].items():require(sha(ROOT/p)==h,'score input '+p)
 policies=('initial',*plan['policies']);require(len(policies)==6 and len(set(policies))==6,'six policies')
 expectedrows=set(itertools.product(range(6),MODELS,policies,range(2)))
 require(len(result['rows'])==144 and {(x['world'],x['model'],x['policy'],x['command'])for x in result['rows']}==expectedrows,'144 fixed rows')
 records={};fits=[]
 for i,m in sorted(expected):
  folder=B/f'world{i}'/m
  for action,filename in [('initial','initial-fit.json'),*[(a,'fits.json')for a in ACTIONS]]:
   fp=folder/filename;raw=read(fp)if fp.exists()else {};fit=raw if action=='initial'else raw.get(action,{})
   fits.append(dict(world=i,model=m,action=action,present=bool(fit),optimizer_success=fit.get('optimizer_success',False),incomplete=fit.get('incomplete',True),message=fit.get('message',fit.get('error')),evaluations=len(fit.get('evaluations',[])),scipy_nfev=fit.get('scipy_nfev')))
  cp=folder/'complete.json'
  if not cp.exists():records[i,m]=None;continue
  c=read(cp);require(c['plan_sha256']==ph and c['generation_sha256']==sha(B/'generation-complete.json'),'completion identity')
  require(set(c['evidence_sha256'])=={'initial-fit.json','selection.json','fits.json','forecasts.npz'},'exact committed files')
  for name,h in c['evidence_sha256'].items():require(sha(folder/name)==h,'forecast seal '+str(folder/name));evidence[str((folder/name).relative_to(ROOT))]=h
  sel=read(folder/'selection.json');require(sel['plan_sha256']==ph and sel['initial_fit_sha256']==sha(folder/'initial-fit.json'),'selection binding')
  require(set(sel['policies'])==set(plan['policies']) and all(a in ACTIONS for a in sel['policies'].values()),'selection mapping')
  require(all(sel['policies']['fixed_'+a]==a for a in ACTIONS),'fixed controls')
  require(sel['policies']['uniform_random']==str(np.random.default_rng(plan['random_policy_seed']).choice(ACTIONS,size=6)[i]),'fixed random seed')
  require(sel['policies']['target_task_variance']==sel['ranking'][0]['action'],'recorded active ranking')
  arrays={}
  with np.load(folder/'forecasts.npz',allow_pickle=False)as a:
   for action in ('initial',*ACTIONS):
    if all(action+k in a for k in ('_mean','_sd')):
     mean,sd=a[action+'_mean'].copy(),a[action+'_sd'].copy()
     if mean.shape==sd.shape==(156,) and np.isfinite(mean).all()and np.isfinite(sd).all()and np.all(sd>=0):arrays[action]=(mean,sd)
  records[i,m]=(sel['policies'],arrays)
 require(not errors,'all pre-target guards')
 rows=[]
 for i in range(6):
  gen=generation['rows'][i];truth=None
  if gen['state']=='complete':
   for p,h in gen['public_sha256'].items():require(sha(ROOT/p)==h,'public seal '+p)
   for p,h in gen['private_sha256'].items():require(sha(ROOT/p)==h,'private seal '+p);evidence[p]=h
   with np.load(B/f'world{i}/private/held-truth.npz',allow_pickle=False)as a:truth=[a[f'held{j}'].ravel().copy()for j in range(2)]
  for m,policy,j in itertools.product(MODELS,policies,range(2)):
   rec=records[i,m];action='initial'if policy=='initial'else rec[0][policy]if rec else None
   row=dict(world=i,model=m,policy=policy,command=j,action=action,state='failed')
   if truth is not None and rec and action in rec[1]:
    mean,sd=rec[1][action];sl=slice(j*78,(j+1)*78);row.update(state='complete',**metrics(mean[sl],sd[sl],truth[j]))
   rows.append(row)
 stored={(x['world'],x['model'],x['policy'],x['command']):x for x in result['rows']}
 for row in rows:
  key=tuple(row[k]for k in ('world','model','policy','command'));old=stored[key];check(old['state']==row['state']and old['action']==row['action'],'row status '+str(key))
  if row['state']=='complete':
   for k in ('position_coordinate_rmse_m','nominal90_coordinate_coverage','mean_band_halfwidth_m'):check(np.isclose(row[k],old[k],rtol=1e-11,atol=1e-15),'metric '+str(key)+' '+k)
 summaries=[];comparisons=[]
 for m,policy in itertools.product(MODELS,policies):
  rr=[x for x in rows if x['model']==m and x['policy']==policy];valid=[x for x in rr if x['state']=='complete'];s=dict(model=m,policy=policy,expected_objects=6,expected_commands=12,complete_commands=len(valid),failed_commands=12-len(valid))
  if len(valid)==12:
   s['mean_coordinate_rmse_m']=float(np.mean([np.mean([x['position_coordinate_rmse_m']for x in rr if x['world']==i])for i in range(6)]));s['mean_coordinate_coverage']=float(np.mean([x['nominal90_coordinate_coverage']for x in rr]))
  old=[x for x in result['summaries']if x['model']==m and x['policy']==policy];check(len(old)==1,'summary census')
  if old:
   for k in ('expected_objects','expected_commands','complete_commands'):check(old[0][k]==s[k],'summary count')
   for k in ('mean_coordinate_rmse_m','mean_coordinate_coverage'):check((k in old[0])==(k in s)and(k not in s or np.isclose(s[k],old[0][k],rtol=1e-11,atol=1e-15)),'unfiltered summary '+k)
  summaries.append(s)
 for m,control in itertools.product(MODELS,[p for p in policies if p!='target_task_variance']):
  differences=[]
  for i in range(6):
   aa=[x for x in rows if x['world']==i and x['model']==m and x['policy']=='target_task_variance'];bb=[x for x in rows if x['world']==i and x['model']==m and x['policy']==control]
   if all(x['state']=='complete'for x in aa+bb):differences.append(float(np.mean([x['position_coordinate_rmse_m']for x in bb])-np.mean([x['position_coordinate_rmse_m']for x in aa])))
  comparisons.append(dict(model=m,control=control,expected_objects=6,paired_objects=len(differences),per_object_control_minus_active_m=differences,full_cohort_mean_gain_m=float(np.mean(differences))if len(differences)==6 else None,wins=sum(x>1e-15 for x in differences),losses=sum(x< -1e-15 for x in differences),ties=sum(abs(x)<=1e-15 for x in differences)))
 audit=dict(passed=not errors,checks=checks,errors=errors,created_unix=time.time(),evidence_sha256=evidence,rows=rows,summaries=summaries,comparisons=comparisons,fit_ledger=fits,fit_counts=dict(expected=48,present=sum(x['present']for x in fits),optimizer_success=sum(bool(x['optimizer_success'])for x in fits),incomplete=sum(bool(x['incomplete'])for x in fits)),scope='Known categories/geometry/load; all15continuous inferred. All144 rows retained. No success-only full-cohort means; nominal correlated-coordinate coverage. Input isolation supported by frozen-source/tests and commitment seals, not an independent OS read trace.')
 (R/'unknown-action-final-independent-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:audit[k]for k in ('passed','checks','errors','fit_counts')},indent=2))
if __name__=='__main__':main()
