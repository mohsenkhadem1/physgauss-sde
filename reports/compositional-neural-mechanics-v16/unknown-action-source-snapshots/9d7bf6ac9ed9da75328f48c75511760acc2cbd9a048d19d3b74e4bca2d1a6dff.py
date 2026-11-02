"""Explicit terminal-gated scoring of all six untouched inference objects."""
import json
import numpy as np
from scripts.compositional_neural_mechanics.unknown_action_campaign import guard,ROOT,R,B,PLAN,MODELS,ACTIONS,sha,write

def score():
 p=guard();terminal=json.loads((B/'fits-terminal.json').read_text());expected={(i,m)for i in range(6)for m in MODELS}
 if terminal['plan_sha256']!=sha(PLAN)or len(terminal['jobs'])!=12 or {(x['world'],x['model'])for x in terminal['jobs']}!=expected:raise ValueError('All12 jobs terminal required')
 generation=json.loads((B/'generation-complete.json').read_text());start=json.loads((B/'run-start.json').read_text())
 if generation['plan_sha256']!=sha(PLAN)or len(generation['rows'])!=6 or {r['world']for r in generation['rows']}!=set(range(6)):raise ValueError('Exact generation plan/census required')
 if terminal['run_start_sha256']!=sha(B/'run-start.json')or start['plan_sha256']!=sha(PLAN)or start['stage_clock_sha256']!=sha(B/'stage-clock.json')or start['generation_sha256']!=sha(B/'generation-complete.json')or terminal['generation_sha256']!=sha(B/'generation-complete.json'):raise ValueError('Sealed generation changed')
 evidence={str((B/'fits-terminal.json').relative_to(ROOT)):sha(B/'fits-terminal.json')};validated={}
 for i,m in sorted(expected):
  folder=B/f'world{i}'/m;complete=folder/'complete.json'
  if not complete.exists():validated[i,m]=None;continue
  d=json.loads(complete.read_text())
  if d['plan_sha256']!=sha(PLAN)or d['generation_sha256']!=sha(B/'generation-complete.json'):raise ValueError('Forecast plan/generation mismatch')
  if set(d['evidence_sha256'])!={'initial-fit.json','selection.json','fits.json','forecasts.npz'}:raise ValueError('Exact forecast evidence allowlist required')
  for name,h in d['evidence_sha256'].items():
   if sha(folder/name)!=h:raise ValueError('Committed forecast evidence changed')
   evidence[str((folder/name).relative_to(ROOT))]=h
  selection=json.loads((folder/'selection.json').read_text())
  if selection['initial_fit_sha256']!=sha(folder/'initial-fit.json'):raise ValueError('Selected initialfit changed')
  if selection['plan_sha256']!=sha(PLAN)or set(selection['policies'])!=set(p['policies'])or any(a not in ACTIONS for a in selection['policies'].values())or any(selection['policies']['fixed_'+a]!=a for a in ACTIONS):raise ValueError('Exact policies required')
  arrays={};invalid=[]
  with np.load(folder/'forecasts.npz',allow_pickle=False)as raw:
   for a in ('initial',*ACTIONS):
    if any(a+k not in raw for k in ('_mean','_sd'))or any(raw[a+k].shape!=(156,)or not np.isfinite(raw[a+k]).all()for k in ('_mean','_sd'))or(raw[a+'_sd']<0).any():invalid.append(a)
    else:arrays[a]=(raw[a+'_mean'].copy(),raw[a+'_sd'].copy())
  validated[i,m]=(selection['policies'],arrays,invalid)
 # All models/worlds are committed and validated before ANY private truth access.
 rows=[];latent=[]
 for i in range(6):
  gen=generation['rows'][i]
  truth=None
  if gen['world']!=i:raise ValueError('Generation identity mismatch')
  if gen['state']=='complete':
   for path,h in gen['private_sha256'].items():
    if sha(ROOT/path)!=h:raise ValueError('Private truth evidence changed')
    evidence[path]=h
   with np.load(B/f'world{i}/private/held-truth.npz',allow_pickle=False)as raw:truth={k:raw[k].copy()for k in raw.files}
  for m in MODELS:
   item=validated[i,m]
   for policy in ('initial',*p['policies']):
    action='initial'if policy=='initial'else item[0][policy]if item else None
    for j in (0,1):
     row=dict(world=i,model=m,policy=policy,action=action,command=j)
     if truth is None or item is None or action not in item[1]:row.update(state='failed',error='Missing reference or terminal finite forecast')
     else:
      mean,sd=item[1][action];sl=slice(j*78,(j+1)*78);target=truth[f'held{j}'].ravel()
      if target.shape!=(78,)or not np.isfinite(target).all():raise ValueError('Invalid truth array')
      error=mean[sl]-target;row.update(state='complete',position_coordinate_rmse_m=float(np.sqrt(np.mean(error**2))),nominal90_coordinate_coverage=float(np.mean(abs(error)<=1.6448536269514722*sd[sl])),mean_band_halfwidth_m=float(np.mean(1.6448536269514722*sd[sl])))
     rows.append(row)
 summaries=[]
 for m in MODELS:
  for policy in ('initial',*p['policies']):
   selected=[r for r in rows if r['model']==m and r['policy']==policy];valid=[r for r in selected if r['state']=='complete'];entry=dict(model=m,policy=policy,expected_objects=6,expected_commands=12,complete_commands=len(valid))
   if len(valid)==12:entry['mean_coordinate_rmse_m']=float(np.mean([r['position_coordinate_rmse_m']for r in valid]));entry['mean_coordinate_coverage']=float(np.mean([r['nominal90_coordinate_coverage']for r in valid]))
   summaries.append(entry)
 write(R/'unknown-action-reserved-results.json',dict(plan_sha256=sha(PLAN),scope='Six fresh continuous objects; known categorical families/geometry/load. All15continuous unknown. Target-task-aware equal-budget probes; n2inference/n4reference discrepancy, nominal localGaussian notcalibrated.',evidence_sha256=evidence,rows=rows,summaries=summaries))
if __name__=='__main__':score()
