"""Independent post-score audit. No reserved access unless explicitly executed.
Does not import the production scorer, selector, truth generator or metric code.
"""
import argparse,hashlib,itertools,json,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/physical-refinement-v15';BASE=ROOT/'runs/physical-refinement-v15'
REST=np.stack(np.meshgrid(np.linspace(0,.24,9),np.linspace(-.04,.04,3),np.linspace(-.04,.04,3),indexing='ij'),-1).reshape(-1,3)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def align(a,b):
 a=a-a.mean(0);b=b-b.mean(0);u,_,v=np.linalg.svd(a.T@b);s=np.eye(3);s[-1,-1]=np.linalg.det(u@v);return a@(u@s@v)-b

def canonical(field):return np.asarray([align(REST+x,REST) for x in field]).reshape(4,243)
def point_metrics(field,wrench,shape,target):
 tf,ts,tw=target
 shape_error=np.mean([np.mean(align(REST+a,REST+b)**2) for a,b in zip(field,tf)])
 loss=.5*np.mean(((shape-ts)/.001)**2)+.25*np.mean(((wrench[:,:3]-tw[:,:3])/.1)**2)+.25*np.mean(((wrench[:,3:]-tw[:,3:])/.01)**2)
 return dict(coordinate_rmse_mm=1000*np.sqrt(np.mean((field-tf)**2)),aligned_shape_rmse_mm=1000*np.sqrt(shape_error),reaction_force_rmse_N=np.sqrt(np.mean((wrench[:,:3]-tw[:,:3])**2)),reaction_torque_rmse_Nm=np.sqrt(np.mean((wrench[:,3:]-tw[:,3:])**2)),normalized_design_mse=loss,normalized_design_utility=-loss)

def proper(values,weights,target):
 """Independent CDF interval integral for half pairwise absolute distance."""
 v=np.asarray(values).reshape(len(weights),-1);y=np.asarray(target).reshape(-1);w=np.asarray(weights,float);w=w/w.sum();crps=[];coverage=[];width=[]
 for j in range(len(y)):
  ids=np.argsort(v[:,j]);x=v[ids,j];p=w[ids];cdf=np.cumsum(p)
  half=np.sum(cdf[:-1]*(1-cdf[:-1])*np.diff(x))
  crps.append(np.sum(w*np.abs(v[:,j]-y[j]))-half)
  lo=x[min(np.searchsorted(cdf,.05),len(x)-1)];hi=x[min(np.searchsorted(cdf,.95),len(x)-1)]
  coverage.append(lo<=y[j]<=hi);width.append(hi-lo)
 return dict(crps_mean=np.mean(crps),central90_coordinate_coverage=np.mean(coverage),mean_interval_width=np.mean(width))

def geometry_names():
 names=[]
 for layout in ('parallel_lower','crossed'):
  for j,g in enumerate(np.linspace(-.8,.8,17)):
   tag=f'g{j//4}' if j%4==0 else 'gref'+('m' if g<0 else 'p')+f'{round(abs(g)*1e6):06d}'
   for r in range(2):names.append(f'{layout}-{tag}-r{r}')
 return names

def main():
 errors=[];checks=0;evidence={}
 def check(ok,msg):
  nonlocal checks
  checks+=1
  if not ok:errors.append(msg)
 def read(path):
  path=Path(path);evidence[str(path.relative_to(ROOT))]=sha(path);return json.loads(path.read_text())
 def bind(path):evidence[str(Path(path).relative_to(ROOT))]=sha(path)
 def close(a,b,msg):check(np.allclose(a,b,rtol=1e-9,atol=1e-10),msg)
 protocol=read(REPORT/'final-protocol.json');freeze=read(REPORT/'method-freeze.json');ph=sha(REPORT/'final-protocol.json');fh=sha(REPORT/'method-freeze.json')
 if freeze['protocol_sha256']!=ph:raise ValueError('Protocol changed')
 for inventory in ('source_sha256','artifact_sha256'):
  for path,digest in freeze[inventory].items():
   if sha(ROOT/path)!=digest:raise ValueError('Frozen evidence changed: '+path)
 # Gate complete fixed cohort and score commitment BEFORE opening response arrays.
 forecast_root=BASE/'reserved/forecasts';commit=read(forecast_root/'complete.json');scores=read(REPORT/'reserved-scores.json')
 for record in (commit,scores):
  if record['protocol_sha256']!=ph or record['method_freeze_sha256']!=fh:raise ValueError('Completed output freeze mismatch')
 if commit['role']!='all_forecasts_committed_before_response_scoring':raise ValueError('Missing all-forecast gate')
 for path,digest in scores['evidence_sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Scored evidence changed: '+path)
 rows=scores['rows'];lookup={(r['world'],r['repeat'],r['track'],r['policy'],r['budget']):r for r in rows}
 if len(lookup)!=len(rows) or len(rows)!=7680:raise ValueError('Fixed score census differs')
 indexed={r['path']:r['sha256'] for r in commit['rows']}
 if len(indexed)!=2688 or len(commit['rows'])!=2688:raise ValueError('Fixed forecast census differs')
 measurement_root=BASE/'reserved/measurements';generation=read(measurement_root/'complete.json')
 if sha(measurement_root/'complete.json')!=commit['measurement_generation_sha256']:raise ValueError('Measurement generation changed')
 manifest_hashes={r['path']:r['sha256'] for world in generation['worlds'] for r in world['manifests']}
 if len(manifest_hashes)!=96:raise ValueError('Measurement census differs')
 metas={};expected=set()
 for world in protocol['reserved_worlds']:
  camera=world['initial_camera'];views=['view_'+x for x in ('left','right','oblique') if x!=camera];queries=views+['probe_left_twist','probe_right_twist','probe_left_bend','probe_right_bend']
  schedules=[('initial_only',[])]+[(f'single-{i:02d}',[q]) for i,q in enumerate(queries)]+[(f'pair-{i:02d}',list(pair)) for i,pair in enumerate(itertools.combinations(queries,2))]
  for repeat in range(4):
   for track in ('V','VF'):
    fixed=[('fixed_'+x['category'],x['sequence']) for x in protocol['fixed_sequences'] if x['initial_camera']==camera and x['track']==track]
    manifest_relative=f"{world['name']}/repeat-{repeat}/{track}/manifest.json";manifest_path=measurement_root/manifest_relative;measurement=read(manifest_path)
    if sha(manifest_path)!=manifest_hashes[manifest_relative]:raise ValueError('Packet manifest changed')
    for policy,seq in schedules+fixed+[('visual_uncertainty',None),('physical_risk',None)]:
     path=f"{world['name']}/repeat-{repeat}/{track}/{policy}";expected.add(path);folder=forecast_root/path;meta=read(folder/'complete.json')
     if sha(folder/'complete.json')!=indexed.get(path):raise ValueError('Forecast manifest changed')
     if (meta['world'],meta['repeat'],meta['track'],meta['policy'],meta['initial_camera'])!=(world['name'],repeat,track,policy,camera):raise ValueError('Identity mismatch')
     actual=meta['rollout']['sequence'];budget=2 if seq is None else len(seq)
     check(len(actual)==budget and len(set(actual))==budget and all(x in queries for x in actual),'query history '+path)
     if seq is not None:check(actual==seq,'fixed query '+path)
     receipt=meta['measurement_receipt'];check(receipt['budget']==budget and receipt['consumed']==['initial']+actual,'receipt '+path)
     check(meta['measurement_manifest_sha256']==sha(manifest_path),'receipt manifest '+path)
     check([x['name'] for x in receipt['packets']]==receipt['consumed'],'receipt packet order '+path)
     for item in receipt['packets']:
      entry=measurement['packets'][item['name']];packet=measurement_root/entry['path'];check(item['sha256']==entry['sha256']==sha(packet) and item['dimension']==entry['dimension'],'packet commitment '+path);bind(packet)
     if sha(folder/'weights.npz')!=meta['weights_sha256']:raise ValueError('Weights changed')
     with np.load(folder/'weights.npz') as z:w=z['weights']
     if w.shape!=(budget+1,6596) or not np.isfinite(w).all() or (w<0).any() or not np.allclose(w.sum(1),1):raise ValueError('Invalid posterior snapshots')
     for b,weight in enumerate(w):
      row=lookup[(world['name'],repeat,track,policy,b)]
      check(row['weights_sha256']==hashlib.sha256(weight.tobytes()).hexdigest() and row['sequence']==actual[:b],'score snapshot '+path)
     metas[(world['name'],repeat,track,policy)]=(folder,meta);bind(folder/'weights.npz')
 if expected!=set(indexed):raise ValueError('Extra/missing forecasts')
 # All gates have passed; actual response access begins here.
 def load(folder,indices):
  fields=[];wrenches=[]
  for i in indices:
   p=folder/f'pose-{i:03d}';m=read(p/'complete.json')
   for filename in ('private-state.npz','reaction-wrench.npz'):
    if sha(p/filename)!=m['artifact_sha256'][filename]:raise ValueError('Response changed')
    bind(p/filename)
   with np.load(p/'private-state.npz') as z:fields.append(z['field'])
   with np.load(p/'reaction-wrench.npz') as z:wrenches.append(np.r_[z['forces_N'][1],z['torques_Nm'][1]])
  f=np.asarray(fields);w=np.asarray(wrenches)
  if f.shape!=(4,81,3) or w.shape!=(4,6) or not np.isfinite(f).all() or not np.isfinite(w).all():raise ValueError('Invalid response')
  return f,canonical(f),w
 bank=[load(BASE/'response-predictions/hypotheses'/name/'n4',range(1,5)) for name in geometry_names()];fields=np.asarray([x[0] for x in bank]);shapes=np.asarray([x[1] for x in bank]);wrenches=np.asarray([x[2] for x in bank]);indices=np.repeat(np.arange(68),97)
 nodes,_=np.polynomial.legendre.leggauss(97);scales=np.tile(np.exp(np.log(.75)+(nodes+1)*np.log(2)/2),68);expanded_wrench=wrenches[indices]*scales[:,None,None];raw_count=0;proper_count=0
 for world in protocol['reserved_worlds']:
  target=load(BASE/'reserved/worlds'/world['name']/'n6',range(5,9))
  for repeat in range(4):
   for track in ('V','VF'):
    for policy in ('initial_only','fixed_all','physical_risk','visual_uncertainty'):
     folder,meta=metas[(world['name'],repeat,track,policy)]
     with np.load(folder/'weights.npz') as z:all_weights=z['weights']
     for budget,weight in enumerate(all_weights):
      row=lookup[(world['name'],repeat,track,policy,budget)];geom=weight.reshape(68,97).sum(1);field=np.einsum('h,h...->...',geom,fields);shape=np.einsum('h,h...->...',geom,shapes);wrench=np.einsum('h,h...->...',weight,expanded_wrench)
      actual=point_metrics(field,wrench,shape,target)
      mode=int(np.argmax(weight));highest=point_metrics(fields[mode//97],expanded_wrench[mode],shapes[mode//97],target)
      for group,values in [('metrics',actual),('highest_weight_metrics',highest)]:
       for key,value in values.items():close(row[group][key],value,'raw '+str((world['name'],repeat,track,policy,budget,group,key)))
      check(row['highest_weight_index']==mode,'highest particle');raw_count+=1
      if world['name']=='reserved-00' and repeat==0 and row['is_final']:
       for label,v,w,t in [('shape_m',shapes,geom,target[1]),('force_N',expanded_wrench[:,:,:3],weight,target[2][:,:3]),('torque_Nm',expanded_wrench[:,:,3:],weight,target[2][:,3:])]:
        for key,value in proper(v,w,t).items():close(row['proper_scores'][label+'_'+key],value,'independent CDF score '+policy+track+label+key)
       proper_count+=1
 # All scored random expectations and summaries, without using scorer helpers.
 for world in protocol['reserved_worlds']:
  for repeat in range(4):
   for track in ('V','VF'):
    group=[r for r in rows if r['world']==world['name'] and r['repeat']==repeat and r['track']==track and r['is_final']]
    pairs=[r for r in group if r['policy'].startswith('pair-')];singles=[r for r in group if r['policy'].startswith('single-')]
    check(len(pairs)==15 and len(singles)==6,'random cohort')
    for name,selected,budget in [('random_exact',pairs,2),('random_probes',[r for r in pairs if all(not x.startswith('view_') for x in r['sequence'])],2),('random_budget_one',singles,1)]:
     result=lookup[(world['name'],repeat,track,name,budget)]
     for metric_group in ('metrics','highest_weight_metrics','proper_scores'):
      for key,value in result[metric_group].items():close(value,np.mean([r[metric_group][key] for r in selected]),'random average '+name+key)
 final=[r for r in rows if r['is_final'] and not r['policy'].startswith(('pair-','single-'))]
 for s in scores['summary']:
  selected=[r for r in final if r['policy']==s['policy'] and r['track']==s['track'] and (s['initial_camera']=='all' or r['initial_camera']==s['initial_camera'])]
  check(s['count']==len(selected) and s['worlds']==len({r['world'] for r in selected}),'summary census')
  for group in ('metrics','highest_weight_metrics','proper_scores'):
   for key,value in s[group].items():close(value,np.mean([r[group][key] for r in selected]),'summary '+key)
 rng=np.random.default_rng(protocol['bootstrap']['seed'])
 for item in scores['paired_world_bootstrap']:
  worlds=item['world_names'];check(len(worlds)==12 and set(worlds)=={w['name'] for w in protocol['reserved_worlds']},'bootstrap scene unit');draws=rng.integers(12,size=(protocol['bootstrap']['replicates'],12))
  for key,result in item['metrics'].items():
   deltas=[]
   for world in worlds:
    values=[]
    for repeat in range(4):
     a=lookup[(world,repeat,item['track'],'physical_risk',2)];b=next(r for r in final if r['world']==world and r['repeat']==repeat and r['track']==item['track'] and r['policy']==item['baseline']);values.append(a['metrics'][key]-b['metrics'][key])
    deltas.append(np.mean(values))
   deltas=np.asarray(deltas);close(result['world_differences'],deltas,'paired scene differences');close(result['mean_active_minus_baseline'],deltas.mean(),'paired mean');close(result['world_bootstrap95'],np.quantile(deltas[draws].mean(1),protocol['bootstrap']['interval']),'scene bootstrap')
 qualification=read(REPORT/'reserved-mechanics-qualification.json');check(qualification==scores['mechanical_qualification'],'qualification ledger embedded');check(len(qualification['jobs'])==24 and len(qualification['comparisons'])==12,'all reference jobs/worlds retained')
 check(qualification['failures']==[x['job']['id'] for x in qualification['jobs'] if x['state']!='complete'],'failed job census')
 check(len({x['world'] for x in qualification['comparisons']})==12,'qualification identities')
 for group in qualification['comparisons']:
  check(len(group['rows'])==9,'qualification action census')
  flags=[]
  for i,row in enumerate(group['rows']):
   pairs=[]
   for n in (4,6):
    folder=BASE/'reserved/worlds'/group['world']/f'n{n}'/f'pose-{i:03d}';meta=read(folder/'complete.json');data=[]
    for filename,keys in [('private-state.npz',['field']),('surface.npz',['vertices']),('reaction-wrench.npz',['forces_N','torques_Nm'])]:
     file=folder/filename
     if sha(file)!=meta['artifact_sha256'][filename]:raise ValueError('Qualification raw file changed')
     bind(file)
     with np.load(file) as z:data.extend(z[key].copy() for key in keys)
    check(np.isfinite([meta['residual_N'],meta['minimum_J'],meta['reaction_balance_N']]).all() and meta['residual_N']<=1e-9 and meta['minimum_J']>.1 and meta['reaction_balance_N']<=1e-7,'reference solver gates')
    pairs.append(data)
   a,b=pairs;diff=[x-y for x,y in zip(a,b)];force=np.linalg.norm(diff[2],axis=1);torque=np.linalg.norm(diff[3],axis=1)
   metrics=dict(field_rms_m=np.sqrt(np.mean(diff[0]**2)),field_max_m=abs(diff[0]).max(),surface_rms_m=np.sqrt(np.mean(diff[1]**2)),surface_max_m=abs(diff[1]).max(),force_difference_N=force,torque_difference_Nm=torque)
   for key,value in metrics.items():close(row[key],value,'reference spatial metric '+key)
   passed=bool(all(np.isfinite(x).all() for pair in pairs for x in pair) and metrics['field_rms_m']<=2e-5 and metrics['field_max_m']<=1e-4 and metrics['surface_rms_m']<=2e-5 and metrics['surface_max_m']<=1e-4 and np.all(force<=1e-4+.005*np.linalg.norm(b[2],axis=1)) and np.all(torque<=1e-5+.005*np.linalg.norm(b[3],axis=1)))
   check(row['passed']==passed,'reference spatial gate');flags.append(passed)
  check(group['passed']==all(flags),'reference world gate')
 check(qualification['all_spatial_gates_pass']==all(x['passed'] for x in qualification['comparisons']),'reference overall gate')
 prior=read(REPORT/'prior-preservation.json')['sha256']
 for path,digest in prior.items():check(sha(ROOT/path)==digest,'prior preservation '+path)
 bind(Path(__file__))
 result=dict(passed=not errors,checks=checks,errors=errors,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),raw_point_snapshot_count=raw_count,raw_proper_final_cases=proper_count,prior_preserved=len(prior),forecast_count=2688,score_row_count=7680,evidence_sha256=evidence,scope='All committed receipts/cohort/seals, raw mean/highest-node metrics for initial/fixed_all/physical/visual at all snapshots, CDF-integral proper scores on fixed world00/repeat0 both tracks, all random arithmetic and scene bootstrap. No inference/optimizer rerun; exploratory limitations unchanged.')
 (REPORT/'independent-results-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 if errors:raise RuntimeError(str(errors[:10]))
 print('Independent result audit passed',checks,'checks')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
 if not args.execute:parser.error('Explicit root authorization and --execute required after forecasts/scores complete')
 main()
