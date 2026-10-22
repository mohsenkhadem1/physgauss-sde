"""Exact frozen BDF2 source, two prospective finer rates; no retry/retuning."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
from experiments.campaign_stable_transient import StableBlockTransient
from experiments.diagnose_nonlinear_physical_probes import interpolate
from experiments.run_static_p2_axial_continuation import sha,dump,record,checked_record
OLD=Path('runs/gaussian-campaign-v01/stable-transient-v01')
RATES=(3840,7680)
def prepare(path):
 old=json.loads((OLD/'summary.json').read_text());r=next(r for r in old['cases'] if r['method']=='BDF2' and r['hz']==1920)
 if not old['completed'] or not r['completed']:raise ValueError('Frozen1920reference unavailable')
 p={'schema':'StableBDF2-temporal-continuation-v0.1','status':'FROZEN_BEFORE_FINER_TRAJECTORIES','old_summary':record(OLD/'summary.json'),'old_protocol':old['protocol'],'old1920':r['capture'],'rates_hz':list(RATES),'method':'BDF2','duration_s':.1,'hard_wall_cap_s':120,'CPU_threads':2,'source_sha256':dict(checked_record(old['protocol'])['source_sha256']),'exact_frozen_core':'No numerical reformulation/looser gate/retry; preservefloat64floorfailureifany','gates':{'objective_R_N':1e-10,'original_physical_R_N':1e-9,'minimum_J':.1,'exact_clamp_m':0.,'finest_adjacent_query_Euclidean_m':20e-6,'all_adjacent_differences_nonincreasing':True},'physics':'Exactunchanged96tetstablelaw,E5000/nu.35/rho625/1Nstep/fullleftclamp/rest0/damp0/gravity0','limits':'Samefixedmesh temporalonly;coarsefailures/spatialinaccuracypreserved'}
 p['source_sha256']['experiments/run_stable_transient_refinement.py']=sha('experiments/run_stable_transient_refinement.py');path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError(path)
 dump(path,p)
def run(protocol,digest,output):
 if sha(protocol)!=digest:raise ValueError('Changed protocol')
 p=json.loads(Path(protocol).read_text());old=checked_record(p['old_summary']);checked_record(p['old_protocol'])
 for f,h in p['source_sha256'].items():
  if sha(f)!=h:raise ValueError('Changed source '+f)
 r=p['old1920']
 if sha(r['path'])!=r['sha256']:raise ValueError('Changed1920capture')
 with np.load(r['path'],allow_pickle=False) as a:points=a['material_probes_m'].copy();prior=a['query_positions_m'].copy()
 out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();d={'schema':'StableBDF2-refinement-results-v0.1','completed':False,'protocol':record(protocol),'cases':[],'comparisons':[],'fixed_mesh_temporal_qualified':False,'spatial_or_GPU_or_inference_qualified':False,'initial_failures_preserved':True,'planned':[{'hz':r,'status':'QUEUED'} for r in RATES]};history=old['temporal_qualification']['BDF2']['all_adjacent_query_errors_m'].copy();priorhz=1920
 def save():d['elapsed_s']=time.monotonic()-start;dump(out/'summary.json',d)
 signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('Frozen120sBDF2budget')));signal.alarm(120)
 try:
  for hz in RATES:
   begin=time.monotonic();row={'method':'BDF2','hz':hz,'completed':False};slot=next(r for r in d['planned'] if r['hz']==hz);slot['status']='RUNNING';save();m=StableBlockTransient();s=m.initial();prev=None;qs=[s.q.copy()];vs=[s.v.copy()];ledger=[];steps=hz//10
   try:
    for n in range(1,steps+1):
     new,check=m.step(s,1/hz,'BDF2',prev);ledger.append({'step':n,'interval_s':[(n-1)/hz,n/hz],**check})
     if check['physical_original_free_R_N']>1e-9 or check['minimum_J']<=.1 or check['constraint_defect_m']!=0 or not np.isfinite(new.v).all():raise RuntimeError('Unchanged physicalgatefailure')
     prev,s=s,new;qs.append(s.q.copy());vs.append(s.v.copy())
    qs=np.asarray(qs);vs=np.asarray(vs);query=interpolate(m.model.rest,m.model.tets,qs[::hz//240],points)[0];file=out/f'BDF2-hz{hz}.npz';np.savez_compressed(file,times_s=np.arange(steps+1)/hz,positions_m=qs,velocities_m_s=vs,query_times_s=np.arange(25)/240,query_positions_m=query,material_probes_m=points,rest_m=m.model.rest,tets=m.model.tets,faces=m.model.faces,physical_masses_kg=m.mass,constraint_mask=m.mask,applied_forces_N=m.force)
    row.update(completed=True,steps=steps,capture=record(file),maximum_objective_free_R_N=max(r['objective_free_R_N'] for r in ledger),maximum_physical_original_free_R_N=max(r['physical_original_free_R_N'] for r in ledger),minimum_J=min(r['minimum_J'] for r in ledger),maximum_constraint_defect_m=max(r['constraint_defect_m'] for r in ledger));error=float(np.linalg.norm(query-prior,axis=2).max());d['comparisons'].append({'rates_hz':[priorhz,hz],'full25frame_query_max_Euclidean_m':error,'passed20um':error<=20e-6});history.append(error);prior,priorhz=query,hz;slot['status']='COMPLETED'
   except TimeoutError as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};slot['status']='BUDGET_INTERRUPTED';raise
   except Exception as e:row['failure']={'exception':repr(e),'history':getattr(e,'history',[])};slot['status']='FAILED'
   finally:row['elapsed_s']=time.monotonic()-begin;d['cases'].append(row);dump(out/f'BDF2-hz{hz}.json',row);dump(out/f'BDF2-hz{hz}-steps.json',ledger);save()
  d['completed']=True;d['all5_adjacent_query_errors_m']=history;d['nonincreasing']=len(history)==5 and all(b<=a for a,b in zip(history,history[1:]));d['fixed_mesh_temporal_qualified']=all(r['completed'] for r in d['cases']) and d['nonincreasing'] and history[-1]<=20e-6
 except Exception as e:d['execution_failure']=repr(e)
 finally:signal.alarm(0);save()
 return d
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');a=a.parse_args()
 if a.action=='prepare':prepare(a.protocol);print(sha(a.protocol))
 else:
  d=run(a.protocol,a.protocol_sha256,a.output);print(json.dumps({k:d.get(k) for k in ['completed','elapsed_s','fixed_mesh_temporal_qualified','comparisons','execution_failure']},indent=2));raise SystemExit(0 if d['completed'] else 1)
