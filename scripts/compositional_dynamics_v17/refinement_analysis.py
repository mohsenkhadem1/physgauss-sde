"""Read-only refinement conjunction; failures are not selected away."""
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.refinement import ROOT,REPORT,RAW,PLAN,OLDPLAN,verify,task_id,sha,write
from scripts.compositional_dynamics_v17.prototype_analysis import physical_events
from scripts.compositional_dynamics_v17.sequence_metrics import matched_sequence_metrics

def comparison(a,b,rows,data,events,p):
 item=dict(a=a,b=b,passed=False)
 if a not in data or b not in data or not rows[a]['completed']or not rows[b]['completed']:return dict(**item,error='Missing or failed declared trajectory')
 x,y=data[a],data[b];L=min(p['dimensions']);g=p['gates'];ref=y['surface_frames'].reshape(len(y['times_s']),-1,3);pred=x['surface_frames'].reshape(len(x['times_s']),-1,3)
 m=matched_sequence_metrics(y['surface_rest'].reshape(-1,3),y['times_s'],ref,x['times_s'],pred,np.ones(ref.shape[1]),characteristic_length_m=L)
 impulse=np.linalg.norm(np.array(events[a]['contact_impulse_on_soft_Ns'])-events[b]['contact_impulse_on_soft_Ns']);tol=g['impulse_absolute_Ns']+g['impulse_relative']*np.linalg.norm(events[b]['contact_impulse_on_soft_Ns'])
 checks=dict(world=m['world_coordinate_rms_m']<=g['world_RMS_over_L']*L,shape=m['shape_coordinate_rms_m']<=g['shape_RMS_over_L']*L,impulse=impulse<=tol,finite_J=rows[a]['minimum_J']>g['minimum_J']and rows[b]['minimum_J']>g['minimum_J'],penetration=max(rows[a]['maximum_dense_penetration_m'],rows[b]['maximum_dense_penetration_m'])<=g['penetration_over_L']*L,energy=max(rows[a]['maximum_relative_energy_drift'],rows[b]['maximum_relative_energy_drift'])<=g['energy_relative'])
 item.update(world_coordinate_RMS_m=m['world_coordinate_rms_m'],shape_coordinate_RMS_m=m['shape_coordinate_rms_m'],reference_nonrigid_point_RMS_m=m['reference_nonrigid_rms_m'],impulse_difference_Ns=float(impulse),impulse_tolerance_Ns=float(tol))
 if rows[a]['case']=='drop':
  ia,ib=events[a]['first_impact_time_s'],events[b]['first_impact_time_s'];checks['impact_present']=ia is not None and ib is not None
  if checks['impact_present']:
   error=abs(ia-ib);checks['impact_time']=error<=g['event_time_over_sqrt_Lg']*np.sqrt(L/9.81);item['impact_time_difference_s']=error
  aa,bb=events[a]['free_flight_apices'],events[b]['free_flight_apices'];checks['first_apex_present']=bool(aa and bb)
  if aa and bb:
   height=abs(aa[0]['com_z_m']-bb[0]['com_z_m']);timing=abs(aa[0]['time_s']-bb[0]['time_s']);checks['first_apex_height']=height<=g['apex_over_L']*L;checks['first_apex_time']=timing<=g['event_time_over_sqrt_Lg']*np.sqrt(L/9.81);item.update(first_apex_height_difference_m=height,first_apex_time_difference_s=timing)
 item.update(checks={k:bool(v)for k,v in checks.items()},passed=all(checks.values()));return item

def main():
 plan=json.loads(PLAN.read_text());verify(plan);rp=REPORT/'refinement-results.json';r=json.loads(rp.read_text());p=json.loads(OLDPLAN.read_text())
 if not r.get('terminal')or r['plan_sha256']!=sha(PLAN)or p['gates']!=plan['gates']:raise ValueError('Terminal unchanged-gate refinement required')
 rows={x['task_id']:x for x in r['rows']};data={};events={};hashes={str(rp.relative_to(ROOT)):sha(rp),str(PLAN.relative_to(ROOT)):sha(PLAN)}
 for key,row in rows.items():
  f=ROOT/row['path']/'trajectory.npz'
  if not f.exists():continue
  if sha(f)!=row['trajectory_sha256']:raise ValueError('Changed trajectory')
  data[key]=dict(np.load(f));events[key]=physical_events(data[key],row,p);hashes[str(f.relative_to(ROOT))]=sha(f)
 def key(case,n,dt=25e-6,bq=3,sq=8):return task_id(dict(case=case,n=n,dt=dt,volume_order=bq,surface_order=sq))
 comparisons=[];qualified={}
 for case in p['cases']:
  required=[]
  for n in [2,3,4]:
   pairs=[('temporal',key(case,n,50e-6),key(case,n)),('volume',key(case,n),key(case,n,bq=4))]
   if case=='drop':pairs.append(('contact_quadrature',key(case,n),key(case,n,sq=16)))
   for label,a,b in pairs:
    item=dict(case=case,type=label,n=n,**comparison(a,b,rows,data,events,p));comparisons.append(item)
    if n in [3,4]:required.append(item)
  for na,nb in [(2,3),(3,4)]:
   item=dict(case=case,type='spatial',pair=[na,nb],**comparison(key(case,na),key(case,nb),rows,data,events,p));comparisons.append(item)
   if(na,nb)==(3,4):required.append(item)
   # Declared diagnostic uses already saved q4 trajectories, not a substitute gate.
   comparisons.append(dict(case=case,type='diagnostic_volume4_spatial',pair=[na,nb],not_used_for_acceptance=True,**comparison(key(case,na,bq=4),key(case,nb,bq=4),rows,data,events,p)))
  qualified[case]=dict(passed=all(x['passed']for x in required),candidate_pair=[3,4],reference_candidate=4,required=[dict(type=x['type'],n=x.get('n'),pair=x.get('pair'),passed=x['passed'])for x in required])
 sources=[Path(__file__),ROOT/'scripts/compositional_dynamics_v17/prototype_analysis.py',ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py']
 write(REPORT/'refinement-analysis.json',dict(source_sha256={str(s.relative_to(ROOT)):sha(s)for s in sources},input_sha256=hashes,comparisons=comparisons,events=events,cases=qualified,all_candidate_gates_pass=all(x['passed']for x in qualified.values()),scope='Current stableNH development only. Latest3/4 plus BOTHmembers temporal/volume/contact/energy gates. All coarse failures preserved; q4 spatial diagnostic cannot replace failed acceptance checks. Uniform material-surface coordinate norm, not exact volume mass norm.'))
if __name__=='__main__':main()
