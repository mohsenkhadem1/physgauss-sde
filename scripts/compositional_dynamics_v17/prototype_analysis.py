"""Read-only feasibility diagnostics using the prospectively declared gates."""
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.prototype import ROOT,RAW,REPORT,PLAN,sha,write,verify
from scripts.compositional_dynamics_v17.sequence_metrics import matched_sequence_metrics

def physical_events(data,row,p):
 L=data['ledger'];dt=row['dt'];mass=data['mass_row_sums'].sum();v=L[:,15:18]/mass
 c0=np.average(data['rest_nodes']+data['u'][0],axis=0,weights=data['mass_row_sums']);g=np.array([0,0,-9.81])if row['case']=='drop'else np.zeros(3)
 acceleration=g-L[:,9:12]/mass
 com=np.vstack([c0,c0+np.cumsum(dt*v[:-1]+.5*dt*dt*acceleration[:-1],axis=0)])
 # Above is exact COM Verlet recurrence for these free-body two prototypes.
 # No attachment/dynamic-rigid terms are present in either case.
 active=L[:,4]>0;on=np.flatnonzero(active & ~np.r_[False,active[:-1]]);off=np.flatnonzero(~active & np.r_[False,active[:-1]])
 peaks=np.flatnonzero((v[:-1,2]>0)&(v[1:,2]<=0))+1
 peaks=[i for i in peaks if len(on)and i>on[0]and not active[i]and not active[i-1]]
 apex=[]
 for i in peaks:
  tau=-v[i-1,2]/g[2];apex.append(dict(time_s=float(L[i-1,0]+tau),com_z_m=float(com[i-1,2]+v[i-1,2]*tau+.5*g[2]*tau*tau)))
 impulse=-np.trapezoid(L[:,9:12],L[:,0],axis=0)
 momentum_error=L[-1,15:18]-L[0,15:18]-mass*g*L[-1,0]-impulse
 return dict(contact_on_times_s=L[on,0].tolist(),contact_off_times_s=L[off,0].tolist(),first_impact_time_s=float(L[on[0],0])if len(on)else None,free_flight_apices=apex,contact_impulse_on_soft_Ns=impulse.tolist(),momentum_balance_Ns=float(np.linalg.norm(momentum_error)),contact_note='Raw contact toggles include elastic chatter; only airborne COM velocity sign changes define recorded apices, not every toggle.',final_COM_m=com[-1].tolist())

def main():
 p=json.loads(PLAN.read_text());verify(p);rp=REPORT/'prototype-results.json';r=json.loads(rp.read_text())
 if not r.get('terminal')or r['plan_sha256']!=sha(PLAN):raise ValueError('Terminal pinned prototype required')
 rows={x['key']:x for x in r['rows']};data={};events={};hashes={str(rp.relative_to(ROOT)):sha(rp),str(PLAN.relative_to(ROOT)):sha(PLAN)}
 for k,row in rows.items():
  f=RAW/k/'trajectory.npz'
  if not f.exists():continue
  if sha(f)!=row['trajectory_sha256']:raise ValueError('Changed trajectory')
  data[k]=dict(np.load(f));events[k]=physical_events(data[k],row,p);hashes[str(f.relative_to(ROOT))]=sha(f)
 comparisons=[];L=min(p['dimensions']);gates=p['gates']
 for case in p['cases']:
  keys=[f'{case}-n{x[0]}-dt{x[1]:.8f}'for x in [(1,.0002),(1,.0001),(1,.00005),(2,.00005)]]
  for label,a,b in [('time_coarse_medium',keys[0],keys[1]),('time_medium_fine',keys[1],keys[2]),('mesh1_2',keys[2],keys[3])]:
   item=dict(case=case,label=label,a=a,b=b,passed=False)
   if a in data and b in data and rows[a]['completed']and rows[b]['completed']:
    x,y=data[a],data[b];ref=y['surface_frames'].reshape(len(y['times_s']),-1,3);pred=x['surface_frames'].reshape(len(x['times_s']),-1,3)
    m=matched_sequence_metrics(y['surface_rest'].reshape(-1,3),y['times_s'],ref,x['times_s'],pred,np.ones(ref.shape[1]),characteristic_length_m=L)
    impulse=np.linalg.norm(np.array(events[a]['contact_impulse_on_soft_Ns'])-events[b]['contact_impulse_on_soft_Ns']);tol=gates['impulse_absolute_Ns']+gates['impulse_relative']*np.linalg.norm(events[b]['contact_impulse_on_soft_Ns'])
    checks=dict(world=m['world_coordinate_rms_m']<=gates['world_RMS_over_L']*L,shape=m['shape_coordinate_rms_m']<=gates['shape_RMS_over_L']*L,impulse=bool(impulse<=tol),finite_J=rows[a]['minimum_J']>.1 and rows[b]['minimum_J']>.1,penetration=max(rows[a]['maximum_dense_penetration_m'],rows[b]['maximum_dense_penetration_m'])<=gates['penetration_over_L']*L)
    item.update(world_coordinate_RMS_m=m['world_coordinate_rms_m'],shape_coordinate_RMS_m=m['shape_coordinate_rms_m'],reference_nonrigid_point_RMS_m=m['reference_nonrigid_rms_m'],impulse_difference_Ns=float(impulse),impulse_tolerance_Ns=float(tol))
    if case=='drop':
     impact=abs(events[a]['first_impact_time_s']-events[b]['first_impact_time_s']);checks['impact_time']=impact<=gates['event_time_over_sqrt_Lg']*np.sqrt(L/9.81);item['impact_time_difference_s']=impact
     aa,bb=events[a]['free_flight_apices'],events[b]['free_flight_apices'];checks['first_apex_present']=bool(aa and bb)
     if aa and bb:
      height=abs(aa[0]['com_z_m']-bb[0]['com_z_m']);timing=abs(aa[0]['time_s']-bb[0]['time_s']);checks['first_apex_height']=height<=gates['apex_over_L']*L;checks['first_apex_time']=timing<=gates['event_time_over_sqrt_Lg']*np.sqrt(L/9.81);item.update(first_apex_height_difference_m=height,first_apex_time_difference_s=timing)
    item.update(checks={k:bool(v)for k,v in checks.items()},passed=all(checks.values()))
   comparisons.append(item)
 write(REPORT/'prototype-analysis.json',dict(source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),str((ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py').relative_to(ROOT)):sha(ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py')},input_sha256=hashes,comparisons=comparisons,events=events,energy_drift={k:row.get('maximum_relative_energy_drift')for k,row in rows.items()},scope='Development feasibility; uniform fixed material-surface grid coordinate norms, not physical mass norm. COM/events use actual consistent mass row sums and exact free-body Verlet recurrence. No final dynamics qualification: no independent surface/volume quadrature or higher-mesh checks yet.'))
if __name__=='__main__':main()
