"""Read-only paired timestep diagnostics for the distinct viscous pilot."""
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.viscous_development import ROOT,REPORT,RAW,PLAN,sha,write,verify
from scripts.compositional_dynamics_v17.sequence_metrics import matched_sequence_metrics

def main():
 p=json.loads(PLAN.read_text());verify(p);rp=REPORT/'viscous-development-results.json';r=json.loads(rp.read_text())
 if not r.get('terminal')or r['plan_sha256']!=sha(PLAN):raise ValueError('Terminal frozen stage required')
 rows={x['name']:x for x in r['rows']};hashes={str(rp.relative_to(ROOT)):sha(rp),str(PLAN.relative_to(ROOT)):sha(PLAN)};data={};summaries={}
 for name,row in rows.items():
  f=RAW/name/'trajectory.npz'
  if not f.exists():continue
  if sha(f)!=row['trajectory_sha256']:raise ValueError('Changed trajectory')
  d=dict(np.load(f));data[name]=d;hashes[str(f.relative_to(ROOT))]=sha(f);L=d['ledger'];scale=row['energy_scale_J'];E0=row['initial_energy_J']
  if not np.isfinite(L).all()or not np.allclose(L[:,23],L[:,1]+L[:,22],atol=1e-13,rtol=1e-13)or np.any(np.diff(L[:,22])<0)or np.any(L[:,21]<0)or L[0,22]<0:raise ValueError('Invalid dissipation ledger')
  active=L[:,4]>0;on=np.flatnonzero(active&~np.r_[False,active[:-1]]);off=np.flatnonzero(~active&np.r_[False,active[:-1]])
  summaries[name]=dict(completed=row['completed'],error=row.get('error'),energy_balance_relative=float(np.max(abs(L[:,23]-E0))/scale),dissipated_J=float(L[-1,22]),initial_energy_J=E0,final_mechanical_energy_J=float(L[-1,1]),contact_impulse_on_rigid_Ns=np.trapezoid(L[:,9:12],L[:,0],axis=0).tolist(),contact_on_times_s=L[on,0].tolist(),contact_off_times_s=L[off,0].tolist(),scope='Contact toggles are not separately counted as bounces; impulse integration is descriptive trapezoidal sampled ledger, not integrator midpoint impulse.')
 for c in p['cases']:
  if c['name']not in summaries:summaries[c['name']]=dict(completed=False,missing_or_unreadable_trajectory=True,error=rows.get(c['name'],{}).get('error','Declared job missing or has no trajectory'))
 comparisons=[]
 for prefix in ['drop-40mm','finite-striker']:
  a=f'{prefix}-dt{25e-6:.8f}';b=f'{prefix}-dt{12.5e-6:.8f}';item=dict(family=prefix,a=a,b=b,passed=False)
  if all(k in data and rows[k]['completed']for k in [a,b]):
   x,y=data[a],data[b];ref=y['surface_frames'].reshape(len(y['times_s']),-1,3);pred=x['surface_frames'].reshape(len(x['times_s']),-1,3);m=matched_sequence_metrics(y['surface_rest'].reshape(-1,3),y['times_s'],ref,x['times_s'],pred,np.ones(ref.shape[1]),characteristic_length_m=.06)
   checks=dict(world_coordinate=m['world_coordinate_rms_m']<=.0003,shape_coordinate=m['shape_coordinate_rms_m']<=.00015,both_diagnostic_gates=all(rows[k]['diagnostic_gates_pass']for k in [a,b]));item.update(world_coordinate_RMS_m=float(m['world_coordinate_rms_m']),shape_coordinate_RMS_m=float(m['shape_coordinate_rms_m']),checks={k:bool(v)for k,v in checks.items()},passed=all(checks.values()))
   if prefix=='finite-striker':item['sphere_center_coordinate_RMS_m']=float(np.sqrt(np.mean((x['sphere_centers_m']-y['sphere_centers_m'])**2)))
  else:item['reason']='Incomplete/failed declared pair retained'
  comparisons.append(item)
 write(REPORT/'viscous-development-analysis.json',dict(source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),str((ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py').relative_to(ROOT)):sha(ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py')},input_sha256=hashes,rows=summaries,comparisons=comparisons,all_temporal_diagnostics_pass=all(c['passed']for c in comparisons),reference_qualified=False,scope='Distinct synthetic viscosity+explicit midpoint, not isolated damping ablation or measured material. Fixed predeclared timestep/energy/J/depth diagnostics; no spatial qualification. All failed prefixes retained.'))
if __name__=='__main__':main()
