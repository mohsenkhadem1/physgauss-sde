"""Independent conserved/dissipated ledger and saved-field audit; no solves."""
from pathlib import Path
import json,hashlib,time
import numpy as np
R=Path(__file__).resolve().parents[2];D=R/'reports/compositional-dynamics-v17'
def main():
 ev={};errors=[];checks=[]
 def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 def rd(p):ev[str(p.relative_to(R))]=h(p);return json.loads(p.read_text())
 def ck(x,s):checks.append(s);errors.extend([]if x else[s])
 def eq(x,y,s):ck(np.allclose(x,y,rtol=1e-8,atol=2e-13),s)
 pp=D/'viscous-recovery-plan.json';p=rd(pp);new=rd(D/'viscous-recovery-results.json');old=rd(D/'viscous-development-results.json');merged=rd(D/'viscous-recovery-combined-inputs.json');a=rd(D/'viscous-recovery-analysis.json');manifest=rd(R/'runs/compositional-dynamics-v17/viscous-recovery/runtime-input-freeze.json')
 ck(h(pp)=='5c9f831ad811d231fa253a6c9867388704f539a0499b9ff86937d02e550699fb','exact plan')
 for obj in (p,a):
  for group in ('source_sha256','input_sha256'):
   for n,v in obj.get(group,{}).items():ck(h(R/n)==v,'pin '+n)
 ck(new['terminal']and len(new['rows'])==2 and all(r['completed']for r in new['rows']),'two completed retries');ck(new['runtime_freeze_sha256']==h(R/'runs/compositional-dynamics-v17/viscous-recovery/runtime-input-freeze.json'),'runtime seal')
 for n,v in manifest['original_artifacts'].items():ck(h(R/n)==v,'old preserved '+n)
 selected=[r['name']for r in old['rows']if not r['completed']];ck(set(selected)==set(manifest['selected_names'])==set(new['selected_names']),'only incomplete retry selection')
 oldrows={r['name']:r for r in old['rows']};newrows={r['name']:r for r in new['rows']};data={};valid={}
 ck(len(merged['rows'])==4 and len({r['name']for r in merged['rows']})==4,'fixed four census')
 for row in merged['rows']:
  name=row['name'];source=newrows[name]if name in selected else oldrows[name];ck(row==source,'exact selected source '+name);path=R/row['path']/'trajectory.npz';ck(h(path)==row['trajectory_sha256'],'array '+name);ev[str(path.relative_to(R))]=h(path)
  with np.load(path)as z:x={k:z[k].copy()for k in z.files}
  data[name]=x;L=x['ledger'];ck(all(np.isfinite(v).all()for v in x.values()),'finite '+name);eq(L[:,23],L[:,1]+L[:,22],'E+D identity '+name);ck(np.all(np.diff(L[:,22])>=0)and np.all(L[:,21]>=0)and L[0,22]>=0,'nonnegative dissipation '+name)
  balance=np.max(abs(L[:,1]+L[:,22]-row['initial_energy_J']))/row['energy_scale_J'];eq(balance,row['maximum_relative_energy_balance_error'],'row balance');eq(balance,a['rows'][name]['energy_balance_relative'],'summary balance');eq(L[-1,22],row['dissipated_energy_J'],'D');eq(L[:,7].min(),row['minimum_J'],'J');eq(x['saved_dense_penetration_m'].max(),row['maximum_dense_penetration_m'],'depth')
  valid[name]=bool(row['completed']and L[:,7].min()>.1 and balance<=.01 and x['saved_dense_penetration_m'].max()<=.0012);ck(valid[name]==row['diagnostic_gates_pass'],'single gates')
  eq(np.trapezoid(L[:,9:12],L[:,0],axis=0),a['rows'][name]['contact_impulse_on_rigid_Ns'],'descriptive impulse')
  if name in selected:
   prior=R/oldrows[name]['path']/'trajectory.npz';ck(h(prior)==oldrows[name]['trajectory_sha256'],'original failed array');ev[str(prior.relative_to(R))]=h(prior)
   with np.load(prior)as z:
    for k in ['times_s','u','velocities','surface_frames','sphere_centers_m','sphere_velocities_m_s','sphere_omega_rad_s','tool_wrenches_N_Nm','ledger']:
     if k in z:ck(np.array_equal(z[k],x[k][:len(z[k])]),'byteexact restart '+name+' '+k)
 def aligned(x,y):
  xx=x-x.mean(0);yy=y-y.mean(0);u,_,vt=np.linalg.svd(xx.T@yy);d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);return xx@u@d@vt+y.mean(0)
 for c in a['comparisons']:
  x,y=data[c['a']],data[c['b']];eq(x['times_s'],y['times_s'],'paired clock');xx=x['surface_frames'].reshape(len(x['times_s']),-1,3);yy=y['surface_frames'].reshape(len(y['times_s']),-1,3);world=float(np.sqrt(np.mean((xx-yy)**2)));shape=float(np.sqrt(np.mean([np.mean((aligned(v,w)-w)**2)for v,w in zip(xx,yy)])));eq(world,c['world_coordinate_RMS_m'],'world');eq(shape,c['shape_coordinate_RMS_m'],'shape');g=dict(world_coordinate=world<=.0003,shape_coordinate=shape<=.00015,both_diagnostic_gates=valid[c['a']]and valid[c['b']]);ck(g==c['checks']and all(g.values())==c['passed'],'paired conjunction')
  if 'sphere_center_coordinate_RMS_m'in c:eq(np.sqrt(np.mean((x['sphere_centers_m']-y['sphere_centers_m'])**2)),c['sphere_center_coordinate_RMS_m'],'sphere')
 ck(a['all_temporal_diagnostics_pass']and not a['reference_qualified'],'temporal only scope')
 out=dict(passed=not errors,checks=len(checks),errors=errors,created_unix=time.time(),evidence_sha256=ev,comparisons=a['comparisons'],scope='All4combined trajectories,2original failed prefixes,source/runtime/retry selection audited. Independent E+D,J,depth,proper-rigid shape/world RMS and descriptive impulse recomputation. Both temporal pairs pass; no spatial qualification or measured-material claim. Original failures preserved; same initial-state restarts, no stitching.',source_sha256={str(Path(__file__).relative_to(R)):h(__file__)})
 (D/'viscous-recovery-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ('passed','checks','errors')}),flush=True)
if __name__=='__main__':main()
