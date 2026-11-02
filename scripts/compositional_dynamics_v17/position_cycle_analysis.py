"""Read-only position/work and timestep diagnostics; no spatial qualification."""
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.position_cycle import ROOT,REPORT,RAW,PLAN,sha,write,verify,controls
from scripts.compositional_dynamics_v17.sequence_metrics import matched_sequence_metrics

def main():
 p=json.loads(PLAN.read_text());verify(p);rp=REPORT/'position-cycle-results.json';r=json.loads(rp.read_text())
 if not r.get('terminal')or r['plan_sha256']!=sha(PLAN):raise ValueError('Terminal exact position-cycle stage required')
 rows={x['name']:x for x in r['rows']};data={};summary={};hashes={str(rp.relative_to(ROOT)):sha(rp),str(PLAN.relative_to(ROOT)):sha(PLAN)}
 for c in p['cases']:
  name=c['name'];row=rows.get(name);f=RAW/name/'trajectory.npz'
  if row is None or not f.exists():summary[name]=dict(completed=False,missing_or_unreadable=True,error=(row or {}).get('error'));continue
  if sha(f)!=row['trajectory_sha256']:raise ValueError('Trajectory changed')
  d=dict(np.load(f));data[name]=d;hashes[str(f.relative_to(ROOT))]=sha(f);L=d['ledger'];poses=np.array([controls(t)for t in d['times_s']])
  if not np.isfinite(L).all()or not np.allclose(poses,d['tool_poses_m_rad'],rtol=0,atol=1e-13)or not np.allclose(L[:,8],L[:,1]-L[:,7]-row['initial_energy_J'],rtol=0,atol=1e-12):raise ValueError('Invalid command/work ledger')
  faces=d['surface_frames'];centers=faces.mean(axis=(2,3));separation=np.linalg.norm(centers[:,1]-centers[:,0],axis=1);actual=1-separation/.12;command=(poses[:,0,0]-poses[:,1,0])/.12
  summary[name]=dict(completed=row['completed'],error=row.get('error'),maximum_commanded_compression_fraction=float(command.max()),maximum_measured_mean_endface_compression_fraction=float(actual.max()),minimum_J=float(L[:,6].min()),maximum_relative_energy_work_error=float(np.max(abs(L[:,8]))/row['energy_scale_J']),maximum_actuator_work_J=float(L[:,7].max()),final_actuator_work_J=float(L[-1,7]),final_mechanical_energy_J=float(L[-1,1]),scope='Actual mean endface separation; command not equated with material strain. Gravity ledger separate from anchor-target load.')
 a,b=[c['name']for c in p['cases']];comparison=dict(passed=False)
 if all(k in data and rows[k]['completed']for k in [a,b]):
  x,y=data[a],data[b];reference=y['surface_frames'].reshape(len(y['times_s']),-1,3);pred=x['surface_frames'].reshape(len(x['times_s']),-1,3);m=matched_sequence_metrics(y['surface_rest'].reshape(-1,3),y['times_s'],reference,x['times_s'],pred,np.ones(reference.shape[1]),characteristic_length_m=.06);checks=dict(world=m['world_coordinate_rms_m']<=p['gates']['world_coordinate_RMS_m'],shape=m['shape_coordinate_rms_m']<=p['gates']['shape_coordinate_RMS_m'],both_diagnostics=all(rows[k]['diagnostic_gates_pass']for k in [a,b]));comparison.update(checks={k:bool(v)for k,v in checks.items()},passed=all(checks.values()),world_coordinate_RMS_m=float(m['world_coordinate_rms_m']),shape_coordinate_RMS_m=float(m['shape_coordinate_rms_m']))
 else:comparison['reason']='Incomplete declared timestep pair retained'
 write(REPORT/'position-cycle-analysis.json',dict(source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),str((ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py').relative_to(ROOT)):sha(ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py')},input_sha256=hashes,rows=summary,temporal_comparison=comparison,reference_qualified=False,scope='Exploratory smooth position-controlled compression, hold and unloading with attachments retained; not sudden release, drop validation, or spatially qualified truth.'))
if __name__=='__main__':main()
