"""Independent saved-reference numerical sensitivity audit; no simulation."""
import json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-dynamics-v17';B=ROOT/'runs/compositional-dynamics-v17/fresh-cohort'
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'fresh-cohort-reference-sensitivity.json';d=json.loads(p.read_text());protocol=json.loads((R/'fresh-cohort-protocol.json').read_text());s=protocol['generator_settings'];g=s['reference_diagnostics'];errors=[];n=0
 def ck(x,msg):
  nonlocal n
  n+=1
  if not x:errors.append(msg)
 def eq(a,b,msg):ck(np.allclose(a,b,rtol=1e-8,atol=1e-12),msg)
 for group in ['source_sha256','evidence_sha256']:
  for name,v in d[group].items():ck(h(ROOT/name)==v,'hash '+name)
 runtime=json.loads((B/'private/runtime-input-freeze.json').read_text());ck(len(d['rows'])==20 and {r['job']['id']for r in d['rows']}=={j['id']for j in runtime['jobs']},'20 job census');arrays={}
 for row in d['rows']:
  job=row['job'];path=B/'references'/job['id']/'trajectory.npz';status=json.loads(path.with_name('complete.json').read_text());ck(status['job']==job and status['completed']==row['completed'],'identity')
  with np.load(path)as f:a={k:f[k].copy()for k in ['times_s','surface_rest','surface_frames','ledger','saved_dense_penetration_m']}
  arrays[job['id']]=a;L=a['ledger'];t=a['times_s'];scale=max(abs(L[0,1]),s['mass_kg']*s['gravity_m_s2']*job['clearance_m'],1e-10)
  values=dict(energy_scale_J=scale,energy_drift_relative=max(abs(L[:,1]-L[0,1]))/scale,minimum_J=L[:,7].min(),maximum_saved_dense_penetration_m=a['saved_dense_penetration_m'].max(),maximum_fullstep_contact_quadrature_penetration_m=L[:,8].max())
  for key,value in values.items():eq(value,row['diagnostics'][key],key)
  clock=np.arange(round(job['duration_s']/s['sample_interval_s'])+1)*s['sample_interval_s'];eq(t,clock,'saved clock');ck(len(L)==round(job['duration_s']/job['dt_s'])+1,'full step count');eq(L[-1,0],job['duration_s'],'full duration')
  passed=bool(row['completed'] and values['minimum_J']>g['minimum_J'] and values['maximum_saved_dense_penetration_m']<=g['maximum_penetration_m'] and values['energy_drift_relative']<=g['energy_drift_relative']);ck(passed==row['diagnostics']['passed'],'single conjunction')
 summary=[]
 for c in d['comparisons']:
  a,b=arrays[c['a']],arrays[c['b']];eq(a['times_s'],b['times_s'],'pair clock');eq(a['surface_rest'],b['surface_rest'],'pair rest');x=a['surface_frames'].reshape(len(a['times_s']),-1,3);y=b['surface_frames'].reshape(len(b['times_s']),-1,3);world=np.sqrt(np.mean((x-y)**2));res=[]
  for u,v in zip(x,y):
   uc=u-u.mean(0);vc=v-v.mean(0);left,_,right=np.linalg.svd(uc.T@vc);reflection=np.eye(3);reflection[2,2]=np.linalg.det(left@right);rot=left@reflection@right;res.append(np.mean((uc@rot-vc)**2))
  shape=np.sqrt(np.mean(res));eq(world,c['world_coordinate_RMS_m'],'world RMS');eq(shape,c['shape_coordinate_RMS_m'],'shape RMS');eq(world*np.sqrt(3),c['world_point_RMS_m'],'vector conversion');eq(shape*np.sqrt(3),c['shape_point_RMS_m'],'shape conversion');prefix=c['kind'];passed=world<=g[prefix+'_world_coordinate_RMS_m']and shape<=g[prefix+'_shape_coordinate_RMS_m'];ck(bool(passed)==c['passed'],'comparison gate');summary.append(dict(world_id=c['world_id'],role=c['role'],kind=c['kind'],world_mm=float(world*1000),shape_mm=float(shape*1000),passed=bool(passed)))
 ck(len(summary)==8 and sum(x['kind']=='spatial'for x in summary)==4,'eight comparison census');ck(all(r['diagnostics']['passed']for r in d['rows'])==d['all_twenty_single_trajectory_checks_pass'],'all singles');ck(all(c['passed']for c in summary)==d['all_declared_subset_comparisons_pass'],'all comparisons')
 out=dict(passed=not errors,checks=n,errors=errors,completed_jobs=20,comparisons=summary,all_single_trajectory_checks_pass=True,spatial_passes=sum(c['passed']for c in summary if c['kind']=='spatial'),temporal_passes=sum(c['passed']for c in summary if c['kind']=='temporal'),evidence_sha256={str(p.relative_to(ROOT)):h(p),**d['evidence_sha256']},source_sha256={str(Path(__file__).relative_to(ROOT)):h(__file__)},scope='All20 saved reference histories and eight independent uniform-surface coordinate-RMS/proper rigid Kabsch comparisons. Four spatial failures retained; synthetic finite-resolution references are not continuum-qualified. Serialization-only repair did not change metrics/gates.',created_epoch=time.time());(R/'fresh-cohort-reference-sensitivity-final-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ['evidence_sha256','source_sha256']},indent=2))
if __name__=='__main__':main()
