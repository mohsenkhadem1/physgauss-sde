"""Post-seal, reporting-only reference resolution diagnostics; no solver calls."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write
from scripts.compositional_dynamics_v17.cohort_geometry_evaluation import committed
from scripts.compositional_dynamics_v17.sequence_metrics import matched_sequence_metrics
BASE=ROOT/'runs/compositional-dynamics-v17/fresh-cohort'

def release_guard(bindings,release_path):
 # Explicit release is checked before any prediction/reference filesystem access.
 release_path=Path(release_path);r=json.loads(release_path.read_text())
 if r.get('analysis_source_sha256')!=sha(__file__)or r.get('role')!='reference_sensitivity_release'or r.get('protocol_sha256')!=bindings['protocol_sha']or r.get('method_freeze_sha256')!=bindings['freeze_sha']:raise ValueError('Explicit correctly bound root release required')
 p,_,evidence=committed(bindings)
 manifest_path=BASE/'native-recovery/recovery-manifest.json'
 if sha(manifest_path)!=r.get('recovery_manifest_sha256'):raise ValueError('Released recovery manifest changed')
 manifest=json.loads(manifest_path.read_text());expected={(w,role)for w in p['generator_settings']['world_ids']for role in ('forecast-40mm','forecast-80mm')}
 if not manifest.get('terminal')or manifest.get('expected')!=8 or len(manifest.get('rows',[]))!=8 or {(x['world_id'],x['role'])for x in manifest['rows']}!=expected:raise ValueError('All eight recovery seals required')
 for row in manifest['rows']:
  path=ROOT/row['recovery_commit'];old=ROOT/row['original_commit']
  if sha(path)!=row['recovery_commit_sha256']or sha(old)!=row['original_commit_sha256']:raise ValueError('Recovery mapping changed')
  status=json.loads(path.read_text());original=json.loads(old.read_text())
  if original.get('completed')is not False:raise ValueError('Recovery may not replace successful original')
  if status.get('world_id')!=row['world_id']or status.get('role')!=row['role']or status.get('protocol_sha256')!=bindings['protocol_sha']or status.get('method_freeze_sha256')!=bindings['freeze_sha']or type(status.get('completed'))is not bool or status.get('recovery_plan_sha256')!=manifest['plan_sha256']or status.get('original_prediction_commit_sha256')!=sha(old):raise ValueError('Recovery seal identity mismatch')
  evidence[str(path.relative_to(ROOT))]=sha(path)
  if status['completed']:
   for filename,key in [('prediction.npz','prediction_sha256'),('complete.json','native_complete_sha256')]:
    artifact=path.parent/filename
    if sha(artifact)!=status[key]:raise ValueError('Recovered prediction artifact changed')
    evidence[str(artifact.relative_to(ROOT))]=sha(artifact)
 evidence[str(manifest_path.relative_to(ROOT))]=sha(manifest_path);evidence[str(release_path.relative_to(ROOT))]=sha(release_path)
 return p,evidence

def trajectory_diagnostics(arrays,job,settings,completed):
 t=np.asarray(arrays['times_s']);surface=np.asarray(arrays['surface_frames']);rest=np.asarray(arrays['surface_rest']);ledger=np.asarray(arrays['ledger']);depth=np.asarray(arrays['saved_dense_penetration_m'])
 expected_count=round(job['duration_s']/settings['sample_interval_s'])+1
 if t.ndim!=1 or surface.shape!=(len(t),6,17,17,3)or rest.shape!=(6,17,17,3)or ledger.ndim!=2 or ledger.shape[1]!=21 or depth.shape!=(len(t),):raise ValueError('Malformed reference arrays')
 if not all(np.isfinite(x).all()for x in (t,surface,rest,ledger,depth))or len(t)<1 or len(ledger)<1 or np.any(np.diff(t)<=0)or np.any(np.diff(ledger[:,0])<=0):raise ValueError('Nonfinite/nonincreasing reference')
 clock=np.arange(expected_count)*settings['sample_interval_s'];full=bool(len(t)==expected_count and np.allclose(t,clock,rtol=0,atol=1e-12)and len(ledger)==round(job['duration_s']/job['dt_s'])+1 and np.isclose(ledger[-1,0],job['duration_s'],rtol=0,atol=1e-12))
 scale=max(abs(float(ledger[0,1])),settings['mass_kg']*settings['gravity_m_s2']*job['clearance_m'],1e-10)
 energy=float(np.max(abs(ledger[:,1]-ledger[0,1]))/scale);J=float(ledger[:,7].min());penetration=float(depth.max());g=settings['reference_diagnostics']
 checks=dict(completed=bool(completed),full_clock=full,positive_J=J>g['minimum_J'],penetration=penetration<=g['maximum_penetration_m'],energy=energy<=g['energy_drift_relative'])
 return dict(passed=all(checks.values()),checks=checks,saved_frames=len(t),expected_frames=expected_count,minimum_J=J,maximum_saved_dense_penetration_m=penetration,maximum_fullstep_contact_quadrature_penetration_m=float(ledger[:,8].max()),energy_drift_relative=energy,energy_scale_J=scale,penetration_scope='Dense65-point-per-face check at saved1ms frames; fullstep quadrature penetration separately reported. No continuous collision certificate.')

def compare_pair(a,b,kind,settings):
 if a is None or b is None:return dict(passed=False,error='Missing/incomplete declared trajectory')
 ar,ad=a;br,bd=b
 if not ar['diagnostics']['passed']or not br['diagnostics']['passed']:return dict(passed=False,error='One or both single-trajectory gates failed')
 if not np.array_equal(ad['surface_rest'],bd['surface_rest']):raise ValueError('Reference material surface changed')
 rest=ad['surface_rest'].reshape(-1,3);x=ad['surface_frames'].reshape(len(ad['times_s']),-1,3);y=bd['surface_frames'].reshape(len(bd['times_s']),-1,3)
 m=matched_sequence_metrics(rest,ad['times_s'],x,bd['times_s'],y,np.ones(len(rest)),characteristic_length_m=min(settings['dimensions_m']))
 if not m['complete']:return dict(passed=False,error='Partial comparison refused')
 prefix='spatial'if kind=='spatial'else'temporal';g=settings['reference_diagnostics'];world=m['world_coordinate_rms_m'];shape=m['shape_coordinate_rms_m']
 checks=dict(world=world<=g[prefix+'_world_coordinate_RMS_m'],shape=shape<=g[prefix+'_shape_coordinate_RMS_m'],both_trajectory_gates=True)
 return dict(passed=all(checks.values()),checks=checks,world_coordinate_RMS_m=world,shape_coordinate_RMS_m=shape,world_point_RMS_m=m['world_point_rms_m'],shape_point_RMS_m=m['shape_point_rms_m'],frame_count=m['expected_frames'],metric_scope='Uniform common6x17x17 surface sample coordinate RMS; proper rigid alignment only for shape, all saved frames includingt0. Not exact volume/mass integral.')

def main(bindings,release_path):
 out=REPORT/'fresh-cohort-reference-sensitivity.json'
 if out.exists():raise ValueError('Preserve previous reference analysis')
 p,evidence=release_guard(bindings,release_path);settings=p['generator_settings'];generation_path=REPORT/'fresh-cohort-generation.json';generation=json.loads(generation_path.read_text());runtime_path=BASE/'private/runtime-input-freeze.json'
 if not generation.get('terminal')or generation.get('protocol_sha256')!=bindings['protocol_sha']or generation.get('method_freeze_sha256')!=bindings['freeze_sha']or generation.get('runtime_manifest_sha256')!=sha(runtime_path):raise ValueError('Terminal bound generation required')
 runtime=json.loads(runtime_path.read_text());jobs=runtime['jobs'];subset=runtime['sensitivity_world_ids'];ids=[j['id']for j in jobs]
 if runtime.get('protocol_sha256')!=bindings['protocol_sha']or runtime.get('method_freeze_sha256')!=bindings['freeze_sha']or len(jobs)!=20 or len(set(ids))!=20 or set(ids)!=set(generation['expected_keys'])or len(subset)!=2 or len(set(subset))!=2:raise ValueError('Declared20-job/subset identity mismatch')
 for f in (generation_path,runtime_path):evidence[str(f.relative_to(ROOT))]=sha(f)
 rows=[];available={}
 for job in jobs:
  folder=BASE/'references'/job['id'];status_path=folder/'complete.json';row=dict(job=job,completed=False,diagnostics=dict(passed=False),error=None)
  if not status_path.exists():
   if job['id']not in generation.get('missing_keys',[]):raise ValueError('Previously present reference status disappeared')
   row['error']='Missing declared reference status';rows.append(row);continue
  status=json.loads(status_path.read_text());evidence[str(status_path.relative_to(ROOT))]=sha(status_path)
  if status.get('job')!=job or status.get('job_id')!=job['id']or status.get('protocol_sha256')!=bindings['protocol_sha']or status.get('method_freeze_sha256')!=bindings['freeze_sha']or type(status.get('completed'))is not bool:raise ValueError('Reference job identity changed')
  row.update(completed=status['completed'],error=status.get('error'))
  path=folder/'trajectory.npz'
  if path.exists():
   if status.get('trajectory_sha256')!=sha(path):raise ValueError('Reference trajectory changed')
   evidence[str(path.relative_to(ROOT))]=sha(path)
   with np.load(path)as f:data={k:f[k].copy()for k in ('times_s','surface_frames','surface_rest','ledger','saved_dense_penetration_m')}
   row['diagnostics']=trajectory_diagnostics(data,job,settings,status['completed'])
   available[job['id']]=(row,data)
  elif status['completed']:raise ValueError('Completed reference missing array')
  rows.append(row)
 comparisons=[]
 def key(w,role,n,dt):return f'{w}/{role}/n{n}-dt{dt:.8f}'
 for world in subset:
  for role in ('calibration','forecast-80mm'):
   a=key(world,role,4,25e-6);b=key(world,role,6,25e-6);comparisons.append(dict(world_id=world,role=role,kind='spatial',a=a,b=b,**compare_pair(available.get(a),available.get(b),'spatial',settings)))
  for n in (4,6):
   a=key(world,'forecast-80mm',n,25e-6);b=key(world,'forecast-80mm',n,12.5e-6);comparisons.append(dict(world_id=world,role='forecast-80mm',kind='temporal',n=n,a=a,b=b,**compare_pair(available.get(a),available.get(b),'temporal',settings)))
 world_summary=[]
 for world in settings['world_ids']:
  selected=world in subset;checks=[x for x in comparisons if x['world_id']==world and x['role']=='forecast-80mm']
  world_summary.append(dict(world_id=world,in_declared_sensitivity_subset=selected,high_drop_subset_checks_pass=all(x['passed']for x in checks)if selected else None,subset_comparison_count=len(checks),scope='80mm only: n4/n6 spatial and both-step checks; calibration spatial reported separately. No40mm spatial check, no quadrature ladder, no continuum certificate.'))
 write(out,dict(protocol_sha256=bindings['protocol_sha'],method_freeze_sha256=bindings['freeze_sha'],source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),str((ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py').relative_to(ROOT)):sha(ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py')},evidence_sha256=evidence,rows=rows,comparisons=comparisons,worlds=world_summary,expected_jobs=20,completed_jobs=sum(r['completed']for r in rows),all_twenty_single_trajectory_checks_pass=all(r['diagnostics']['passed']for r in rows),sensitivity_subset=subset,all_declared_subset_comparisons_pass=all(c['passed']for c in comparisons),scope='Post-seal reference sensitivity, all declared jobs/failures retained. Finite-resolution synthetic teacher; subset consistency does not qualify everyworld/material/load or establish continuumtruth. No prediction changes or gate relaxation.'))

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for key in ('protocol-path','freeze-path','approval-path','protocol-sha','freeze-sha'):parser.add_argument('--'+key,required=True)
 parser.add_argument('--release-path',required=True);args=vars(parser.parse_args());release=args.pop('release_path');main(args,release)
