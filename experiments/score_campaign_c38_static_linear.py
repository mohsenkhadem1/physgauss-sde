"""Prospective fixed engineering comparison, not a static image-study admission.

All predictions are sealed before nonlinear target array reads. Qualified
accuracy uses the SAME63 fixed reference points only. Positive-Keast volume
metrics are descriptive diagnostics on the saved fields, with no whole-field
spatial-error certificate. Target-fitted rigid motion is a privileged lower
bound, never presented as a public-observation predictor.
"""
from __future__ import annotations
import argparse,hashlib,json,os,resource,shutil,signal,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_campaign_c38_static_linear as linear
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.campaign_static_p2_block import shapes

ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/score_campaign_c38_static_linear.py'
META={
 'nonlinear_summary':'runs/operator-dev/c38-graded-family-warp-v01/completed-summary.json',
 'nonlinear_terminal':'runs/operator-dev/c38-graded-family-warp-v01/terminal-proof.json',
 'linear_summary':'runs/operator-dev/c38-static-linear-predictions-v01/completed-summary.json',
 'linear_terminal':'runs/operator-dev/c38-static-linear-predictions-v01/terminal-proof.json',
 'linear_protocol':'runs/operator-dev/c38-static-linear-predictions-v01/protocol-v01.json',
 'linear_seal':'runs/gaussian-campaign-v01/c38-static-linear-predictions-v01/predictions-seal.json',
}
LIMITS=dict(wall_cap_s=300,RSS_cap_bytes=12*1024**3,disk_free_reserve_bytes=30*1024**3,projected_output_bytes=10*1024**2,CPU_affinity_cores=[34,35],CPU_threads=2)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rec(p):
 p=Path(p).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Repository-local metadata required')
 return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def key(c):return (c['id'],c['young_Pa'],c['support'],c['Fz_N'])
def _checked(r):
 p=Path(r['path'])
 if p.is_absolute() or '..' in p.parts or p.as_posix()!=r['path'] or sha(ROOT/p)!=r['sha256']:raise ValueError('Changed/noncanonical comparison record')
 return ROOT/p


def metadata_plan():
 """Metadata only; no target/prediction NPZ loads or array hashes here."""
 meta={k:read(ROOT/v) for k,v in META.items()};nl=meta['nonlinear_summary'];ls=meta['linear_summary'];seal=meta['linear_seal'];lp=meta['linear_protocol']
 if not (nl.get('completed') and nl.get('all14_finite_case_q63_displacement_indicators_passed') and len(nl['rows'])==14 and ls.get('completed') and ls.get('predictions_completed')==28 and seal.get('completed') and seal.get('all28_predictions_sealed_before_nonlinear_target_access') and not seal.get('nonlinear_target_access')):raise ValueError('Complete finite14 truth qualification and28 prediction seal required')
 if meta['nonlinear_terminal'].get('all_owned_PIDs_absent') is not True or meta['nonlinear_terminal'].get('all28_capture_hashes_match') is not True or meta['linear_terminal'].get('GPU2_compute_idle') is not True or meta['linear_terminal'].get('all39_source_pins_unchanged') is not True or not all(meta['linear_terminal'].get('owned_PID_absence',{}).values()):raise ValueError('Terminal numerical stages required')
 nproof={(r['case_id'],r['cells'][0],r['capture']['path'],r['capture']['sha256']) for r in meta['nonlinear_terminal']['captures']}
 lproof={(r['path'],r['sha256']) for r in meta['linear_terminal']['captures']}
 if len(nproof)!=28 or len(lproof)!=32:raise ValueError('Exact terminal28target/32linear capture proof required')
 expected={c['id']:c for c in lp['all_cases']}
 if len(expected)!=14 or len(seal['rows'])!=14 or len(ls['rows'])!=14 or {r['case']['id'] for r in nl['rows']}!=set(expected) or {r['case']['id'] for r in seal['rows']}!=set(expected):raise ValueError('Exact14 case joins required; no filtering')
 nrows={r['case']['id']:r for r in nl['rows']};lrows={r['case']['id']:r for r in seal['rows']};basis={}
 for record in read(ROOT/META['linear_summary'])['models']:
  job=record['job'];result_path=Path(lp['output'])/job['id']/'result.json';result=read(ROOT/result_path)
  if not result.get('completed') or not result.get('passed') or result.get('job')!=job or len(result['predictions'])!=7 or (result['basis_capture']['path'],result['basis_capture']['sha256']) not in lproof:raise ValueError('Complete exact numerical basis required')
  basis[(job['support'],job['cells'][0])]=dict(result=rec(ROOT/result_path),basis=result['basis_capture'])
 if len(basis)!=4 or set(basis)!={(j['support'],j['cells'][0]) for j in lp['jobs']}:raise ValueError('Exact4 basis model joins required')
 pairs=[]
 for cell,case in expected.items():
  nr=nrows[cell];lr=lrows[cell]
  if key(nr['case'])!=key(case) or key(lr['case'])!=key(case) or nr.get('conditional_displacement_passed') is not True:raise ValueError('Physical ID/E/support/F join or q63 gate mismatch')
  for nx in (32,40):
   ns=[r for r in nr['meshes'] if r['cells'][0]==nx];ps=[r for r in lr['prediction_records'] if r['cells'][0]==nx]
   if len(ns)!=1 or len(ps)!=1 or key(ps[0]['case'])!=key(case):raise ValueError('Exact physical case/resolution join required')
   prediction=ps[0];b=basis[(case['support'],nx)]
   if (cell,nx,ns[0]['capture']['path'],ns[0]['capture']['sha256']) not in nproof or (prediction['capture']['path'],prediction['capture']['sha256']) not in lproof:raise ValueError('Joined capture absent from terminal physical/prediction proof')
   pairs.append(dict(case=case,cells=ns[0]['cells'],target=ns[0]['capture'],prediction=prediction['capture'],basis=b['basis'],basis_result=b['result'],prediction_J_passed=prediction['prediction_J_passed'],linear_q63_indicator_m=lr['conditional_linear_spatial_indicator_m'],linear_q63_indicator_passed=lr['linear_spatial_indicator_passed'],nonlinear_q63_indicator_m=nr['conditional_actual_h_q63_indicator_m'],sampled_target_peak_strain=ns[0]['sampled_peak_principal_strain']))
 return pairs


def prepare(protocol,output):
 destination=Path(protocol).resolve();out=Path(output).resolve()
 if destination.exists() or out.exists() or not destination.is_relative_to(ROOT) or not out.is_relative_to(ROOT):raise ValueError('Fresh exclusive repository-local protocol/output required')
 pairs=metadata_plan();sources=sorted(set(linear.source_closure())|{SOURCE})
 p=dict(schema='C38-static-linear-scoring-v01',status='FROZEN_BEFORE_NONLINEAR_TARGET_ARRAY_READS',source_sha256={n:sha(ROOT/n) for n in sources},metadata={k:rec(ROOT/v) for k,v in META.items()},pairs=pairs,limits=LIMITS,output=str(out.relative_to(ROOT)),qualified_error_scope='Same63 fixed material points only, coordinate RMSE + maximum/per-point Euclidean error',bulk_scope='Positive Keast24 volume-weighted saved-field diagnostics only; no spatial error certificate',rest_baseline='Deployable no-motion rest baseline',rigid_baselines={'linear_rigid':'Rigid fit to LINEAR predictions only, no target dependence','target_rigid':'PRIVILEGED rigid fit to target for descriptive best-rigid lower bound'},comparison_kind='Supplied E/support physics oracleA, engineering matrix, static equilibrium not temporal forecast or RGB-D',no_dataset_admission=True,no_parameter_fit_or_selection=True,raw_local=True)
 destination.parent.mkdir(parents=True,exist_ok=True)
 with destination.open('x') as f:json.dump(p,f,indent=2,allow_nan=False);f.write('\n')
 return sha(destination)


def validate(protocol,digest):
 if sha(protocol)!=digest:raise ValueError('Frozen scoring protocol changed')
 p=read(protocol)
 if p.get('schema')!='C38-static-linear-scoring-v01' or p.get('status')!='FROZEN_BEFORE_NONLINEAR_TARGET_ARRAY_READS' or p.get('limits')!=LIMITS or p.get('pairs')!=metadata_plan():raise ValueError('Exact28 metadata joins and frozen budget required')
 if p.get('metadata')!={k:rec(ROOT/v) for k,v in META.items()}:raise ValueError('Completed seal/physical metadata changed')
 if p.get('source_sha256')!={n:sha(ROOT/n) for n in sorted(set(linear.source_closure())|{SOURCE})}:raise ValueError('Full scoring/source closure changed')
 value=p.get('output')
 if not isinstance(value,str) or Path(value).is_absolute() or '..' in Path(value).parts or Path(value).as_posix()!=value:raise ValueError('Canonical output required')
 return p


def rigid_fits(chunks):
 """Weighted Kabsch using replayable positive-volume chunks, two stable passes.

 Returns row-vector rotations: fit(X)=(X-Xmean)@R+Qmean. We center
 before cross products; signed P2 nodal body weights are never metric masses.
 """
 mass=0.;sumX=np.zeros(3);sumP=np.zeros(3);sumT=np.zeros(3)
 for X,P,T,w in chunks():
  if X.ndim!=2 or X.shape[1]!=3 or P.shape!=X.shape or T.shape!=X.shape or w.shape!=(len(X),) or not all(np.isfinite(a).all() for a in (X,P,T,w)) or np.any(w<=0):raise ValueError('Finite compatible points and strictly positive volume weights required')
  mass+=w.sum();sumX+=np.einsum('n,ni->i',w,X);sumP+=np.einsum('n,ni->i',w,P);sumT+=np.einsum('n,ni->i',w,T)
 if not mass>0:raise ValueError('Positive physical reference volume required')
 xm,pm,tm=sumX/mass,sumP/mass,sumT/mass;cp=np.zeros((3,3));ct=np.zeros((3,3))
 for X,P,T,w in chunks():
  A=X-xm;cp+=np.einsum('n,ni,nj->ij',w,A,P-pm);ct+=np.einsum('n,ni,nj->ij',w,A,T-tm)
 def rotation(c):
  u,s,v=np.linalg.svd(c)
  if s[-1]<=np.finfo(float).eps*s[0]:raise ValueError('Nondegenerate 3D rigid-fit reference required')
  D=np.eye(3);D[-1,-1]=1 if np.linalg.det(u@v)>0 else -1
  return u@D@v
 return dict(volume=mass,reference_mean=xm,prediction_mean=pm,target_mean=tm,prediction_rotation=rotation(cp),target_rotation=rotation(ct))


def bulk_rigid_metrics(chunks):
 fits=rigid_fits(chunks);xm=fits['reference_mean'];pm=fits['prediction_mean'];tm=fits['target_mean'];rp=fits['prediction_rotation'];rt=fits['target_rotation'];sums={k:0. for k in ('linear','rest','linear_rigid','target_rigid','corotated_deformation_error','target_nonrigid','linear_nonrigid')}
 for X,P,T,w in chunks():
  A=X-xm;pr=A@rp+pm;tr=A@rt+tm;pd=(P-pm)@rp.T-A;td=(T-tm)@rt.T-A
  for name,d in [('linear',P-T),('rest',X-T),('linear_rigid',pr-T),('target_rigid',tr-T),('corotated_deformation_error',pd-td),('target_nonrigid',td),('linear_nonrigid',pd)]:sums[name]+=np.einsum('n,ni,ni->',w,d,d)
 output={k+'_coordinate_RMS_m':float(np.sqrt(max(v,0)/(3*fits['volume']))) for k,v in sums.items()}
 output.update(reference_volume_m3=float(fits['volume']),metric_weights='Positive reference-volume Keast24; no signed nodal masses',spatial_qualification='Descriptive bulk diagnostics, NOT whole-field20um error qualification')
 return fits,output


def positive_volume_chunks(rest,elements,prediction,target,*,element_chunk=4096):
 """Replayable straight-P2 field quadrature; never samples signed nodal masses."""
 X=np.asarray(rest);e=np.asarray(elements);P=np.asarray(prediction);T=np.asarray(target)
 if type(element_chunk) is not int or element_chunk<1 or X.ndim!=2 or X.shape[1]!=3 or e.ndim!=2 or e.shape[1]!=10 or e.dtype.kind not in 'iu' or not len(e) or np.any(e<0) or np.any(e>=len(X)) or P.shape!=X.shape or T.shape!=X.shape:raise ValueError('Compatible P2 geometry/fields required')
 bary,weights=positive_keast24();N=shapes(bary)
 def chunks():
  for start in range(0,len(e),element_chunk):
   ids=e[start:start+element_chunk];corners=X[ids[:,:4]];volume=np.linalg.det(np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2))/6
   if not np.isfinite(volume).all() or np.any(volume<=0):raise ValueError('Positive affine reference tetrahedra required')
   arrays=[np.einsum('qa,eai->eqi',N,q[ids],optimize=True).reshape(-1,3) for q in (X,P,T)]
   yield *arrays,(volume[:,None]*weights[None]).reshape(-1)
 return chunks


def point_metrics(error):
 d=np.asarray(error)
 if d.ndim!=2 or d.shape[1]!=3 or not len(d) or not np.isfinite(d).all():raise ValueError('Finite point displacement errors required')
 distances=np.linalg.norm(d,axis=1)
 return dict(coordinate_RMSE_m=float(np.sqrt(np.mean(d*d))),maximum_point_L2_error_m=float(distances.max()),per_point_L2_error_m=distances.tolist())


def compare_arrays(basis,prediction,target):
 for name in ('rest_m','elements','tets','faces','constraint_mask'):
  if name not in basis or name not in target or not np.array_equal(basis[name],target[name]) or basis[name].dtype!=target[name].dtype:raise ValueError('Exact baseline/target geometry and support mismatch')
 points=linear.QUERY_POINTS
 if not np.array_equal(prediction['material_probes_m'],points) or not np.array_equal(target['material_probes_m'],points) or prediction['probe_positions_m'].shape!=(63,3) or target['probe_positions_m'].shape!=(63,3):raise ValueError('Exact common63 material-point binding required')
 P=prediction['positions_m'];T=target['positions_m'];X=basis['rest_m'];fits,bulk=bulk_rigid_metrics(positive_volume_chunks(X,basis['elements'],P,T))
 pr=(points-fits['reference_mean'])@fits['prediction_rotation']+fits['prediction_mean'];tr=(points-fits['reference_mean'])@fits['target_rotation']+fits['target_mean'];true=target['probe_positions_m']
 q63={name:point_metrics(error) for name,error in [('linear',prediction['probe_positions_m']-true),('rest_no_motion',points-true),('linear_rigid_only',pr-true),('PRIVILEGED_target_fitted_rigid',tr-true)]}
 return dict(q63=q63,bulk_descriptive=bulk)


def run(p,protocol):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or set(os.sched_getaffinity(0))!=set(LIMITS['CPU_affinity_cores']) or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('CPU2 only; explicit empty CUDA visibility required')
 out=ROOT/p['output'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic();s=dict(completed=False,protocol=rec(protocol),pid=os.getpid(),rows=[],raw_local=True,no_parameters_fitted=True,no_dataset_admitted=True)
 def save():s['seconds']=time.monotonic()-start;linear.write(out/'summary.json',s)
 def cap(*_):raise TimeoutError('Frozen300s comparison wall cap')
 old=signal.signal(signal.SIGALRM,cap);signal.setitimer(signal.ITIMER_REAL,300);save()
 try:
  # Validate/hash ALL records before the first nonlinear target NPZ is opened.
  for pair in p['pairs']:
   for name in ('target','prediction','basis','basis_result'):_checked(pair[name])
  for pair in p['pairs']:
   if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>LIMITS['RSS_cap_bytes']:raise MemoryError('12GiB comparison RSS cap')
   if shutil.disk_usage(ROOT).free-LIMITS['projected_output_bytes']<LIMITS['disk_free_reserve_bytes']:raise OSError('30GiB+prospectivewrites reserve required')
   data=[]
   for name in ('basis','prediction','target'):
    with np.load(_checked(pair[name]),allow_pickle=False) as z:data.append({k:z[k].copy() for k in z.files})
   metrics=compare_arrays(*data);s['rows'].append(dict(case=pair['case'],cells=pair['cells'],capture_records={k:pair[k] for k in ('target','prediction','basis')},prediction_J_passed=pair['prediction_J_passed'],linear_q63_indicator_m=pair['linear_q63_indicator_m'],linear_q63_indicator_passed=pair['linear_q63_indicator_passed'],nonlinear_q63_indicator_m=pair['nonlinear_q63_indicator_m'],**metrics));save();del data
  s.update(completed=True,comparison_scope='Finite14 engineering configurations/q63 only; oracleA supplied E/support; no temporal/image/inference claims',bulk_spatial_error_qualified=False,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 except BaseException as error:s['failure']=dict(type=type(error).__name__,message=str(error));s['traceback']=traceback.format_exc();save();raise
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);save()
 return s


def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','run'));a.add_argument('--protocol',required=True);a.add_argument('--protocol-sha256');a.add_argument('--output');args=a.parse_args()
 if args.mode=='prepare':print(prepare(args.protocol,args.output));return
 p=validate(args.protocol,args.protocol_sha256);run(p,args.protocol)
if __name__=='__main__':main()
