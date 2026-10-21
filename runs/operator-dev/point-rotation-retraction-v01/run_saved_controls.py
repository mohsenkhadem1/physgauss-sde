"""Exactlyeight savedCALIB0/15 kinematicretractions, no newforwardqueries."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
from experiments.campaign_point_rotation_retraction import retract_point_rotation
from experiments import campaign_point_postview_material_audit as audit
from experiments.run_point_prior_analysis import validate_execution
ROOT=Path.cwd();OUT=ROOT/'runs/gaussian-campaign-v01/point-calibration-rotation-retraction-v01'
CONTROL=dict(path='runs/gaussian-campaign-v01/point-calibration-true-forward-controls-v01/summary.json',sha256='6879f8745a24934ffb67126ec147b45dc0d363e1fa14968f8817da4e91b3b0b6')
MAXIMA=dict(path='runs/operator-dev/point-postview-material-audit-v01/calibration-maxima-diagnosis.json',sha256='4af991eff6263c32c5103c78597895f22b4444f06071e2e4166fe71f8f579506')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):return dict(path=Path(p).resolve().relative_to(ROOT).as_posix(),sha256=sha(p))
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
if any(os.environ.get(v)!='2' for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 required')
OUT.mkdir(parents=True,exist_ok=False)
protocol=dict(scope='Exactlysaved8CALIB0/15×A/B×full/modal pointanchoredgeometryretractions; noqueries/physics/tuning',
 inputs=CONTROL,source_joint_maxima=MAXIMA,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),'experiments/campaign_point_rotation_retraction.py':sha(ROOT/'experiments/campaign_point_rotation_retraction.py')},
 fit='Fixeduniform455nodes, perframe anchoredLS omega; residual unchanged; exp replacesI+skewomega',
 diagnostics='Allcoordinate/future31..60RMSE, originalmaxcoordinateerror, posteriororiginalargmax(A), all9nodepatchlengthdefects, fittedomega, savedendpointJ',
 reserved_values_opened=False,new_dynamics=False,CPU_threads=2)
dump(OUT/'protocol.json',protocol);dump(OUT/'execution.json',dict(pid=os.getpid(),terminal=False,completed=False));started=time.monotonic()
try:
 _,_,plan=validate_execution('runs/operator-dev/point-prior-analysis-v01/execution-protocol.json','fabad32cb79a94bde93890506f8862f0ea00d9535f001bfad026236aaa08242c',check_environment=False)
 public=audit._read(audit._checked(plan['public_hardware']));rest=np.asarray(public['rest_points_m']);assert len(rest)==455;tets=np.asarray(public['tetrahedra'])
 controls=audit._read(audit._checked(CONTROL));assert controls['completed'] and controls['only_object_indices']==[0,15] and controls['queries_completed']==8 and not controls['reserved_data_opened']
 maxima=audit._read(audit._checked(MAXIMA));assert maxima['only_object_indices']==[0,15]
 det_rest=np.linalg.det(np.stack([rest[tets[:,j]]-rest[tets[:,0]] for j in [1,2,3]],axis=-1));rows=[]
 def patch_defect(q):
  values=[]
  for h in public['handles']:
   p=np.array(h['ordered_patch_nodes']);ref=np.linalg.norm(rest[p,None]-rest[None,p],axis=-1);now=np.linalg.norm(q[:,p,None]-q[:,None,p],axis=-1);values.append(float(np.abs(now-ref).max()))
  return max(values)
 def metrics(q,truth):
  error=q-truth;dm=np.stack([q[:,tets[:,j]]-q[:,tets[:,0]] for j in [1,2,3]],axis=-1);J=np.linalg.det(dm)/det_rest
  return dict(all_coordinate_RMSE_m=float(np.sqrt(np.mean(error**2))),future31_60_coordinate_RMSE_m=float(np.sqrt(np.mean(error[31:]**2))),max_abs_error_m=float(np.abs(error).max()),max_rigid_patch_pair_distance_defect_m=patch_defect(q),minimum_saved_endpoint_J=float(J.min()),maximum_saved_endpoint_J=float(J.max()))
 for row in controls['rows']:
  assert row['role']=='CALIBRATION' and row['object_index'] in [0,15] and row['program'] in ['A','B']
  with np.load(audit._checked(row['finite_trajectory']),allow_pickle=False) as data:truth=data['positions_m'];times=data['times_s']
  for name in ['full_linear','packed_cached_modal']:
   original=row['controls'][name]['predictions']
   with np.load(audit._checked(original),allow_pickle=False) as data:q=data['positions_m']
   result=retract_point_rotation(rest,q-rest,int(row['true_support_node']));corrected=rest+result['displacement_m']
   if not np.array_equal(corrected[:,row['true_support_node']],np.broadcast_to(rest[row['true_support_node']],(61,3))):raise ValueError('Pointpin lost')
   loc=tuple(map(int,np.unravel_index(np.abs(q-truth).argmax(),q.shape)))
   posteriorloc=next(o['joint_maximum'] for o in maxima['objects'] if o['object_index']==row['object_index'])
   extra=None
   if row['program']=='A':
    f,n,c=posteriorloc['frame'],posteriorloc['node'],posteriorloc['component_index']
    extra=dict(frame=f,node=n,component=posteriorloc['component'],truth_displacement_m=float(truth[f,n,c]-rest[n,c]),before_displacement_m=float(q[f,n,c]-rest[n,c]),after_displacement_m=float(corrected[f,n,c]-rest[n,c]),before_abs_error_m=float(abs(q[f,n,c]-truth[f,n,c])),after_abs_error_m=float(abs(corrected[f,n,c]-truth[f,n,c])))
   dest=OUT/f"object{row['object_index']:02d}__{row['program']}__{name}.npz"
   np.savez_compressed(dest,positions_m=corrected,times_s=times,fitted_infinitesimal_rotation_rad=result['fitted_infinitesimal_rotation_rad'],rotation_fit_linear_residual_RMS_m=result['linear_residual_coordinate_RMS_m'])
   rows.append(dict(object_index=row['object_index'],role='CALIBRATION',program=row['program'],model=name,input_prediction=original,output_prediction=rec(dest),
    before=metrics(q,truth),after=metrics(corrected,truth),old_maximum_coordinate=dict(frame=loc[0],node=loc[1],component=loc[2],before_abs_error_m=float(abs(q[loc]-truth[loc])),after_abs_error_m=float(abs(corrected[loc]-truth[loc]))),
    original_posterior_joint_argmax=extra,max_fitted_infinitesimal_angle_deg=float(np.degrees(np.linalg.norm(result['fitted_infinitesimal_rotation_rad'],axis=-1)).max()),
    fit_normal_condition_number=result['rotation_fit_normal_condition_number'],exact_pointpin_preserved=True))
   print(json.dumps(rows[-1]),flush=True)
 report=dict(completed=True,role='CALIBRATION',only_object_indices=[0,15],saved_trajectories_processed=len(rows),protocol=rec(OUT/'protocol.json'),elapsed_s=time.monotonic()-started,rows=rows,
 new_forwardqueries=0,new_physics_or_inference_runs=0,reserved_values_opened=False,source_sha256=protocol['source_sha256'],
 claim='Geometryonlydiagnostic on twoseenobjects. Doesnot correct dynamics, inertial/coriolis/constitutive terms, enforce allrigidpatchconstraints, or validate newposteriors/calibratedbands.',
 saved_20Hz_velocities_or_force_ledger_recomputed=False)
 dump(OUT/'summary.json',report);dump(OUT/'execution.json',dict(pid=os.getpid(),terminal=True,completed=True,processed=8));print(json.dumps(rec(OUT/'summary.json')),flush=True)
except BaseException as error:
 dump(OUT/'failure.json',dict(completed=False,error=type(error).__name__,message=str(error)));dump(OUT/'execution.json',dict(pid=os.getpid(),terminal=True,completed=False));raise
