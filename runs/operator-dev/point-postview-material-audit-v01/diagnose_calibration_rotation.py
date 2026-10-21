"""CALIB0/15 saved-field uniform-node rigidfit; no model calls or new physics."""
from pathlib import Path
import hashlib,json
import numpy as np
from experiments import campaign_point_postview_material_audit as audit
from experiments.run_point_prior_analysis import validate_execution
ROOT=Path.cwd();OUT=ROOT/'runs/operator-dev/point-postview-material-audit-v01'
REPORT=dict(path='runs/operator-dev/point-postview-material-audit-v01/calibration-maxima-diagnosis.json',sha256='4af991eff6263c32c5103c78597895f22b4444f06071e2e4166fe71f8f579506')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
declaration=dict(scope='OnlysavedCALIBRATION0/15 rigidgeometrydiagnostic, no reserved/models/queries/physics',input=REPORT,source_sha256=sha(__file__))
dump(OUT/'rotation-diagnosis-declaration.json',declaration)
first=audit._read(audit._checked(REPORT));assert first['role']=='CALIBRATION' and first['only_object_indices']==[0,15]
_,_,plan=validate_execution('runs/operator-dev/point-prior-analysis-v01/execution-protocol.json','fabad32cb79a94bde93890506f8862f0ea00d9535f001bfad026236aaa08242c',check_environment=False)
public=audit._read(audit._checked(plan['public_hardware']));rest=np.asarray(public['rest_points_m']);rows=[]

def fit(reference,target):
 x=reference-reference.mean(0);y=target-target.mean(0);u,s,vh=np.linalg.svd(x.T@y);d=np.eye(3);d[-1,-1]=np.linalg.det(vh.T@u.T);R=vh.T@d@u.T;t=target.mean(0)-R@reference.mean(0)
 residual=target-(reference@R.T+t);angle=float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)));axial=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
 phi=axial/2 if angle<1e-8 else axial*angle/(2*np.sin(angle))
 return R,t,phi,angle,float(np.sqrt(np.mean(np.sum(residual**2,axis=-1))))

for obj in first['objects']:
 assert obj['object_index'] in [0,15] and obj['role']=='CALIBRATION';worst=obj['joint_maximum'];assert worst['program']=='A'
 node=worst['node'];frame=worst['frame'];pivot=rest[obj['true_fixture_nodes'][0]]
 with np.load(audit._checked(obj['source_capture_pairs']['A']['trajectory']),allow_pickle=False) as data:q=data['positions_m'][frame]
 with np.load(audit._checked(obj['source_states']),allow_pickle=False) as data:mu=data[worst['state']+'__A__mean_displacement_m'][frame]
 R,t,phi,angle,rmse=fit(rest,q);finite=(R-np.eye(3))@(rest[node]-pivot);linear=np.cross(phi,rest[node]-pivot);second=finite-linear
 pivot_q=(rest-pivot)@R.T+pivot;point_pivot_rmse=float(np.sqrt(np.mean(np.sum((q-pivot_q)**2,axis=-1))))
 _,_,_,pred_angle,pred_rmse=fit(rest,rest+mu)
 rows.append(dict(object_index=obj['object_index'],role='CALIBRATION',state=worst['state'],frame=frame,node=node,true_fixture_node=obj['true_fixture_nodes'][0],
  uniform_node_rigid_fit_angle_deg=float(np.degrees(angle)),uniform_node_rigid_fit_q_RMSE_m=rmse,
  rigid_rotation_about_true_fixture_q_RMSE_m=point_pivot_rmse,actual_displacement_RMS_m=float(np.sqrt(np.mean(np.sum((q-rest)**2,axis=-1)))),
  fitted_rotation_vector_rad=phi.tolist(),fitted_translation_m=t.tolist(),
  actual_worstnode_displacement_m=(q[node]-rest[node]).tolist(),predicted_worstnode_displacement_m=mu[node].tolist(),
  fitted_finite_rotation_about_fixture_displacement_m=finite.tolist(),fitted_first_order_rotation_displacement_m=linear.tolist(),
  fitted_finite_minus_first_order_displacement_m=second.tolist(),x_finite_rotation_remainder_m=float(second[0]),
  x_prediction_error_m=worst['error_m'],x_error_minus_fitted_rotation_remainder_m=float(worst['error_m']-second[0]),
  predicted_mean_best_rigid_fit_angle_deg=float(np.degrees(pred_angle)),predicted_mean_rigid_fit_RMSE_m=pred_rmse,
  interpretation='Uniformnode leastsquares rigidfit, notphysicalmassfit. Rigidrotationfits truefield closely; its omitted higher-order axialterm is compared with error. This supports a geometriclinearization explanation but doesnot decompose allmodel/inference errors.'))
result=dict(completed=True,input_maximum_report=REPORT,declaration_source_sha256=sha(__file__),only_CALIBRATION_indices=[0,15],objects=rows,
 reserved_values_opened=False,no_models_or_physics_queried=True,pinned_sources_changed=False)
dump(OUT/'calibration-rotation-diagnosis.json',result);print(json.dumps(result))
