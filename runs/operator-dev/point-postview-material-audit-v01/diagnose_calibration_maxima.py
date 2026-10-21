"""Root-authorized CALIBRATION0/15 saved-field diagnostic only; no models queried."""
from pathlib import Path
import hashlib,json
import numpy as np
from experiments import campaign_point_postview_material_audit as audit
from experiments.run_point_prior_analysis import validate_execution,eligible_objects
from experiments.summarize_point_prior_analysis import validate_freeze
ROOT=Path.cwd()
OUT=ROOT/'runs/operator-dev/point-postview-material-audit-v01'
EXEC='runs/operator-dev/point-prior-analysis-v01/execution-protocol.json';EXECSHA='fabad32cb79a94bde93890506f8862f0ea00d9535f001bfad026236aaa08242c'
CALIB=dict(path='runs/gaussian-campaign-v01/point-prior-analysis-v01/calibration-frozen.json',sha256='80c61a592588cd61c571caf3804bcaefe5d0cea2627c656ae590f08655dac3df')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):return dict(path=Path(p).resolve().relative_to(ROOT).as_posix(),sha256=sha(p))
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
declaration=dict(scope='OnlyCALIBRATIONobjects0and15 savedfields/labels; no reserved, queries, physics or sourcechanges',indices=[0,15],execution=dict(path=EXEC,sha256=EXECSHA),calibration_freeze=CALIB,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),'experiments/campaign_point_postview_material_audit.py':sha(ROOT/'experiments/campaign_point_postview_material_audit.py')})
dump(OUT/'maxima-diagnosis-declaration.json',declaration)
freeze=validate_freeze(CALIB,EXECSHA);execution,method,plan=validate_execution(EXEC,EXECSHA,check_environment=False)
objects=eligible_objects(plan,'CALIBRATION',execution_sha256=EXECSHA,method_sha256=audit.METHOD_SHA)
records,status=audit.preflight_saved_objects(objects,execution,EXECSHA,'CALIBRATION')
for r in records:
 if r['result'] not in freeze['calibration_results']:raise ValueError('CALIBresult not infrozen16')
public=audit._read(audit._checked(plan['public_hardware']));rest=np.asarray(public['rest_points_m']);tets=np.asarray(public['tetrahedra']);faces=np.asarray(public['surface_faces']);surface=set(faces.flatten().tolist())
restDm=np.stack([rest[tets[:,j]]-rest[tets[:,0]] for j in [1,2,3]],axis=-1);restJ=np.linalg.det(restDm)
audit_path=ROOT/'runs/gaussian-campaign-v01/point-postview-material-audit-v01/CALIBRATION/object-diagnostics-LOCAL.json'
audit_all=audit._read(audit_path);reports=[]
for index in [0,15]:
 r=records[index];result=audit._read(audit._checked(r['result']));obj=r['object'];spec=audit._read(audit._checked(obj['realized_hypothesis']));truth={};mechanics={};positions={};rots={};centers={}
 for program in ['A','B']:
  trajectory=result['capture_pairs'][program]['trajectory']
  with np.load(audit._checked(trajectory),allow_pickle=False) as data:
   q=data['positions_m'];times=data['times_s'];R=data['handle_rotations'];C=data['handle_centers_m']
  if not np.array_equal(times,np.arange(61)/20):raise ValueError('Exact61frames required')
  positions[program]=q;truth[program]=q-rest;rots[program]=R;centers[program]=C
  dm=np.stack([q[:,tets[:,j]]-q[:,tets[:,0]] for j in [1,2,3]],axis=-1);J=np.linalg.det(dm)/restJ
  angles=np.arccos(np.clip((np.trace(R,axis1=-2,axis2=-1)-1)/2,-1,1))
  mechanics[program]=dict(minimum_saved_tet_J=float(J.min()),maximum_saved_tet_J=float(J.max()),max_saved_handle_rotation_deg=float(np.degrees(angles).max()),maximum_saved_node_displacement_m=float(np.linalg.norm(truth[program],axis=-1).max()))
 perstate=[]
 with np.load(audit._checked(r['states']),allow_pickle=False) as predicted:
  for name in audit.STATE_NAMES:
   maxima=[]
   for program in ['A','B']:
    mean=predicted[name+'__'+program+'__mean_displacement_m'];std=predicted[name+'__'+program+'__latent_std_m'];start=21 if name=='no_additional_view' and program=='A' else 31 if program=='A' else 0
    ratios=np.abs(truth[program][start:]-mean[start:])/np.maximum(std[start:],1e-9)
    local_frame,node,component=map(int,np.unravel_index(int(np.argmax(ratios)),ratios.shape));frame=start+local_frame
    handle_ids=[i for i,h in enumerate(public['handles']) if node in h['ordered_patch_nodes']]
    row=dict(state=name,program=program,frame=frame,time_s=float(times[frame]),node=node,component=['x','y','z'][component],component_index=component,
     score=float(ratios[local_frame,node,component]),truth_displacement_m=float(truth[program][frame,node,component]),mean_displacement_m=float(mean[frame,node,component]),
     error_m=float(truth[program][frame,node,component]-mean[frame,node,component]),latent_std_m=float(std[frame,node,component]),denominator_m=float(max(std[frame,node,component],1e-9)),numerical_floor_active=bool(std[frame,node,component]<1e-9),
     rest_position_m=rest[node].tolist(),truth_world_position_m=positions[program][frame,node].tolist(),mean_world_position_m=(rest[node]+mean[frame,node]).tolist(),
     truth_displacement_xyz_m=truth[program][frame,node].tolist(),mean_displacement_xyz_m=mean[frame,node].tolist(),latent_std_xyz_m=std[frame,node].tolist(),
     is_true_fixture=node in spec['fixture_nodes'],on_surface=node in surface,known_handle_membership=handle_ids)
    if handle_ids:
     h=handle_ids[0];handle=public['handles'][h];local=rest[node]-np.asarray(handle['origin_world_m']);rotation=rots[program][frame,h];angle=float(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1)))
     axial=np.array([rotation[2,1]-rotation[1,2],rotation[0,2]-rotation[2,0],rotation[1,0]-rotation[0,1]])
     phi=axial/2 if angle<1e-8 else axial*angle/(2*np.sin(angle));finite=(rotation-np.eye(3))@local;first=np.cross(phi,local);remainder=finite-first
     rigid_expected=centers[program][frame,h]+rotation@local
     row['true_rigid_geometry']=dict(handle=h,angle_deg=float(np.degrees(angle)),local_reference_offset_m=local.tolist(),finite_rotation_displacement_m=finite.tolist(),first_order_rotation_displacement_m=first.tolist(),finite_minus_first_order_m=remainder.tolist(),component_remainder_m=float(remainder[component]),rigid_attachment_q_residual_m=float(np.linalg.norm(positions[program][frame,node]-rigid_expected)),interpretation='Actual finite-rotation vs first-order geometry only; doesnot alone identify predictionerror cause')
    maxima.append(row)
   worst=max(maxima,key=lambda v:v['score'])
   if not np.isclose(worst['score'],result['states'][name]['individual_band_score'],rtol=1e-12,atol=1e-12):raise ValueError('Savedscore not reproduced')
   d=next(o for o in audit_all if o['object_index']==index)['diagnostic']['states'][name]
   worst['support_and_material_same2048']={k:d[k] for k in ['empirical_ESS','maximum_particle_weight','unique_geometry_ids_in_common_particles','positive_weight_geometry_ids','support_true_node_probability','support_true_node_rank','support_mode_node','support_mode_probability','support_entropy_nats','material']}
   perstate.append(worst)
 worst=max(perstate,key=lambda v:v['score'])
 if not np.isclose(worst['score'],result['joint_band_score'],rtol=1e-12,atol=1e-12):raise ValueError('Jointscore not reproduced')
 reports.append(dict(object_index=index,role='CALIBRATION',source_result=r['result'],source_states=r['states'],source_private_hypothesis=obj['realized_hypothesis'],source_capture_pairs=result['capture_pairs'],
  true_fixture_nodes=spec['fixture_nodes'],true_material=spec['material'],selected_active_view=result['selected_active_view'],selected_random_view=result['selected_random_view'],
  recorded_joint_score=result['joint_band_score'],reproduced_joint_score=worst['score'],joint_maximum=worst,state_maxima=perstate,saved_truth_mechanics=mechanics))
output=dict(completed=True,role='CALIBRATION',only_object_indices=[0,15],declaration=rec(OUT/'maxima-diagnosis-declaration.json'),calibration_freeze=CALIB,postview_material_audit=rec(audit_path),original_stage_terminal=status,
 observations_or_models_requeried=False,pinned_sources_changed=False,reserved_values_opened=False,objects=reports,
 caveat='Standardized maximum is dimensionless and can be large from small denominator; numericalfloor is not physicaldiscrepancy. Saved finitefield/rigidgeometrychecks are posthoc interpretation, not nativeenergy/mass or continuumconvergence proof.')
dump(OUT/'calibration-maxima-diagnosis.json',output)
print(json.dumps({'output':rec(OUT/'calibration-maxima-diagnosis.json'),'maxima':[{k:o['joint_maximum'][k] for k in ['state','program','frame','time_s','node','component','score','error_m','latent_std_m','numerical_floor_active','known_handle_membership']} for o in reports]}))
