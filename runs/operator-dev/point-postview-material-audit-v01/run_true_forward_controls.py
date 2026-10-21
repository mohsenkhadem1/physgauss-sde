"""Prospective supplied-true CALIB0/15 full-linear/modal A/B controls, CPU2."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch
from experiments import campaign_point_postview_material_audit as audit
from experiments.run_point_prior_analysis import validate_execution,eligible_objects
from experiments.summarize_point_prior_analysis import validate_freeze
from experiments.campaign_hypothesis_inputs import known_hardware_from_public,validate_realized_hypothesis
from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
from experiments.campaign_support_basis_cache import load_support_basis_cache
from experiments.campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS
ROOT=Path.cwd();OUT=ROOT/'runs/gaussian-campaign-v01/point-calibration-true-forward-controls-v01'
EXEC='runs/operator-dev/point-prior-analysis-v01/execution-protocol.json';ESHA='fabad32cb79a94bde93890506f8862f0ea00d9535f001bfad026236aaa08242c'
CALIB=dict(path='runs/gaussian-campaign-v01/point-prior-analysis-v01/calibration-frozen.json',sha256='80c61a592588cd61c571caf3804bcaefe5d0cea2627c656ae590f08655dac3df')
MAXIMA=dict(path='runs/operator-dev/point-postview-material-audit-v01/calibration-maxima-diagnosis.json',sha256='4af991eff6263c32c5103c78597895f22b4444f06071e2e4166fe71f8f579506')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):return dict(path=Path(p).resolve().relative_to(ROOT).as_posix(),sha256=sha(p))
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
if any(os.environ.get(v)!='2' for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 required')
torch.set_num_threads(2)
execution,method,plan=validate_execution(EXEC,ESHA,check_environment=False);freeze=validate_freeze(CALIB,ESHA)
objects=eligible_objects(plan,'CALIBRATION',execution_sha256=ESHA,method_sha256=audit.METHOD_SHA);records,status=audit.preflight_saved_objects(objects,execution,ESHA,'CALIBRATION')
for r in records:
 if r['result'] not in freeze['calibration_results']:raise ValueError('FrozenCALIBresults required')
# Freeze every source and the exact8queries BEFORE reading suppliedtruth/modelconstruction.
OUT.mkdir(parents=True,exist_ok=False)
protocol=dict(scope='ExactlyCALIB0/15 × A/B × full_linear/cached_modal once fromrest; suppliedtrue parameters; no fitting/inference/finite simulation',execution=dict(path=EXEC,sha256=ESHA),calibration=CALIB,maxima=MAXIMA,
 selected_indices=[0,15],models=['full_linear','packed_cached_modal'],programs=list(PROGRAM_IDS),cache=execution['cache'],source_sha256={**execution['source_sha256'],str(Path(__file__).relative_to(ROOT)):sha(__file__)},CPU_threads=2,GPU_used=False,queries=8,
 output=OUT.relative_to(ROOT).as_posix(),kinematic_retraction_not_performed=True)
dump(OUT/'protocol.json',protocol);started=time.monotonic();dump(OUT/'execution.json',dict(pid=os.getpid(),terminal=False,completed=False,queries_completed=0))
rows=[]
try:
 maxima=audit._read(audit._checked(MAXIMA));assert maxima['only_object_indices']==[0,15]
 packed,public,ids,_=load_support_basis_cache(execution['cache']['path'],execution['cache']['sha256'])
 if public!=audit._read(audit._checked(plan['public_hardware'])):raise ValueError('SamepublicH0 required')
 known=known_hardware_from_public(public);rest=known.points
 for index in [0,15]:
  r=records[index];spec=audit._read(audit._checked(r['object']['realized_hypothesis']));validate_realized_hypothesis(known,spec)
  if spec['boundary_kind']!='point_pin' or len(spec['fixture_nodes'])!=1 or spec['fixture_nodes'][0] not in ids:raise ValueError('Exactknown pointcategory required')
  full=LinearHypothesisDynamics.from_hypothesis(known,spec)
  theta=torch.tensor([[spec['material'][k] for k in audit.MATERIAL_KEYS]],dtype=torch.float64);geometry=torch.tensor(spec['fixture_nodes'],dtype=torch.int64)
  original=audit._read(audit._checked(r['result']));worst=next(x['joint_maximum'] for x in maxima['objects'] if x['object_index']==index)
  for label,program in zip(['A','B'],PROGRAM_IDS):
   clock=make_exciting_commands(program);record=original['capture_pairs'][label]['trajectory']
   with np.load(audit._checked(record),allow_pickle=False) as data:
    truth=data['positions_m'];times=data['times_s'];forces=data['force_step_commands_N']
   if not np.array_equal(times,clock.output_times_s) or not np.array_equal(forces,clock.handle_forces_world_N):raise ValueError('Originalknowncommands/outputclock changed')
   predictions={};query_seconds={}
   with torch.no_grad():
    before=time.monotonic();predictions['full_linear']=full.predict(clock,integration_hz=120);query_seconds['full_linear']=time.monotonic()-before
    before=time.monotonic();raw=packed.predict(clock,geometry,theta,device='cpu');query_seconds['packed_cached_modal']=time.monotonic()-before
    predictions['packed_cached_modal']={k:v[0] if k in ['positions_m','handle_centers_m','handle_rotation_coordinates_rad'] else v for k,v in raw.items()}
   body={};qs={}
   for name,prediction in predictions.items():
    arrays={k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in prediction.items() if isinstance(v,(np.ndarray,torch.Tensor))};q=arrays['positions_m'];qs[name]=q
    destination=OUT/f'object{index:02d}__{label}__{name}.npz';np.savez_compressed(destination,**arrays)
    error=q-truth;entry=dict(predictions=rec(destination),query_seconds=query_seconds[name],all_coordinate_RMSE_m=float(np.sqrt(np.mean(error**2))),future31_60_coordinate_RMSE_m=float(np.sqrt(np.mean(error[31:]**2))),max_abs_error_m=float(np.abs(error).max()))
    if label=='A':
     frame,node,comp=worst['frame'],worst['node'],worst['component_index']
     entry['original_joint_argmax']=dict(frame=frame,node=node,component=worst['component'],truth_displacement_m=float(truth[frame,node,comp]-rest[node,comp]),predicted_displacement_m=float(q[frame,node,comp]-rest[node,comp]),error_pred_minus_truth_m=float(q[frame,node,comp]-truth[frame,node,comp]),posterior_error_truth_minus_pred_m=worst['error_m'])
    defects=[]
    for h in known.handles:
     p=h.ordered_patch_nodes;ref=np.linalg.norm(rest[p,None]-rest[None,p],axis=-1);now=np.linalg.norm(q[:,p,None]-q[:,None,p],axis=-1);defects.append(float(np.abs(now-ref).max()))
    entry['max_rigid_patch_pair_distance_defect_m']=max(defects);body[name]=entry
   rows.append(dict(object_index=index,role='CALIBRATION',program=label,known_program_id=program,true_support_node=spec['fixture_nodes'][0],supplied_true_material=spec['material'],finite_trajectory=record,
    controls=body,modal_vs_full_coordinate_RMSE_m=float(np.sqrt(np.mean((qs['packed_cached_modal']-qs['full_linear'])**2))),modal_vs_full_max_abs_m=float(np.abs(qs['packed_cached_modal']-qs['full_linear']).max())))
   dump(OUT/'execution.json',dict(pid=os.getpid(),terminal=False,completed=False,queries_completed=2*len(rows),elapsed_s=time.monotonic()-started))
   print(json.dumps(rows[-1]),flush=True)
 report=dict(completed=True,role='CALIBRATION',only_object_indices=[0,15],queries_completed=8,protocol=rec(OUT/'protocol.json'),calibration_freeze=CALIB,rows=rows,elapsed_seconds=time.monotonic()-started,source_sha256=protocol['source_sha256'],
  reserved_data_opened=False,no_finite_or_posterior_or_reference_reruns=True,cache_rebuilt=False,kinematic_retraction_performed=False,
  claim='Suppliedtruth control separates forwardapproximation from inference; doesnot repair dynamics or qualify unseenmodels.')
 dump(OUT/'summary.json',report);dump(OUT/'execution.json',dict(pid=os.getpid(),terminal=True,completed=True,queries_completed=8,elapsed_s=time.monotonic()-started));print(json.dumps(rec(OUT/'summary.json')),flush=True)
except BaseException as error:
 dump(OUT/'failure.json',dict(completed=False,error=type(error).__name__,message=str(error)));dump(OUT/'execution.json',dict(pid=os.getpid(),terminal=True,completed=False,queries_completed=2*len(rows)));raise
