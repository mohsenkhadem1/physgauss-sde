"""Separate temporal adapter: exact frozen suite worker body, new strict protocol."""
from __future__ import annotations
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
from isaac_block_gravity_suite import ROOT,geometry_and_fixed,mass_and_reference,ballistic_reference,check_flags,native_geometry,write,sha

from isaac_block_gravity_refinement_v02 import validate_worker_protocol as validate_protocol


def worker(path,digest,case_id):
 d=validate_protocol(path,digest);case=next(c for c in d['cases'] if c['id']==case_id);out=ROOT/d['output']/case_id
 if out.exists():raise FileExistsError('Preserve original case')
 out.mkdir();write(out/'case.json',case);X,T,F,fixed=geometry_and_fixed(case);masses,w,det0=mass_and_reference(X,T)
 np.savez_compressed(out/'reference.npz',rest_m=X,tets=T,faces=F,fixed_node_ids=fixed,nominal_lumped_mass_kg=masses)
 for name in d['source_sha256']:
  dest=out/'provenance'/Path(name).name;dest.parent.mkdir(exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
 from isaacsim import SimulationApp
 app=SimulationApp(dict(headless=True,active_gpu=0,physics_gpu=0,multi_gpu=False,disable_viewport_updates=True,fast_shutdown=True,width=64,height=64))
 state=dict(schema='isaac-block-gravity-case-v0.1',case=case,completed=False,pid=os.getpid(),protocol=dict(path=path,sha256=digest),floor_height_m=0. if case['kind']=='drop' else None,nominal_solid_mass_kg=float(masses.sum()),native_nodal_mass_observed=False,native_contact_accuracy_qualified=False)
 start=time.monotonic();snapq=[];snapv=[];times=[];actualtimes=[];before=[];after=[];centers=[];velocities=[];Jextrema=[];clamp=[];step=-1
 code=1
 try:
  import carb
  from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Vt,Gf
  from isaacsim.core.api import World
  from isaacsim.core.experimental.prims import DeformablePrim
  from isaacsim.core.simulation_manager import SimulationManager
  carb.settings.get_settings().set_bool('/persistent/physics/enableDeformableBeta',True)
  hz=d['physics_hz'];world=World(stage_units_in_meters=1.,physics_dt=1/hz,rendering_dt=1/hz,backend='warp',device='cuda:0')
  stage=world.stage;UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);scene=UsdPhysics.Scene.Get(stage,'/physicsScene')
  scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
  api=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());api.CreateEnableGPUDynamicsAttr(True);api.CreateBroadphaseTypeAttr('GPU');api.CreateSolverTypeAttr('PGS');api.CreateEnableExternalForcesEveryIterationAttr(False)
  mesh=UsdGeom.TetMesh.Define(stage,'/World/Block');mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(X.astype(np.float32)));mesh.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(T));mesh.GetSurfaceFaceVertexIndicesAttr().Set(Vt.Vec3iArray.FromNumpy(F));mesh.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.zeros_like(X,dtype=np.float32)));prim=mesh.GetPrim()
  for name in ['OmniPhysicsDeformableBodyAPI','OmniPhysicsVolumeDeformableSimAPI','PhysxBaseDeformableBodyAPI']:
   if not prim.ApplyAPI(name):raise RuntimeError('Missing native deformable schema '+name)
  def setting(name,value):
   if not prim.GetAttribute(name).Set(value):raise RuntimeError('Native setting refused '+name)
  setting('omniphysics:restShapePoints',mesh.GetPointsAttr().Get());setting('omniphysics:restTetVtxIndices',mesh.GetTetVertexIndicesAttr().Get());setting('omniphysics:kinematicEnabled',False)
  for k,v in dict(disableGravity=False,solverPositionIterationCount=32,sleepThreshold=0.,settlingThreshold=0.,settlingDamping=0.,linearDamping=0.,selfCollision=False).items():setting('physxDeformableBody:'+k,v)
  UsdPhysics.CollisionAPI.Apply(prim);collision=PhysxSchema.PhysxCollisionAPI.Apply(prim);collision.CreateContactOffsetAttr(.001);collision.CreateRestOffsetAttr(0.)
  material=UsdShade.Material.Define(stage,'/World/Material')
  for name in ['OmniPhysicsBaseMaterialAPI','OmniPhysicsDeformableMaterialAPI','PhysxDeformableMaterialAPI']:
   if not material.GetPrim().ApplyAPI(name):raise RuntimeError('Missing native material schema '+name)
  for name,value in {'omniphysics:density':625.,'omniphysics:youngsModulus':case['young_Pa'],'omniphysics:poissonsRatio':.35,'physxDeformableMaterial:elasticityDamping':0.}.items():
   if not material.GetPrim().GetAttribute(name).Set(value):raise RuntimeError('Material attr refused '+name)
  UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
  floor_record=None
  if case['kind']=='drop':
   floor=UsdGeom.Cube.Define(stage,'/World/Floor');floor.CreateSizeAttr(1.);floor.AddTranslateOp().Set(Gf.Vec3d(0,0,-.01));floor.AddScaleOp().Set(Gf.Vec3f(2.,2.,.02));fp=floor.GetPrim();UsdPhysics.CollisionAPI.Apply(fp);fc=PhysxSchema.PhysxCollisionAPI.Apply(fp);fc.CreateContactOffsetAttr(.001);fc.CreateRestOffsetAttr(0.)
   fm=UsdShade.Material.Define(stage,'/World/FloorMaterial');fmapi=UsdPhysics.MaterialAPI.Apply(fm.GetPrim());fmapi.CreateStaticFrictionAttr(.5);fmapi.CreateDynamicFrictionAttr(.5);fmapi.CreateRestitutionAttr(0.);UsdShade.MaterialBindingAPI.Apply(fp).Bind(fm,materialPurpose='physics')
   if fp.HasAPI(UsdPhysics.RigidBodyAPI) or not fp.HasAPI(UsdPhysics.CollisionAPI):raise RuntimeError('Floor must be static collision only')
   actualtop=float(np.max(np.array(UsdGeom.BBoxCache(0.,[UsdGeom.Tokens.default_]).ComputeWorldBound(fp).ComputeAlignedRange().GetMax())[2]))
   if abs(actualtop)>1e-8:raise RuntimeError('Floor worldtop notz0')
   floor_record=dict(path='/World/Floor',static=True,collision_enabled=bool(UsdPhysics.CollisionAPI(fp).GetCollisionEnabledAttr().Get()),world_top_z_m=actualtop,contact_offset_m=fc.GetContactOffsetAttr().Get(),rest_offset_m=fc.GetRestOffsetAttr().Get(),friction=.5,restitution=0.)
  if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.GetTypeName()=='OmniPhysicsVtxXformAttachment' for p in stage.Traverse()):raise RuntimeError('No handles/rigid actors/attachments allowed')
  stage.GetRootLayer().Export(str(out/'initial-scene.usda'));world.reset();body=DeformablePrim('/World/Block',deformable_type='volume');body.set_nodal_positions(X[None].astype(np.float32));body.set_nodal_velocities(np.zeros((1,len(X),3),np.float32))
  enabled=np.zeros((1,len(X),1),bool);enabled[0,fixed,0]=True;body.set_nodal_kinematic_position_targets(X[None].astype(np.float32),enabled=enabled)
  view=body._physics_deformable_body_view;sim=SimulationManager.get_physics_simulation_view();sim.set_gravity(d['gravity_m_s2']);gravity=np.asarray(sim.get_gravity(),dtype=float)
  np.testing.assert_array_equal(gravity.astype(np.float32),np.array(d['gravity_m_s2'],dtype=np.float32))
  flags=check_flags(body.get_nodal_kinematic_position_targets()[1].numpy(),view.get_simulation_nodal_kinematic_targets().numpy(),len(X),fixed)
  geom=native_geometry(view.get_rest_nodal_positions().numpy(),view.get_rest_element_indices().numpy(),X,T)
  mv=sim.create_deformable_material_view('/World/Material')
  if mv.count!=1:raise RuntimeError('Native material count differs')
  native={}
  for name,value,expected in [('E_Pa',mv.get_youngs_modulus().numpy(),case['young_Pa']),('nu',mv.get_poissons_ratio().numpy(),.35)]:
   a=np.asarray(value);e=np.float32(expected)
   if a.shape!=(1,1) or abs(float(a[0,0])-float(e))>2*abs(float(np.spacing(e))):raise RuntimeError('Native material binding wrong '+name)
   native[name]=float(a[0,0])
  deviceinfo=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
  allowed=deviceinfo.splitlines()[0].split(',')[1].strip();compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
  own=[r for r in compute.splitlines() if r.split(',')[0].strip()==str(os.getpid())]
  if not own or any(r.split(',')[1].strip()!=allowed for r in own):raise RuntimeError('Worker compute must be solelyGPU0')
  q0=body.get_nodal_positions()[0].numpy()[0].copy();v0=body.get_nodal_velocities().numpy()[0].copy();np.testing.assert_array_equal(q0,X.astype(np.float32))
  if np.max(abs(v0))>1e-9:raise RuntimeError('Native initial velocity notzero')
  state['native_verification']=dict(material=native,geometry=geom,flags=flags,gravity_readback_m_s2=gravity.tolist(),density_authored_kg_m3=material.GetPrim().GetAttribute('omniphysics:density').Get(),floor=floor_record,own_compute_records=own,gpu_enumeration=deviceinfo,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),solver_authored=api.GetSolverTypeAttr().Get(),deform_iterations_authored=prim.GetAttribute('physxDeformableBody:solverPositionIterationCount').Get(),gravity_disabled_authored=prim.GetAttribute('physxDeformableBody:disableGravity').Get())
  write(out/'native-verification.json',state['native_verification']);baseclock=world.current_time
  center0=w@q0.astype(float);stepcount=round(case['duration_s']*hz)
  for step in range(stepcount+1):
   if step:
    before.append(world.current_time-baseclock);world.step(render=False,update_fabric=True);after.append(world.current_time-baseclock)
   q=body.get_nodal_positions()[0].numpy()[0].copy();v=body.get_nodal_velocities().numpy()[0].copy()
   if not np.isfinite(q).all() or not np.isfinite(v).all():raise RuntimeError('Nonfinite native q/v')
   J=np.linalg.det(q[T[:,1:]].astype(float)-q[T[:,:1]].astype(float))/det0
   if J.min()<=0:raise RuntimeError('Native inverted tet')
   pinerror=float(np.max(abs(q[fixed]-q0[fixed]))) if len(fixed) else 0.
   if pinerror>2e-6:raise RuntimeError('Direct fixed-node drift exceeds2um')
   Jextrema.append([float(J.min()),float(J.max())]);clamp.append(pinerror);centers.append(w@q.astype(float));velocities.append(w@v.astype(float))
   if step%(hz//d['fps'])==0:
    check_flags(body.get_nodal_kinematic_position_targets()[1].numpy(),view.get_simulation_nodal_kinematic_targets().numpy(),len(X),fixed)
    times.append(step/hz);actualtimes.append(world.current_time-baseclock);snapq.append(q);snapv.append(v)
   if step%24==0:write(out/'progress.json',dict(step=step,total=stepcount,actual_time_s=world.current_time-baseclock,minimum_J=float(J.min())))
  dt=np.array(after)-np.array(before);expected=float(np.float32(1/hz))
  if np.max(abs(dt-expected))>1e-9 or np.max(abs(np.array(before)[1:]-np.array(after)[:-1]))>1e-12:raise RuntimeError('Native actual stepclock differs')
  if abs(world.get_physics_dt()-1/hz)>1e-8:raise RuntimeError('Native dtgetter mismatch')
  np.testing.assert_array_equal(np.asarray(sim.get_gravity()).astype(np.float32),gravity.astype(np.float32))
  qarr=np.asarray(snapq);varr=np.asarray(snapv);centerarr=np.asarray(centers);velocityarr=np.asarray(velocities)
  displacement=qarr.astype(float)-q0.astype(float);nonuniform=displacement-(np.einsum('n,tnc->tc',w,displacement))[:,None]
  floorstats=None
  if case['kind']=='drop':
   mins=qarr[:,:,2].min(1);floorstats=dict(minimum_saved_node_z_m=float(mins.min()),maximum_saved_nodal_penetration_m=float(max(0,-mins.min())),saved_first_node_at_or_below_floor_s=next((float(t) for t,z in zip(times,mins) if z<=0),None),expected_freefall_firstcontact_s=float(np.sqrt(2*.05/9.81)),qualification='Nodal snapshot penetration only; no contact impulse/accuracy or continuous surface nonpenetration certification')
  state.update(completed=True,frames=len(times),steps=stepcount,clock=dict(float32_dt_s=expected,actual_dt_min_s=float(dt.min()),actual_dt_max_s=float(dt.max()),max_step_error_s=float(np.max(abs(dt-expected))),final_actual_time_s=float(after[-1])),minimum_J=float(np.min(Jextrema)),maximum_J=float(np.max(Jextrema)),maximum_fixed_drift_m=float(max(clamp)),maximum_translation_removed_coordinate_RMS_m=float(np.sqrt(np.mean(nonuniform**2,axis=(1,2))).max()),floor_diagnostic=floorstats)
  if case['kind']=='freefall':
   ev,exact,kick=ballistic_reference(gravity,dt);ve=velocityarr[1:]-ev;pe=centerarr[1:]-center0;tol=max(2e-5,.001*float(np.linalg.norm(ev,axis=1).max()));spread=state['maximum_translation_removed_coordinate_RMS_m'];jdev=float(np.max(abs(np.array(Jextrema)-1)))
   state['ballistic_control']=dict(velocity_max_vector_error_m_s=float(np.linalg.norm(ve,axis=1).max()),velocity_tolerance_m_s=tol,position_coordinate_RMS_vs_exact_m=float(np.sqrt(np.mean((pe-exact)**2))),position_coordinate_RMS_vs_kickdrift_m=float(np.sqrt(np.mean((pe-kick)**2))),maximum_J_deviation=jdev,passed=bool(np.linalg.norm(ve,axis=1).max()<=tol and spread<=2e-5 and jdev<=.001),position_convention_not_selected_for_qualification=True)
  state['elapsed_s']=time.monotonic()-start
 except Exception as e:
  state.update(completed=False,failed_step=step,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-start);traceback.print_exc()
 finally:
  # Persist all evidence BEFORE fast_shutdown; exit0 alone is not completion.
  if not state['completed'] and 'q' in locals() and 'v' in locals():
   np.savez_compressed(out/'failure-state.npz',positions_m=q,velocities_m_s=v,step=np.array(step),minimum_J=np.array(float(J.min()) if 'J' in locals() else np.nan))
  np.savez_compressed(out/('trajectory.npz' if state['completed'] else 'partial-trajectory.npz'),times_s=np.asarray(times),actual_times_s=np.asarray(actualtimes),positions_m=np.asarray(snapq),velocities_m_s=np.asarray(snapv),rest_m=X,faces=F,tets=T,fixed_node_ids=fixed,step_clock_before_s=np.asarray(before),step_clock_after_s=np.asarray(after),step_centers_m=np.asarray(centers),step_velocities_m_s=np.asarray(velocities),step_J_minmax=np.asarray(Jextrema),step_clamp_max_error_m=np.asarray(clamp),floor_z_m=np.array(0. if case['kind']=='drop' else np.nan))
  state['trajectory_path']='trajectory.npz' if state['completed'] else 'partial-trajectory.npz';state['trajectory_sha256']=sha(out/state['trajectory_path']);write(out/'summary.json',state);code=0 if state['completed'] else 1;print(json.dumps(dict(case=case_id,completed=state['completed'],error=state.get('error'))),flush=True);app.close()
 return code

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--case',required=True);a=p.parse_args();raise SystemExit(worker(a.protocol,a.protocol_sha256,a.case))
