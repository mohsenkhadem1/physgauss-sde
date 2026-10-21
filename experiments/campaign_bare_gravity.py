"""Mass-independent bare deformable gravity-pulse diagnostic; no sensor claims."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
try:
    import isaac_calibrated_probes_v03 as base
    import campaign_numerical_lowrate as geometry_wrapper
    import campaign_numerical_matched_rigid as verified
    from campaign_numerical_convergence import atomic,checked_write,sha,digest
except ImportError:
    from experiments import isaac_calibrated_probes_v03 as base
    from experiments import campaign_numerical_lowrate as geometry_wrapper
    from experiments import campaign_numerical_matched_rigid as verified
    from experiments.campaign_numerical_convergence import atomic,checked_write,sha,digest


def integrate_held_acceleration(acceleration,dt):
    a=np.asarray(acceleration,dtype=float);dt=np.asarray(dt,dtype=float)
    if a.ndim!=2 or a.shape[1]!=3 or dt.shape!=(len(a),) or not np.isfinite(a).all() or not np.isfinite(dt).all() or np.any(dt<=0):
        raise ValueError('Finite Nx3 acceleration and positive actual timesteps required')
    velocity=np.cumsum(a*dt[:,None],axis=0)
    previous=np.concatenate([np.zeros((1,3)),velocity[:-1]],axis=0)
    exact=np.cumsum(previous*dt[:,None]+.5*a*dt[:,None]**2,axis=0)
    kick=np.cumsum(velocity*dt[:,None],axis=0)
    return velocity,exact,kick


def clock_audit(before,after,hz):
    before=np.asarray(before,dtype=float);after=np.asarray(after,dtype=float)
    if before.shape!=after.shape or before.ndim!=1 or not len(before):raise ValueError('Aligned nonempty clock sequences required')
    dt=after-before;expected=float(np.float32(1/hz))
    if not np.isfinite(dt).all() or np.any(dt<=0):raise ValueError('Physics clock must advance')
    if np.max(np.abs(dt-expected))>1e-9:raise ValueError('Physics advance differs from representable float32 dt')
    if len(before)>1 and np.max(np.abs(before[1:]-after[:-1]))>1e-12:raise ValueError('Clock sequence discontinuity')
    return {'float32_nominal_dt_s':expected,'actual_dt_min_s':float(dt.min()),'actual_dt_max_s':float(dt.max()),
        'max_step_error_vs_float32_dt_s':float(np.max(np.abs(dt-expected))),
        'max_cumulative_error_vs_float32_dt_s':float(np.max(np.abs(after-before[0]-np.arange(1,len(dt)+1)*expected)))}


def volume_weights(points,tets):
    det=np.linalg.det(points[tets[:,1:]]-points[tets[:,:1]])
    if np.any(det<=0):raise ValueError('Positive reference tetrahedra required')
    weights=np.zeros(len(points));np.add.at(weights,tets.ravel(),np.repeat(det/24,4));weights/=weights.sum()
    return weights,det


def worker(args,config):
    cell=next(c for c in config['cells'] if c['name']==args.cell)
    if sha(base.__file__)!=config['base_worker_sha256']:raise ValueError('Frozen base changed')
    if sha(verified.__file__)!=config['native_verifier_source_sha256']:raise ValueError('Native verifier changed')
    if sha(geometry_wrapper.__file__)!=config['geometry_wrapper_sha256']:raise ValueError('Geometry wrapper changed')
    root=Path(args.cell_output);root.mkdir(parents=True,exist_ok=False);(root/'truth').mkdir();(root/'provenance').mkdir()
    original=Path(config['dataset'])/'episodes/t0001';case=json.loads((original/'truth/recipe.json').read_text())
    if (case['split'],case['episode_id'],case['boundary_type'])!=('train','t0001','fixed_patch'):raise ValueError('Wrong TRAIN geometry source')
    case=copy.deepcopy(case);case['physics_hz']=cell['physics_hz'];case['material']['linear_damping_s_inverse']=0.;case['material']['elasticity_damping']=0.
    case.pop('recipe_sha256');case['recipe_sha256']=digest(case)
    atomic(root/'truth/recipe.json',case)
    atomic(root/'truth/diagnostic-overrides.json',{'original_recipe_sha256':sha(original/'truth/recipe.json'),
        'physical_fixture_nodes':[],'original_fixture_ids_are_unused':True,'handles_removed':True,'attachments_removed':True,
        'linear_damping_s_inverse':0.,'elasticity_damping':0.,'self_collision':False,
        'body_gravity_enabled':True,'scene_solver_request':cell['solver'],'translation_m':cell['translation_m'],
        'qualification':config['qualification']})
    for path in [Path(__file__),Path(base.__file__),Path(verified.__file__),Path(geometry_wrapper.__file__)]:
        checked_write(root/'provenance'/path.name,path.read_bytes())
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':config['gpu'],'physics_gpu':config['gpu'],
        'multi_gpu':False,'disable_viewport_updates':True,'fast_shutdown':True,'width':64,'height':64})
    code=0;step=None
    try:
        import carb
        from pxr import UsdGeom,UsdPhysics,PhysxSchema
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import DeformablePrim
        from isaacsim.core.simulation_manager import SimulationManager
        from isaac_dataset import envelope
        carb.settings.get_settings().set_bool('/persistent/physics/enableDeformableBeta',True)
        hz=cell['physics_hz'];world=World(stage_units_in_meters=1.,physics_dt=1/hz,rendering_dt=1/hz,backend='warp',device=f"cuda:{config['gpu']}")
        stage=world.stage;UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        scene=UsdPhysics.Scene.Get(stage,'/physicsScene');scene.CreateGravityMagnitudeAttr(0.)
        scene_api=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());scene_api.CreateEnableGPUDynamicsAttr(True);scene_api.CreateBroadphaseTypeAttr('GPU')
        scene_api.CreateSolverTypeAttr(cell['solver']);scene_api.CreateEnableExternalForcesEveryIterationAttr(False)
        geometry_wrapper.install_geometry(config,cell,hz)
        env=base.create_environment(stage,case,root);points,tets=env['points'],env['tets']
        for i in range(2):
            if not stage.RemovePrim(f'/World/ProbeObject/Attachment_{i}'):raise ValueError('Attachment removal failed')
            if not stage.RemovePrim(f'/World/ProbeObject/Load_{i}'):raise ValueError('Rigid handle removal failed')
        prim=stage.GetPrimAtPath('/World/ProbeObject/Block')
        for name,value in [('disableGravity',False),('linearDamping',0.),('selfCollision',False),('solverPositionIterationCount',32)]:
            if not prim.GetAttribute('physxDeformableBody:'+name).Set(value):raise ValueError('Bare-body setting refused: '+name)
        material_prim=stage.GetPrimAtPath('/World/ProbeObject/Material')
        if material_prim.GetAttribute('physxDeformableMaterial:elasticityDamping').Get()!=0.:raise ValueError('Elasticity damping override failed')
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()):raise ValueError('Bare diagnostic contains a rigid body')
        stage.GetRootLayer().Export(str(root/'initial-scene.usda'))
        world.reset();body=DeformablePrim('/World/ProbeObject/Block',deformable_type='volume')
        body.set_nodal_positions(points[None].astype(np.float32));body.set_nodal_velocities(np.zeros((1,len(points),3),np.float32))
        disabled=np.zeros((1,len(points),1),bool)
        body.set_nodal_kinematic_position_targets(points[None].astype(np.float32),enabled=disabled)
        _,enabled=body.get_nodal_kinematic_position_targets()
        if np.any(enabled.numpy()):raise ValueError('Bare node remains kinematic')
        view=body._physics_deformable_body_view
        raw_targets=view.get_simulation_nodal_kinematic_targets().numpy()
        if not np.all(raw_targets[0,:,3]==1.):raise ValueError('Raw native kinematic target flags must all be1/free')
        sim=SimulationManager.get_physics_simulation_view();mv=sim.create_deformable_material_view('/World/ProbeObject/Material')
        if mv.count!=1:raise ValueError('Expected one native deformable material')
        native_material=verified.verify_native_material(mv.get_youngs_modulus().numpy(),mv.get_poissons_ratio().numpy(),case['material'])
        native_rest=view.get_rest_nodal_positions().numpy().copy();native_tets=view.get_rest_element_indices().numpy().copy()
        native_geometry=verified.verify_native_rest(native_rest,native_tets,points,tets)
        atomic(root/'native-runtime-state.json',{'material':native_material,'rest_buffers':native_geometry,
            'raw_kinematic_fourth_component_unique':np.unique(raw_targets[0,:,3]).tolist(),
            'body_disableGravity_authored':prim.GetAttribute('physxDeformableBody:disableGravity').Get(),
            'linear_damping_authored':prim.GetAttribute('physxDeformableBody:linearDamping').Get(),
            'elasticity_damping_authored':material_prim.GetAttribute('physxDeformableMaterial:elasticityDamping').Get(),
            'scene_solver_request':scene_api.GetSolverTypeAttr().Get(),'native_nodal_mass_or_inv_mass_observed':False,
            'node_flags_qualification':'Raw kinematic-target flag1 means disabled target. It is not a direct readback of inverse mass.'})
        np.savez_compressed(root/'native-rest.npz',rest_points=native_rest,rest_tets=native_tets)
        weights,det0=volume_weights(points,tets)
        q0=body.get_nodal_positions()[0].numpy()[0].copy();v0=body.get_nodal_velocities().numpy()[0].copy()
        if np.max(abs(v0))>1e-7:raise ValueError('Initial velocity notzero')
        center0=weights@q0.astype(float)
        snapshots={'times_s':[0.],'positions_m':[q0],'velocities_m_s':[v0]}
        before=[];after=[];commands=[];native_commands=[];centers=[];velocities=[];node_spread=[];minJ=1.;maxJ=1.
        sample_steps=set(np.rint(np.linspace(0,3,31)*hz).astype(int));peak=np.array(config['peak_acceleration_m_s2'])
        for step in range(1,3*hz+1):
            midpoint=(step-.5)/hz;command=np.asarray(peak*float(envelope('push_pulse',midpoint,3.)),dtype=np.float32)
            if sim.set_gravity(command.tolist()) is False:raise ValueError('Native gravity setter rejected command')
            actual=np.asarray(sim.get_gravity(),dtype=float)
            np.testing.assert_array_equal(actual.astype(np.float32),command)
            before.append(world.current_time);world.step(render=False,update_fabric=True);after.append(world.current_time)
            q=body.get_nodal_positions()[0].numpy()[0].copy();v=body.get_nodal_velocities().numpy()[0].copy()
            if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Nonfinite bare-body state')
            J=np.linalg.det(q[tets[:,1:]].astype(float)-q[tets[:,:1]].astype(float))/det0
            minJ=min(minJ,float(J.min()));maxJ=max(maxJ,float(J.max()))
            if J.min()<=0:raise ValueError('Bare body inverted')
            center=weights@q.astype(float);velocity=weights@v.astype(float)
            centers.append(center);velocities.append(velocity);commands.append(command);native_commands.append(actual)
            displacement=q.astype(float)-q0.astype(float)
            node_spread.append([float(np.sqrt(np.mean((displacement-(center-center0))**2))),float(np.sqrt(np.mean((v-velocity)**2)))])
            if step in sample_steps:
                if np.any(body.get_nodal_kinematic_position_targets()[1].numpy()):raise ValueError('Node became kinematic')
                snapshots['times_s'].append(step/hz);snapshots['positions_m'].append(q);snapshots['velocities_m_s'].append(v)
        clock=clock_audit(before,after,hz);dt=np.asarray(after)-np.asarray(before)
        expected_v,exact,kick=integrate_held_acceleration(native_commands,dt)
        observed_p=np.asarray(centers)-center0;observed_v=np.asarray(velocities)
        velocity_error=observed_v-expected_v;scale=float(np.linalg.norm(expected_v,axis=1).max())
        velocity_rmse=float(np.sqrt(np.mean(velocity_error**2)));velocity_max=float(np.linalg.norm(velocity_error,axis=1).max())
        basis=kick-exact;den=float(np.sum(basis*basis));theta=.5+.5*float(np.sum((observed_p-exact)*basis))/den if den else None
        report={'cell':cell,'clock':clock,'world_get_physics_dt_s':float(world.get_physics_dt()),
            'minimum_saved_and_runtime_volume_ratio':minJ,'maximum_saved_and_runtime_volume_ratio':maxJ,
            'expected_final_velocity_m_s':expected_v[-1].tolist(),'observed_final_velocity_m_s':observed_v[-1].tolist(),
            'expected_final_position_exact_m':exact[-1].tolist(),'expected_final_position_kick_drift_m':kick[-1].tolist(),
            'observed_final_position_m':observed_p[-1].tolist(),'velocity_rmse_m_s':velocity_rmse,
            'velocity_max_error_m_s':velocity_max,'velocity_max_error_relative_to_peak_speed':velocity_max/max(scale,1e-15),
            'position_rmse_vs_exact_held_acceleration_m':float(np.sqrt(np.mean((observed_p-exact)**2))),
            'position_rmse_vs_kick_drift_m':float(np.sqrt(np.mean((observed_p-kick)**2))),
            'fitted_uniform_acceleration_position_theta':theta,
            'theta_qualification':'Descriptive fit only: kick-drift corresponds1, exactheldacceleration.5; clock/force timing and representational errors can bias this fit. No convention selected to certify physics.',
            'maximum_node_displacement_nonuniformity_coordinate_rms_m':float(np.max(np.asarray(node_spread)[:,0])),
            'maximum_node_velocity_nonuniformity_coordinate_rms_m_s':float(np.max(np.asarray(node_spread)[:,1])),
            'all_native_material_rest_and_free_flag_guards_passed':True,
            'velocity_reference_tolerance_m_s':max(2e-6,1e-3*scale),'velocity_reference_gate_passed':velocity_max<max(2e-6,1e-3*scale),
            'qualification':config['qualification'],'native_reference_qualified':False}
        np.savez_compressed(root/'trajectory.npz',**{k:np.asarray(v) for k,v in snapshots.items()},
            step_clock_before_s=np.asarray(before),step_clock_after_s=np.asarray(after),native_gravity_acceleration_m_s2=np.asarray(native_commands),
            requested_midpoint_acceleration_m_s2=np.asarray(commands),observed_volume_weighted_centers_m=np.asarray(centers),
            observed_volume_weighted_velocities_m_s=np.asarray(velocities),expected_velocity_m_s=expected_v,
            expected_displacement_exact_m=exact,expected_displacement_kick_drift_m=kick,volume_weights=weights,node_spread=np.asarray(node_spread))
        report['trajectory_sha256']=sha(root/'trajectory.npz');atomic(root/'analysis.json',report)
        print(json.dumps({'cell':args.cell,'velocity_reference_gate_passed':report['velocity_reference_gate_passed'],
            'relative_velocity_error':report['velocity_max_error_relative_to_peak_speed'],'position_rmse_exact_m':report['position_rmse_vs_exact_held_acceleration_m']}),flush=True)
    except Exception as error:
        import traceback
        atomic(root/'failure.json',{'step':step,'error':repr(error),'traceback':traceback.format_exc()});traceback.print_exc();code=1
    finally:app.close()
    return code


def driver(args,config):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    expected=[('TGS',120,[0.,0.,0.]),('TGS',120,[0.,0.,-.25]),('TGS',960,[0.,0.,0.]),('TGS',960,[0.,0.,-.25]),('PGS',960,[0.,0.,0.]),('PGS',960,[0.,0.,-.25])]
    if [(c['solver'],c['physics_hz'],c['translation_m']) for c in config['cells']]!=expected:raise ValueError('Frozen six-cell design required')
    protocol={'config':config,'config_sha256':sha(args.config),'wrapper_sha256':sha(__file__)}
    checked_write(output/'protocol.json',(json.dumps(protocol,indent=2)+'\n').encode())
    for path in [Path(__file__),Path(args.config)]:checked_write(output/'provenance'/path.name,path.read_bytes())
    if not args.launch:return {'prepared':True,'simulation_launched':False}
    if (output/'execution.json').exists():raise FileExistsError('Existing execution preserved')
    ledger={'driver_pid':os.getpid(),'cells':[],'terminal':False,'completed':False}
    for cell in config['cells']:
        if sha(__file__)!=protocol['wrapper_sha256']:raise ValueError('Source changed')
        command=[config['isaac_python'],'-u',str(Path(__file__)),'--config',str(args.config),'--worker','--cell',cell['name'],'--cell-output',str(output/cell['name'])]
        row={'cell':cell,'command':command,'started_unix_s':time.time()}
        with (output/(cell['name']+'.log')).open('x') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
            row.update(pid=child.pid,state='running');ledger['cells'].append(row);atomic(output/'execution.json',ledger);code=child.wait()
        row.update(exit_code=code,state='completed' if code==0 else 'failed',finished_unix_s=time.time());atomic(output/'execution.json',ledger)
    ledger.update(terminal=True,completed=all(r['state']=='completed' for r in ledger['cells']));atomic(output/'execution.json',ledger)
    reports=[]
    for row in ledger['cells']:
        path=output/row['cell']['name']/('analysis.json' if row['exit_code']==0 else 'failure.json')
        reports.append({'name':row['cell']['name'],'state':row['state'],'result':json.loads(path.read_text())})
    result={'cells':reports,'native_reference_qualified':False,'qualification':config['qualification']}
    atomic(output/'analysis.json',result)
    return {'terminal':True,'completed':ledger['completed'],'cells':[{k:r[k] for k in ['name','state']} for r in reports]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign_bare_gravity_config.json'))
    p.add_argument('--output',default='runs/gaussian-campaign-v01/bare-gravity-v01');p.add_argument('--launch',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--cell');p.add_argument('--cell-output')
    args=p.parse_args();config=json.loads(args.config.read_text())
    if args.worker:raise SystemExit(worker(args,config))
    print(json.dumps(driver(args,config),indent=2))
