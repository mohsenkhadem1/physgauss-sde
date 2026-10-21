"""Free whole-body momentum diagnostic with the original two rigid handles.

The fixture and background drag are removed; no external contact/image capture. Volume-weighted
nodal velocity is a declared uniform-lumped-density proxy, NOT a native mass
getter. Three declared cells separate inferred mass/COM dynamics from support
work and test coordinate sensitivity without claiming native accuracy.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
try:
    from campaign_free_handle_control import sha,write_json,safe,command_schedule
    from campaign_free_handle_clock_adapter import validated_native_clock
    from campaign_numerical_matched_rigid import verify_native_material,verify_native_rest
except ImportError:
    from experiments.campaign_free_handle_control import sha,write_json,safe,command_schedule
    from experiments.campaign_free_handle_clock_adapter import validated_native_clock
    from experiments.campaign_numerical_matched_rigid import verify_native_material,verify_native_rest


def volume_weights(points,tets):
    points=np.asarray(points,dtype=float);tets=np.asarray(tets,dtype=int)
    volumes=np.linalg.det(points[tets[:,1:]]-points[tets[:,:1]])/6
    if not np.isfinite(volumes).all() or np.any(volumes<=0):raise ValueError('Positive rest volumes required')
    weights=np.zeros(len(points));np.add.at(weights,tets.ravel(),np.repeat(volumes/4,4))
    total=float(volumes.sum());weights/=total
    return weights,total,volumes


def mass_momentum_fit(body_velocity,rigid_velocity,rigid_masses,impulse,declared_body_mass):
    body=np.asarray(body_velocity,dtype=float);rigid=np.asarray(rigid_velocity,dtype=float)
    body=body-body[0];rigid=rigid-rigid[0]
    rigid_momentum=np.einsum('thc,h->tc',rigid,np.asarray(rigid_masses,dtype=float))
    target=np.asarray(impulse,dtype=float)-rigid_momentum
    denominator=float(np.sum(body**2))
    if denominator<=1e-24:raise ValueError('No identifiable COM velocity signal for mass fit')
    fitted=float(np.sum(body*target)/denominator)
    fitted_residual=fitted*body+rigid_momentum-impulse
    declared_residual=declared_body_mass*body+rigid_momentum-impulse
    impulse_scale=float(np.max(np.linalg.norm(impulse,axis=-1)))
    return {'fitted_body_mass_kg':fitted,'declared_body_mass_kg':declared_body_mass,
        'fitted_to_declared_mass_ratio':fitted/declared_body_mass,'velocity_design_squared_norm_m2_s2':denominator,
        'fitted_mass_momentum_coordinate_rmse_Ns':float(np.sqrt(np.mean(fitted_residual**2))),
        'declared_mass_momentum_coordinate_rmse_Ns':float(np.sqrt(np.mean(declared_residual**2))),
        'maximum_declared_mass_momentum_residual_Ns':float(np.linalg.norm(declared_residual,axis=-1).max()),
        'maximum_declared_mass_relative_momentum_residual':float(np.linalg.norm(declared_residual,axis=-1).max())/max(impulse_scale,1e-30),
        'maximum_fitted_mass_relative_momentum_residual':float(np.linalg.norm(fitted_residual,axis=-1).max())/max(impulse_scale,1e-30),
        'proxy_qualification':'Scalar LS using uniform density and rest-volume-lumped nodal weights. A good fit is not a native nodal mass getter; nonuniform effective masses or momentum injection can invalidate interpretation.'}


def cotranslation_control(body_velocity,rigid_velocity,rigid_masses,declared_body_mass,tolerance=.01):
    body=np.asarray(body_velocity,dtype=float);rigid=np.asarray(rigid_velocity,dtype=float)
    momentum=declared_body_mass*body+np.einsum('thc,h->tc',rigid,np.asarray(rigid_masses))
    initial=float(np.linalg.norm(momentum[0]))
    if initial<=1e-12:raise ValueError('Co-translation control requires nonzero initial momentum')
    relative=float(np.linalg.norm(momentum-momentum[0],axis=-1).max())/initial
    body_relative=float(np.linalg.norm(body-body[0],axis=-1).max())/float(np.linalg.norm(body[0]))
    return {'maximum_relative_proxy_total_momentum_drift':relative,'maximum_relative_body_COM_velocity_drift':body_relative,
        'predeclared_relative_tolerance':tolerance,'passed':bool(relative<=tolerance and body_relative<=tolerance),
        'qualification':'Engineering proxy control with declared mass/rest-volume lumping. If this fails, driven mass fit cannot be interpreted as density.'}


def inputs(dataset):
    root=Path(dataset)/'episodes/t0001'
    case=json.loads((root/'truth/recipe.json').read_text());spec=json.loads((root/'truth/boundaries.json').read_text())
    if case['split']!='train' or case['episode_id']!='t0001':raise ValueError('TRAIN t0001 only')
    with np.load(root/'reference.npz') as raw:points=raw['rest_points_m'];tets=raw['tetrahedra'];faces=raw['surface_faces']
    return case,spec,points,tets,faces


def create(stage,case,spec,points,tets,faces,solver):
    from pxr import Gf,PhysxSchema,UsdGeom,UsdPhysics,UsdShade,Vt
    prefix='/World/FreeObject';mesh=UsdGeom.TetMesh.Define(stage,prefix+'/Block')
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(tets));mesh.GetSurfaceFaceVertexIndicesAttr().Set(Vt.Vec3iArray.FromNumpy(faces))
    mesh.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.zeros_like(points,dtype=np.float32)))
    prim=mesh.GetPrim()
    for schema in ['OmniPhysicsDeformableBodyAPI','OmniPhysicsVolumeDeformableSimAPI','PhysxBaseDeformableBodyAPI']:
        if not prim.ApplyAPI(schema):raise ValueError('Missing schema '+schema)
    prim.GetAttribute('omniphysics:restShapePoints').Set(mesh.GetPointsAttr().Get())
    prim.GetAttribute('omniphysics:restTetVtxIndices').Set(mesh.GetTetVertexIndicesAttr().Get())
    for key,value in {'disableGravity':True,'solverPositionIterationCount':32,'sleepThreshold':0.,'settlingThreshold':0.,
        'settlingDamping':0.,'linearDamping':0.,'selfCollision':False}.items():
        if not prim.GetAttribute('physxDeformableBody:'+key).Set(value):raise ValueError('Deformable property refused '+key)
    mat=UsdShade.Material.Define(stage,prefix+'/Material')
    for schema in ['OmniPhysicsBaseMaterialAPI','OmniPhysicsDeformableMaterialAPI','PhysxDeformableMaterialAPI']:
        if not mat.GetPrim().ApplyAPI(schema):raise ValueError('Material schema unavailable '+schema)
    for key,value in {'omniphysics:density':case['material']['density_kg_m3'],'omniphysics:youngsModulus':case['material']['youngs_modulus_Pa'],
        'omniphysics:poissonsRatio':case['material']['poissons_ratio'],'physxDeformableMaterial:elasticityDamping':case['material']['elasticity_damping']}.items():
        if not mat.GetPrim().GetAttribute(key).Set(value):raise ValueError('Material refused '+key)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,materialPurpose='physics')
    # Preserve the original volume actor's collision construction, with no
    # external collider, no rigid collision shape and explicit self-collision off.
    UsdPhysics.CollisionAPI.Apply(prim)
    contact=PhysxSchema.PhysxCollisionAPI.Apply(prim);contact.CreateContactOffsetAttr(.0001);contact.CreateRestOffsetAttr(0.)
    handles=[];inertias=[]
    for h,(node,patch) in enumerate(zip(spec['load_nodes'],spec['load_patches'])):
        path=prefix+f'/Load_{h}';handle=UsdGeom.Sphere.Define(stage,path);handle.CreateRadiusAttr(.003)
        handle.AddTranslateOp().Set(Gf.Vec3d(*points[node]));handle.AddOrientOp().Set(Gf.Quatf(1.,Gf.Vec3f(0.)))
        hp=handle.GetPrim();UsdPhysics.RigidBodyAPI.Apply(hp)
        mass=UsdPhysics.MassAPI.Apply(hp);mass.CreateMassAttr(spec['handle_mass_kg_each'])
        offsets=points[patch]-points[node]
        inertia=np.maximum(spec['handle_mass_kg_each']*np.mean(np.sum(offsets**2,axis=1)[:,None]-offsets**2,axis=0),.4*spec['handle_mass_kg_each']*.003**2)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia));inertias.append(inertia)
        api=PhysxSchema.PhysxRigidBodyAPI.Apply(hp)
        api.CreateDisableGravityAttr(True);api.CreateSleepThresholdAttr(0.);api.CreateStabilizationThresholdAttr(0.)
        api.CreateLinearDampingAttr(0.);api.CreateAngularDampingAttr(0.)
        api.CreateSolverPositionIterationCountAttr(32);api.CreateSolverVelocityIterationCountAttr(1)
        attachment=stage.DefinePrim(prefix+f'/Attachment_{h}','OmniPhysicsVtxXformAttachment')
        attachment.GetRelationship('omniphysics:src0').SetTargets([mesh.GetPath()]);attachment.GetRelationship('omniphysics:src1').SetTargets([handle.GetPath()])
        attachment.GetAttribute('omniphysics:vtxIndicesSrc0').Set(patch)
        attachment.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray.FromNumpy(offsets.astype(np.float32)))
        handles.append(path)
    if any(stage.GetPrimAtPath(p).HasAPI(UsdPhysics.CollisionAPI) for p in handles):raise ValueError('Rigid contact collision API unexpectedly present')
    if [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]!=[str(mesh.GetPath())]:
        raise ValueError('Unexpected external collider')
    return mesh,handles,np.asarray(inertias)


def worker(args,protocol):
    if sha(__file__)!=protocol['source_sha256']:raise ValueError('Frozen worker changed')
    for path,digest in protocol['dependency_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen helper changed '+path)
    for path,digest in protocol['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen input changed '+path)
    cell=next(c for c in protocol['config']['cells'] if c['id']==args.cell)
    root=Path(args.output)/cell['id'];root.mkdir(parents=True,exist_ok=False)
    app=None;success=False;action_name=None;step=-1;frames={}
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'active_gpu':1,'physics_gpu':1,'multi_gpu':False,'disable_viewport_updates':True,'fast_shutdown':True})
        import carb
        from pxr import Gf,PhysxSchema,UsdGeom,UsdPhysics,Vt
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import DeformablePrim,RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        carb.settings.get_settings().set_bool('/persistent/physics/enableDeformableBeta',True)
        case,spec,original,tets,faces=inputs(protocol['config']['dataset']);points=original+np.asarray(cell['translation_m'])
        weights,total_volume,volumes=volume_weights(points,tets)
        world=World(stage_units_in_meters=1.,physics_dt=1/960,rendering_dt=1/960,backend='warp',device='cuda:1')
        stage=world.stage;UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        scene=UsdPhysics.Scene.Get(stage,'/physicsScene');scene.CreateGravityMagnitudeAttr(0.)
        scene_api=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());scene_api.CreateEnableGPUDynamicsAttr(True)
        scene_api.CreateBroadphaseTypeAttr('GPU');scene_api.CreateSolverTypeAttr(cell['solver'])
        flag=scene.GetPrim().GetAttribute('physxScene:enableExternalForcesEveryIteration')
        if not flag or not flag.Set(False):raise ValueError('External force flag unavailable')
        mesh,paths,authored_inertias=create(stage,case,spec,points,tets,faces,cell['solver'])
        stage.GetRootLayer().Export(str(root/'initial-scene.usda'))
        np.savez_compressed(root/'reference.npz',points_m=points,tetrahedra=tets,nodal_volume_weights=weights,tet_volumes_m3=volumes)
        record=[];body=None;handles=None
        for action_name,peak,initial_velocity in [('co_translation',[0,0,0],[.001,0,0]),('evaluation',[.01,.01,.005],[0,0,0])]:
            directory=root/action_name;directory.mkdir()
            world.stop();body=None;handles=None;gc.collect()
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.zeros_like(points,dtype=np.float32)))
            for h,path in enumerate(paths):
                prim=stage.GetPrimAtPath(path);xform=UsdGeom.Xformable(prim);xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(*points[spec['load_nodes'][h]]));xform.AddOrientOp().Set(Gf.Quatf(1.,Gf.Vec3f(0.)))
                rigid=UsdPhysics.RigidBodyAPI(prim);rigid.GetVelocityAttr().Set(Gf.Vec3f(0.));rigid.GetAngularVelocityAttr().Set(Gf.Vec3f(0.))
            world.reset();body=DeformablePrim('/World/FreeObject/Block',deformable_type='volume');handles=RigidPrim(paths)
            native_paths=list(handles._physics_rigid_body_view.prim_paths);order=np.array([paths.index(p) for p in native_paths]);inverse=np.argsort(order)
            body.set_nodal_positions(points[None].astype(np.float32));body.set_nodal_velocities(np.broadcast_to(np.asarray(initial_velocity,np.float32),(1,len(points),3)).copy())
            body.set_nodal_kinematic_position_targets(points[None].astype(np.float32),enabled=np.zeros((1,len(points),1),bool))
            _,enabled=body.get_nodal_kinematic_position_targets()
            if enabled.numpy().any():raise ValueError('Fixture flags must ALL be disabled')
            handles.set_world_poses(positions=points[spec['load_nodes']][order].astype(np.float32),orientations=np.tile([1.,0,0,0],(2,1)).astype(np.float32))
            handles.set_velocities(np.tile(initial_velocity,(2,1)).astype(np.float32),np.zeros((2,3),np.float32))
            sim=SimulationManager.get_physics_simulation_view();mv=sim.create_deformable_material_view('/World/FreeObject/Material')
            if mv.count!=1:raise ValueError('Native material count mismatch')
            dv=body._physics_deformable_body_view;rv=handles._physics_rigid_body_view
            native_rest=dv.get_rest_nodal_positions().numpy().copy();native_tets=dv.get_rest_element_indices().numpy().copy()
            material_check=verify_native_material(mv.get_youngs_modulus().numpy(),mv.get_poissons_ratio().numpy(),case['material'])
            rest_check=verify_native_rest(native_rest,native_tets,points,tets)
            np.savez_compressed(directory/'native-rest.npz',points_m=native_rest,tets=native_tets)
            rigid_masses=rv.get_masses().numpy().reshape(-1)[inverse];inertias=rv.get_inertias().numpy().reshape(-1,3,3)[inverse]
            coms=rv.get_coms().numpy()[inverse]
            np.testing.assert_allclose(rigid_masses,spec['handle_mass_kg_each'],atol=1e-10,rtol=1e-6)
            np.testing.assert_allclose(inertias,np.array([np.diag(a) for a in authored_inertias]),atol=1e-13,rtol=1e-6)
            if np.max(abs(coms[:,:3]))>1e-8:raise ValueError('Handle link origins must coincide with native COMs')
            def attributes(prim):return {a.GetName():{'value':safe(a.Get()),'authored':a.HasAuthoredValueOpinion()} for a in prim.GetAttributes() if a.GetName().startswith(('physics:','physx','omniphysics:')) and 'Points' not in a.GetName() and 'Indices' not in a.GetName()}
            metadata={'native_material':material_check,'native_rest':rest_check,'rigid_masses_kg':rigid_masses.tolist(),'rigid_inertias_kg_m2':inertias.tolist(),
                'rigid_COM_native_buffer':coms.tolist(),'native_paths':native_paths,'all_fixture_flags_disabled':True,
                'scene':attributes(scene.GetPrim()),'deformable':attributes(mesh.GetPrim()),'handles':[attributes(stage.GetPrimAtPath(p)) for p in paths],
                'nodal_mass_getter_available':False,'volume_weighted_COM_is_proxy':True,'declared_body_mass_kg':case['material']['density_kg_m3']*total_volume,
                'rest_volume_m3':total_volume,'only_collision_API_on_volume_no_external_colliders':True,'self_collision_explicitly_disabled':True,
                'background_linear_damping_override_s_inverse':0.,'original_background_linear_damping_s_inverse':case['material']['linear_damping_s_inverse'],
                'initial_uniform_velocity_m_s':initial_velocity,'images_generated':False,'sensor_qc_performed':False}
            write_json(directory/'native-state.json',metadata)
            frames={key:[] for key in ['world_times_s','world_step_indices','manager_step_counts','world_dt_s','manager_dt_s','commands_N','command_midpoints_nominal_s',
                'body_volume_weighted_COM_position_m','body_volume_weighted_COM_velocity_m_s','rigid_COM_positions_m','rigid_linear_velocities_m_s','minimum_volume_ratio',
                'sample_times_nominal_s','sample_positions_m','sample_velocities_m_s']}
            midpoint,commands=command_schedule(960,peak=peak);capture_steps=set(np.rint(np.linspace(0,2880,31)).astype(int))
            rest_determinants=volumes*6
            def save_state(index):
                q=body.get_nodal_positions()[0].numpy()[0].copy();v=body.get_nodal_velocities().numpy()[0].copy()
                rp,_=handles.get_world_poses();velocity,_=handles.get_velocities()
                rp=rp.numpy()[inverse].copy();velocity=velocity.numpy()[inverse].copy()
                if not all(np.isfinite(a).all() for a in [q,v,rp,velocity]):raise ValueError('Nonfinite free-whole-body state')
                ratio=np.linalg.det(q[tets[:,1:]].astype(float)-q[tets[:,:1]].astype(float))/rest_determinants
                if np.min(ratio)<=0:raise ValueError('Nonpositive tet volume in free control')
                frames['world_times_s'].append(float(world.current_time));frames['world_step_indices'].append(int(world.current_time_step_index));frames['manager_step_counts'].append(int(SimulationManager.get_num_physics_steps()))
                frames['body_volume_weighted_COM_position_m'].append(weights@q.astype(float));frames['body_volume_weighted_COM_velocity_m_s'].append(weights@v.astype(float))
                frames['rigid_COM_positions_m'].append(rp);frames['rigid_linear_velocities_m_s'].append(velocity);frames['minimum_volume_ratio'].append(float(np.min(ratio)))
                if index in capture_steps:
                    frames['sample_times_nominal_s'].append(index/960);frames['sample_positions_m'].append(q);frames['sample_velocities_m_s'].append(v)
            save_state(0)
            for step,(t,command) in enumerate(zip(midpoint,commands)):
                if flag.Get() is not False:raise ValueError('External force flag changed')
                frames['world_dt_s'].append(float(world.get_physics_dt()));frames['manager_dt_s'].append(float(SimulationManager.get_physics_dt()))
                frames['commands_N'].append(command.copy());frames['command_midpoints_nominal_s'].append(float(t))
                applied=np.zeros((2,3),np.float32);applied[0]=command;handles.apply_forces(applied[order])
                world.step(render=False,update_fabric=True);save_state(step+1)
            np.savez_compressed(directory/'trajectory.npz',**{k:np.asarray(v) for k,v in frames.items()})
            clock=validated_native_clock(frames);dt=clock.pop('measured_step_dt_s')
            if not np.all(np.diff(frames['world_step_indices'])==1) or not np.all(np.diff(frames['manager_step_counts'])==1):raise ValueError('Physics step-count mismatch')
            impulse=np.vstack([np.zeros(3),np.cumsum(commands.astype(float)*dt[:,None],axis=0)])
            body_velocity=np.asarray(frames['body_volume_weighted_COM_velocity_m_s']);rigid_velocity=np.asarray(frames['rigid_linear_velocities_m_s'])
            declared_mass=metadata['declared_body_mass_kg'];momentum=declared_mass*(body_velocity-body_velocity[0])+np.einsum('thc,h->tc',rigid_velocity-rigid_velocity[0],rigid_masses)
            summary={'action':action_name,'complete_steps':2880,'sample_frames':31,'clock':clock,
                'minimum_volume_ratio':float(np.min(frames['minimum_volume_ratio'])),'maximum_declared_mass_momentum_residual_Ns':float(np.linalg.norm(momentum-impulse,axis=-1).max()),
                'maximum_body_COM_drift_m':float(np.linalg.norm(np.asarray(frames['body_volume_weighted_COM_position_m'])-frames['body_volume_weighted_COM_position_m'][0],axis=-1).max()),
                'maximum_body_COM_velocity_m_s':float(np.linalg.norm(body_velocity,axis=-1).max()),
                'trajectory_sha256':sha(directory/'trajectory.npz'),'native_state_sha256':sha(directory/'native-state.json')}
            if action_name=='evaluation':summary['mass_fit']=mass_momentum_fit(body_velocity,rigid_velocity,rigid_masses,impulse,declared_mass)
            if action_name=='co_translation':summary['co_translation_control']=cotranslation_control(body_velocity,rigid_velocity,rigid_masses,declared_mass,protocol['config']['co_translation_relative_tolerance'])
            write_json(directory/'analysis.json',summary);record.append(summary)
            print(json.dumps({'cell':cell['id'],**summary}),flush=True)
        write_json(root/'analysis.json',{'cell':cell,'actions':record,'capture_verified':True,
            'mass_fit_interpretable_under_control':record[0]['co_translation_control']['passed'],
            'qualification':protocol['config']['qualification']});success=True
    except Exception as error:
        if action_name is not None:
            np.savez_compressed(root/action_name/'partial-trajectory.npz',**{k:np.asarray(v) for k,v in frames.items()})
        write_json(root/'failure.json',{'error':str(error),'traceback':traceback.format_exc(),'action':action_name,'step':step});raise
    finally:
        write_json(root/'worker-status.json',{'terminal':True,'success':success,'pid':os.getpid(),'action':action_name,'step':step,'finished_unix_s':time.time()})
        if app is not None:app.close()


def driver(args,config):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    base_dir=Path(__file__).parent
    dependencies=[base_dir/p for p in ['campaign_free_handle_control.py','campaign_free_handle_clock_adapter.py','campaign_numerical_matched_rigid.py']]
    case,spec,points,tets,_=inputs(config['dataset']);weights,volume,_=volume_weights(points,tets)
    original=Path(config['dataset'])/'episodes/t0001'
    protocol={'config':config,'config_sha256':sha(args.config),'source_sha256':sha(__file__),
        'dependency_sha256':{str(p):sha(p) for p in dependencies},'input_sha256':{str(original/p):sha(original/p) for p in ['reference.npz','truth/recipe.json','truth/boundaries.json']},
        'declared_body_mass_kg':volume*case['material']['density_kg_m3'],'rest_volume_m3':volume,
        'force_clock':'Commands at nominal midpoints float32; impulse integrates measured world-clock intervals independently checked against float32-rounded nominal getters.',
        'mass_proxy':'Uniform rest-tet volume lumping determines normalized nodal weights. This is a proxy mass constraint, not a native nodal mass getter.',
        'intentional_background_linear_damping_override_s_inverse':0.,'original_background_linear_damping_s_inverse':case['material']['linear_damping_s_inverse'],
        'elasticity_damping_preserved':case['material']['elasticity_damping'],
        'co_translation_initial_velocity_m_s':[.001,0,0],'evaluation_initial_velocity_m_s':[0,0,0],
        'momentum_caveat':'Internal elasticity damping is retained and expected to conserve global momentum, but native implementation is not assumed correct. The unforced co-translation control must pass a predeclared1% momentum/velocity-drift engineering proxy gate before interpreting driven mass fit.',
        'cpu_environment_threads':2,'gpu':1}
    if (root/'protocol.json').exists():
        old=json.loads((root/'protocol.json').read_text())
        if old!=protocol:raise ValueError('Frozen protocol differs; preserve output')
    else:
        write_json(root/'protocol.json',protocol);(root/'source.py').write_bytes(Path(__file__).read_bytes());(root/'config.json').write_bytes(Path(args.config).read_bytes())
    if not args.launch:return {'prepared':True,'simulation_launched':False,'protocol':str(root/'protocol.json')}
    if (root/'execution.json').exists():raise FileExistsError('Never rerun existing execution')
    ledger={'driver_pid':os.getpid(),'cells':[],'terminal':False};write_json(root/'execution.json',ledger)
    for cell in config['cells']:
        command=[config['isaac_python'],'-u',str(Path(__file__).resolve()),'--worker','--cell',cell['id'],'--output',str(root)]
        row={**cell,'state':'running','started_unix_s':time.time(),'command':command}
        with (root/(cell['id']+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            row['pid']=process.pid;ledger['cells'].append(row);write_json(root/'execution.json',ledger);code=process.wait()
        status_path=root/cell['id']/'worker-status.json';status=json.loads(status_path.read_text()) if status_path.exists() else {}
        row.update(exit_code=code,state='completed_verified' if status.get('success') and (root/cell['id']/'analysis.json').exists() else 'failed',finished_unix_s=time.time())
        write_json(root/'execution.json',ledger)
    ledger['terminal']=True;write_json(root/'execution.json',ledger)
    rows=[]
    for cell in config['cells']:
        path=root/cell['id']/'analysis.json';rows.append(json.loads(path.read_text()) if path.exists() else {'cell':cell,'capture_verified':False,'failure_path':str(root/cell['id']/'failure.json')})
    comparison=None
    if all((root/c/'evaluation/trajectory.npz').exists() for c in ['original-tgs','recentered-tgs']):
        with np.load(root/'original-tgs/evaluation/trajectory.npz') as a,np.load(root/'recentered-tgs/evaluation/trajectory.npz') as b:
            comparison={'sample_q_coordinate_rmse_translation_undone_m':float(np.sqrt(np.mean((a['sample_positions_m'].astype(float)-(b['sample_positions_m'].astype(float)+[0,0,.25]))**2))),
                'body_COM_velocity_coordinate_rmse_m_s':float(np.sqrt(np.mean((a['body_volume_weighted_COM_velocity_m_s']-b['body_volume_weighted_COM_velocity_m_s'])**2)))}
    result={'cells':rows,'translation_comparison':comparison,'qualification':config['qualification']};write_json(root/'analysis.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign_free_deformable_control_config.json'))
    p.add_argument('--output',default='runs/gaussian-campaign-v01/native-free-deformable-v02');p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--cell');args=p.parse_args()
    if args.worker:worker(args,json.loads((Path(args.output)/'protocol.json').read_text()))
    else:print(json.dumps(driver(args,json.loads(args.config.read_text())),indent=2))
