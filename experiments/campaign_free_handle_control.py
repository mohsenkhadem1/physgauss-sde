"""Native free-rigid force API diagnostic; no deformable/attachment/sensor data.

One authored TRAIN t0001 1g handle, no collision or gravity. Four predeclared
rate/flag cells isolate force magnitude/integration without establishing the
correctness of rigid-deformable coupling. Existing workers remain untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def pulse(times,seconds=3.):
    phase=(np.asarray(times,dtype=float)/seconds-.1)/.45
    return np.where((phase>0)&(phase<1),np.sin(np.pi*np.clip(phase,0,1))**2,0.)


def command_schedule(hz,seconds=3.,peak=(.01,.01,.005)):
    steps=round(seconds*hz)
    if hz<=0 or abs(steps-seconds*hz)>1e-10:raise ValueError('Positive Hz/integral step count required')
    midpoints=(np.arange(steps)+.5)/hz
    commands=(pulse(midpoints,seconds)[:,None]*np.asarray(peak,dtype=float)).astype(np.float32)
    return midpoints,commands


def free_held_reference(initial_position,initial_velocity,mass,forces,dt):
    forces=np.asarray(forces,dtype=float);dt=np.broadcast_to(np.asarray(dt,dtype=float),(len(forces),))
    if mass<=0 or np.any(dt<=0) or forces.shape!=(len(dt),3):raise ValueError('Invalid reference mass/force/clock')
    q=[np.asarray(initial_position,dtype=float)];kick=[q[0].copy()];v=[np.asarray(initial_velocity,dtype=float)]
    for force,step_dt in zip(forces,dt):
        dv=force*step_dt/mass
        q.append(q[-1]+v[-1]*step_dt+.5*dv*step_dt)
        kick.append(kick[-1]+(v[-1]+dv)*step_dt)
        v.append(v[-1]+dv)
    return {'positions_exact_piecewise_constant_m':np.asarray(q),
            'positions_kick_drift_m':np.asarray(kick),'velocities_m_s':np.asarray(v),
            'impulse_Ns':np.sum(forces*dt[:,None],axis=0)}


def continuous_pulse_reference(times,initial_position,mass,peak=(.01,.01,.005),seconds=3.):
    times=np.asarray(times,dtype=float);start=.1*seconds;duration=.45*seconds;omega=2*np.pi/duration
    elapsed=np.maximum(times-start,0);s=np.minimum(elapsed,duration)
    integral=.5*(s-np.sin(omega*s)/omega)
    displacement_integral=s*s/4+(np.cos(omega*s)-1)/(2*omega*omega)+(elapsed-s)*integral
    return {'positions_m':np.asarray(initial_position)+displacement_integral[:,None]*np.asarray(peak)/mass,
            'velocities_m_s':integral[:,None]*np.asarray(peak)/mass,
            'impulse_Ns':np.asarray(peak)*duration/2}


def hardware_from_train(dataset):
    root=Path(dataset)/'episodes/t0001';case=json.loads((root/'truth/recipe.json').read_text())
    spec=json.loads((root/'truth/boundaries.json').read_text())
    if case['episode_id']!='t0001' or case['split']!='train' or spec['handle_mass_kg_each']!=.001:
        raise ValueError('Expected unchanged TRAIN t0001 hardware')
    with np.load(root/'reference.npz') as raw:points=raw['rest_points_m']
    node=spec['load_nodes'][0];offsets=points[spec['load_patches'][0]]-points[node];mass=spec['handle_mass_kg_each']
    inertia=np.maximum(mass*np.mean(np.sum(offsets**2,axis=1)[:,None]-offsets**2,axis=0),.4*mass*.003**2)
    return {'mass_kg':mass,'diagonal_inertia_kg_m2':inertia.tolist(),'position_m':points[node].tolist(),
            'radius_m':.003,'source_patch_nodes':spec['load_patches'][0],'source_load_node':node,
            'input_sha256':{str(root/name):sha(root/name) for name in ['truth/recipe.json','truth/boundaries.json','reference.npz']}}


def safe(value):
    if value is None or isinstance(value,(str,bool,int,float)):return value
    try:return [safe(item) for item in value]
    except TypeError:return str(value)


def analyze_cell(root,cell,protocol):
    root=Path(root);metadata=json.loads((root/'native-state.json').read_text())
    with np.load(root/'trajectory.npz') as archive:data={k:archive[k].copy() for k in archive.files}
    hz=cell['hz'];midpoints,commands=command_schedule(hz)
    np.testing.assert_array_equal(data['command_midpoints_nominal_s'],midpoints)
    np.testing.assert_array_equal(data['commands_N'],commands)
    n=len(commands)
    if data['positions_m'].shape!=(n+1,3) or data['linear_velocities_m_s'].shape!=(n+1,3):raise ValueError('Incomplete saved state')
    for key,value in data.items():
        if not np.isfinite(value).all():raise ValueError('Nonfinite record '+key)
    if not np.all(data['world_step_indices']-data['world_step_indices'][0]==np.arange(n+1)):
        raise ValueError('Native world step count mismatch')
    if not np.all(data['manager_step_counts']-data['manager_step_counts'][0]==np.arange(n+1)):
        raise ValueError('Native physics-manager step count mismatch')
    np.testing.assert_array_equal(data['world_dt_s'],data['manager_dt_s'])
    dt=data['world_dt_s'];elapsed=data['world_times_s']-data['world_times_s'][0]
    clock_error=float(abs(elapsed-np.r_[0.,np.cumsum(dt)]).max())
    if clock_error>1e-9:raise ValueError('Actual native dt/clock mismatch')
    mass=float(np.asarray(metadata['native_properties']['get_masses']).reshape(-1)[0])
    reference=free_held_reference(data['positions_m'][0],data['linear_velocities_m_s'][0],mass,commands,dt)
    continuous=continuous_pulse_reference(np.arange(n+1)/hz,data['positions_m'][0],mass)
    momentum_change=mass*(data['linear_velocities_m_s']-data['linear_velocities_m_s'][0])
    expected_momentum=np.vstack([np.zeros(3),np.cumsum(commands.astype(float)*dt[:,None],axis=0)])
    impulse=reference['impulse_Ns'];scale=float(np.linalg.norm(impulse))
    step_impulses=commands.astype(float)*dt[:,None];dv=np.diff(data['linear_velocities_m_s'].astype(float),axis=0)
    gain=float(np.sum(step_impulses*(mass*dv))/np.sum(step_impulses**2))
    relative_final_impulse_error=float(np.linalg.norm(momentum_change[-1]-impulse)/scale)
    relative_velocity_error=float(np.max(np.linalg.norm(momentum_change-expected_momentum,axis=-1))/scale)
    newton_error=mass*dv-step_impulses
    coordinate_exact=data['positions_m'].astype(float)-reference['positions_exact_piecewise_constant_m']
    coordinate_kick=data['positions_m'].astype(float)-reference['positions_kick_drift_m']
    # Infer the free position integration coefficient: dx-v_old dt = theta dv dt.
    position_increment=np.diff(data['positions_m'].astype(float),axis=0)-data['linear_velocities_m_s'][:-1]*dt[:,None]
    predictor=step_impulses/mass*dt[:,None]
    theta=float(np.sum(position_increment*predictor)/np.sum(predictor**2))
    np.savez_compressed(root/'references.npz',**reference,
        continuous_nominal_positions_m=continuous['positions_m'],continuous_nominal_velocities_m_s=continuous['velocities_m_s'])
    expected_inertia=np.diag(np.asarray(protocol['hardware']['diagonal_inertia_kg_m2']))
    native_inertia=np.asarray(metadata['native_properties']['get_inertias']).reshape(-1,3,3)[0]
    native_com=np.asarray(metadata['native_properties']['get_coms']).reshape(-1,7)[0,:3]
    native_mass_relative=abs(mass/protocol['hardware']['mass_kg']-1)
    inertia_relative=float(np.max(abs(native_inertia-expected_inertia))/abs(expected_inertia).max())
    force_gate=relative_velocity_error<=protocol['config']['relative_momentum_tolerance']
    rotation_gate=float(abs(data['angular_velocities_rad_s']).max())<=protocol['config']['angular_velocity_tolerance_rad_s']
    scene=metadata['scene_attributes'];rigid=metadata['rigid_attributes']
    scene_valid=bool(scene['physxScene:enableExternalForcesEveryIteration']['value'])==cell['enabled']
    counts_valid=rigid['physxRigidBody:solverPositionIterationCount']['value']==32 and rigid['physxRigidBody:solverVelocityIterationCount']['value']==1
    hardware_valid=native_mass_relative<1e-6 and inertia_relative<1e-5 and np.max(abs(native_com))<1e-8
    result={'cell':cell,'native_mass_kg':mass,'native_mass_relative_error':native_mass_relative,
            'native_inertia_max_relative_error':inertia_relative,'native_local_COM_m':native_com.tolist(),
            'clock_error_s':clock_error,'first_world_dt_s':float(dt[0]),'last_elapsed_native_time_s':float(elapsed[-1]),
            'applied_command_impulse_using_native_dt_Ns':impulse.tolist(),
            'continuous_analytic_pulse_impulse_Ns':continuous['impulse_Ns'].tolist(),
            'native_final_momentum_change_Ns':momentum_change[-1].tolist(),
            'relative_final_impulse_error':relative_final_impulse_error,'maximum_relative_cumulative_momentum_error':relative_velocity_error,
            'fitted_step_impulse_gain':gain,'maximum_step_impulse_residual_Ns':float(np.linalg.norm(newton_error,axis=-1).max()),
            'positions_coordinate_rmse_vs_exact_held_force_m':float(np.sqrt(np.mean(coordinate_exact**2))),
            'positions_coordinate_rmse_vs_kick_drift_m':float(np.sqrt(np.mean(coordinate_kick**2))),
            'positions_max_nodal_error_vs_exact_held_force_m':float(np.linalg.norm(coordinate_exact,axis=-1).max()),
            'fitted_position_integration_theta':theta,'theta_interpretation':'Exact constant-force step theta=.5; kick-drift theta=1. Position comparison diagnoses integrator convention; it is not a force-accuracy gate.',
            'maximum_angular_velocity_rad_s':float(abs(data['angular_velocities_rad_s']).max()),
            'zero_command_velocity_change_max_m_s':float(np.linalg.norm(dv[np.all(commands==0,axis=-1)],axis=-1).max()),
            'native_property_gate_passed':bool(hardware_valid),'scene_rigid_flags_passed':bool(scene_valid and counts_valid),
            'force_api_momentum_gate_passed':bool(force_gate),'no_torque_rotation_gate_passed':bool(rotation_gate),
            'passed':bool(force_gate and rotation_gate and hardware_valid and scene_valid and counts_valid),
            'source_sha256':protocol['source_sha256'],'trajectory_sha256':sha(root/'trajectory.npz'),
            'native_state_sha256':sha(root/'native-state.json'),'references_sha256':sha(root/'references.npz'),
            'qualification':protocol['config']['qualification']}
    write_json(root/'analysis.json',result);return result


def worker(args,protocol):
    if sha(__file__)!=protocol['source_sha256']:raise ValueError('Frozen diagnostic source changed')
    for path,digest in protocol['hardware']['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen TRAIN hardware input changed')
    cell=next(cell for cell in protocol['config']['cells'] if cell['id']==args.cell)
    root=Path(args.output)/cell['id'];root.mkdir(parents=True,exist_ok=False)
    app=None
    records={key:[] for key in ['positions_m','orientations_wxyz','linear_velocities_m_s','angular_velocities_rad_s',
        'world_times_s','world_step_indices','manager_step_counts','world_dt_s','manager_dt_s',
        'command_midpoints_nominal_s','commands_N']}
    success=False;step=-1
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'active_gpu':args.gpu,'physics_gpu':args.gpu,
            'multi_gpu':False,'disable_viewport_updates':True,'fast_shutdown':True})
        from pxr import Gf,PhysxSchema,UsdGeom,UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        hz=cell['hz'];hardware=protocol['hardware']
        world=World(stage_units_in_meters=1.,physics_dt=1/hz,rendering_dt=1/hz,backend='warp',device=f'cuda:{args.gpu}')
        stage=world.stage;UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        scene=UsdPhysics.Scene.Get(stage,'/physicsScene');scene.CreateGravityMagnitudeAttr(0.)
        scene_api=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        scene_api.CreateEnableGPUDynamicsAttr(True);scene_api.CreateBroadphaseTypeAttr('GPU');scene_api.CreateSolverTypeAttr('TGS')
        flag=scene.GetPrim().GetAttribute('physxScene:enableExternalForcesEveryIteration')
        if not flag or not flag.Set(cell['enabled']):raise ValueError('Requested scene flag unavailable')
        sphere=UsdGeom.Sphere.Define(stage,'/World/FreeHandle');sphere.CreateRadiusAttr(hardware['radius_m'])
        sphere.AddTranslateOp().Set(Gf.Vec3d(*hardware['position_m']))
        sphere.AddOrientOp().Set(Gf.Quatf(1.,Gf.Vec3f(0.)))
        prim=sphere.GetPrim();UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api=UsdPhysics.MassAPI.Apply(prim);mass_api.CreateMassAttr(hardware['mass_kg'])
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*hardware['diagonal_inertia_kg_m2']))
        rigid_api=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rigid_api.CreateDisableGravityAttr(True);rigid_api.CreateSleepThresholdAttr(0.);rigid_api.CreateStabilizationThresholdAttr(0.)
        rigid_api.CreateLinearDampingAttr(0.);rigid_api.CreateAngularDampingAttr(0.)
        rigid_api.CreateSolverPositionIterationCountAttr(32);rigid_api.CreateSolverVelocityIterationCountAttr(1)
        if prim.HasAPI(UsdPhysics.CollisionAPI):raise ValueError('Free handle must have no CollisionAPI')
        stage.GetRootLayer().Export(str(root/'initial-scene.usda'))
        world.reset();body=RigidPrim('/World/FreeHandle')
        body.set_world_poses(positions=np.asarray(hardware['position_m'],np.float32)[None],orientations=np.array([[1.,0,0,0]],np.float32))
        body.set_velocities(np.zeros((1,3),np.float32),np.zeros((1,3),np.float32))
        view=body._physics_rigid_body_view
        native={name:getattr(view,name)().numpy().copy().tolist() for name in ['get_masses','get_inv_masses','get_inertias','get_coms']}
        def attributes(prim):
            return {a.GetName():{'value':safe(a.Get()),'authored':a.HasAuthoredValueOpinion()} for a in prim.GetAttributes()
                    if a.GetName().startswith(('physics:','physx'))}
        metadata={'provider':'Native RigidBodyView tensor getters after reset; composed USD solver flags separately identified',
            'gpu':args.gpu,'prim_paths':list(view.prim_paths),'native_properties':native,
            'scene_attributes':attributes(scene.GetPrim()),'rigid_attributes':attributes(prim),
            'world_dt_s':float(world.get_physics_dt()),'manager_dt_s':float(SimulationManager.get_physics_dt()),
            'no_collision_api':not prim.HasAPI(UsdPhysics.CollisionAPI),'no_deformable_or_attachments':True,
            'physics_only':True,'images_generated':False,'sensor_qc_performed':False,
            'rigid_prim_implementation_file':str(sys.modules[RigidPrim.__module__].__file__),
            'rigid_prim_implementation_sha256':sha(sys.modules[RigidPrim.__module__].__file__)}
        write_json(root/'native-state.json',metadata)
        midpoints,commands=command_schedule(hz)
        def save_state():
            q,orientation=body.get_world_poses();velocity,angular=body.get_velocities()
            for key,value in [('positions_m',q.numpy()[0]),('orientations_wxyz',orientation.numpy()[0]),
                ('linear_velocities_m_s',velocity.numpy()[0]),('angular_velocities_rad_s',angular.numpy()[0])]:
                value=np.asarray(value).copy()
                if not np.isfinite(value).all():raise ValueError('Nonfinite free state '+key)
                records[key].append(value)
            records['world_times_s'].append(float(world.current_time));records['world_step_indices'].append(int(world.current_time_step_index))
            records['manager_step_counts'].append(int(SimulationManager.get_num_physics_steps()))
        save_state()
        for step,(midpoint,command) in enumerate(zip(midpoints,commands)):
            if flag.Get()!=cell['enabled']:raise ValueError('Scene flag changed during run')
            records['world_dt_s'].append(float(world.get_physics_dt()));records['manager_dt_s'].append(float(SimulationManager.get_physics_dt()))
            records['command_midpoints_nominal_s'].append(float(midpoint));records['commands_N'].append(command.copy())
            body.apply_forces(command[None],local_frame=False)
            world.step(render=False,update_fabric=True)
            save_state()
        np.savez_compressed(root/'trajectory.npz',**{key:np.asarray(value) for key,value in records.items()})
        result=analyze_cell(root,cell,protocol);success=True
        print(json.dumps(result),flush=True)
    except Exception as error:
        np.savez_compressed(root/'partial-trajectory.npz',**{key:np.asarray(value) for key,value in records.items()})
        write_json(root/'failure.json',{'error':str(error),'traceback':traceback.format_exc(),'step':step,'source_sha256':protocol['source_sha256']})
        raise
    finally:
        write_json(root/'worker-status.json',{'terminal':True,'success':success,'pid':os.getpid(),'step':step,'finished_unix_s':time.time()})
        if app is not None:app.close()


def driver(args,config):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    protocol={'config':config,'config_sha256':sha(args.config),'source_sha256':sha(__file__),
              'hardware':hardware_from_train(config['dataset']),
              'gpu':args.gpu,'physics_only':True,'no_sensor_claim':True,
              'references':'Native-mass and actual-dt piecewise-held exact free response, kick-drift convention, and continuous analytic pulse. Commands evaluated at nominal midpoints and rounded once to float32.',
              'prepared_unix_s':time.time()}
    path=root/'protocol.json'
    if path.exists():
        protocol=json.loads(path.read_text())
        if protocol['source_sha256']!=sha(__file__) or protocol['config_sha256']!=sha(args.config) or protocol['gpu']!=args.gpu:
            raise ValueError('Existing frozen protocol differs; preserve it and use a new output')
    else:
        write_json(path,protocol);(root/'source.py').write_bytes(Path(__file__).read_bytes())
        (root/'config.json').write_bytes(Path(args.config).read_bytes())
    if not args.launch:return {'prepared':True,'simulation_launched':False,'protocol':str(path)}
    if (root/'execution.json').exists():raise FileExistsError('Preserve existing native trials; no restart/overwrite')
    ledger={'driver_pid':os.getpid(),'gpu':args.gpu,'cells':[],'completed':False}
    write_json(root/'execution.json',ledger)
    for cell in config['cells']:
        if sha(__file__)!=protocol['source_sha256']:raise ValueError('Diagnostic source changed')
        command=[config['isaac_python'],'-u',str(Path(__file__).resolve()),'--worker','--cell',cell['id'],
                 '--output',str(root),'--gpu',str(args.gpu)]
        row={**cell,'state':'running','command':command,'started_unix_s':time.time()}
        with (root/(cell['id']+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            row['pid']=process.pid;ledger['cells'].append(row);write_json(root/'execution.json',ledger)
            code=process.wait()
        row.update(exit_code=code,state='completed' if code==0 else 'failed',finished_unix_s=time.time())
        write_json(root/'execution.json',ledger)
    rows=[]
    for cell in config['cells']:
        path=root/cell['id']/'analysis.json'
        rows.append(json.loads(path.read_text()) if path.exists() else {'cell':cell,'passed':False,'failure_path':str(root/cell['id']/'failure.json')})
    ledger['completed']=True;ledger['all_cells_passed']=all(row['passed'] for row in rows)
    write_json(root/'execution.json',ledger)
    result={'cells':rows,'all_cells_passed':ledger['all_cells_passed'],'source_sha256':protocol['source_sha256'],
            'protocol_sha256':sha(root/'protocol.json'),'qualification':config['qualification']}
    write_json(root/'analysis.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign_free_handle_control_config.json'))
    parser.add_argument('--output',default='runs/gaussian-campaign-v01/native-free-handle-v01')
    parser.add_argument('--gpu',type=int,choices=[1,2],default=1)
    parser.add_argument('--launch',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--cell')
    args=parser.parse_args()
    if args.worker:worker(args,json.loads((Path(args.output)/'protocol.json').read_text()))
    else:print(json.dumps(driver(args,json.loads(args.config.read_text())),indent=2))
