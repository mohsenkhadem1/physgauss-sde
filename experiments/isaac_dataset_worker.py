"""One immutable, moving-camera Isaac dataset episode on a selected GPU."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--episode',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--physics-hz',type=int,help='Separate preflight/refinement override; recorded explicitly')
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    case=next(c for c in plan['episodes'] if c['episode_id']==args.episode)
    hz=args.physics_hz or case['physics_hz'];fps=case['fps']
    if hz<fps or hz%fps:parser.error('physics rate must be a multiple of capture rate')
    if args.output.exists():parser.error('Use a fresh output directory')
    root=args.output;root.mkdir(parents=True)
    for folder in ['truth/rgb','truth/depth','truth/masks','observations/rgb','observations/depth','observations/masks','provenance']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    names=['isaac_dataset_worker.py','isaac_dataset.py','boundary_block.py']
    sources={n:Path(__file__).with_name(n).read_bytes() for n in names}
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in sources.items()}
    for n,b in sources.items():(root/'provenance'/n).write_bytes(b)
    (root/'truth/recipe.json').write_text(json.dumps(case,indent=2)+'\n')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':args.gpu,'physics_gpu':args.gpu,
                       'multi_gpu':False,'disable_viewport_updates':True,'fast_shutdown':True,
                       'width':640,'height':480})
    exit_code=0
    try:
        import carb
        import numpy as np
        from PIL import Image
        from pxr import Gf,PhysxSchema,UsdGeom,UsdLux,UsdPhysics,UsdShade,Vt
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import DeformablePrim,RigidPrim
        import omni.replicator.core as rep
        from isaac_dataset import geometry,appearance,forces_at,boundary_shift,camera_pose
        from boundary_block import lumped_masses
        settings=carb.settings.get_settings()
        settings.set_bool('/persistent/physics/enableDeformableBeta',True)
        settings.set_bool('/physics/updateToUsd',True)
        settings.set_bool('/physics/updateVelocitiesToUsd',True)
        world=World(stage_units_in_meters=1.,physics_dt=1/hz,rendering_dt=1/hz,
                    backend='warp',device=f'cuda:{args.gpu}')
        stage=world.stage;UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        scene=UsdPhysics.Scene.Get(stage,'/physicsScene');scene.CreateGravityMagnitudeAttr(0.)
        api=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        api.CreateEnableGPUDynamicsAttr(True);api.CreateBroadphaseTypeAttr('GPU')
        points,tets,faces,spec=geometry(case);material=case['material']
        colors,gaussians=appearance(points,faces)
        (root/'truth/boundaries.json').write_text(json.dumps(spec,indent=2)+'\n')
        np.savez_compressed(root/'reference.npz',rest_points_m=points,tetrahedra=tets,
                            surface_faces=faces,vertex_colors=colors,**gaussians)
        mesh=UsdGeom.TetMesh.Define(stage,'/World/Block')
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(tets))
        mesh.GetSurfaceFaceVertexIndicesAttr().Set(Vt.Vec3iArray.FromNumpy(faces))
        mesh.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.zeros_like(points,dtype=np.float32)))
        prim=mesh.GetPrim()
        for schema in ['OmniPhysicsDeformableBodyAPI','OmniPhysicsVolumeDeformableSimAPI','PhysxBaseDeformableBodyAPI']:
            if not prim.ApplyAPI(schema):raise RuntimeError('Missing schema '+schema)
        prim.GetAttribute('omniphysics:restShapePoints').Set(mesh.GetPointsAttr().Get())
        prim.GetAttribute('omniphysics:restTetVtxIndices').Set(mesh.GetTetVertexIndicesAttr().Get())
        for key,value in {'disableGravity':True,'solverPositionIterationCount':32,'sleepThreshold':0.,
                          'settlingThreshold':0.,'settlingDamping':0.,
                          'linearDamping':material['linear_damping_s_inverse']}.items():
            prim.GetAttribute('physxDeformableBody:'+key).Set(value)
        UsdPhysics.CollisionAPI.Apply(prim)
        contact=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact.CreateContactOffsetAttr(.0001);contact.CreateRestOffsetAttr(0.)
        mat=UsdShade.Material.Define(stage,'/World/Material')
        for schema in ['OmniPhysicsBaseMaterialAPI','OmniPhysicsDeformableMaterialAPI','PhysxDeformableMaterialAPI']:
            if not mat.GetPrim().ApplyAPI(schema):raise RuntimeError('Missing schema '+schema)
        for key,value in {'omniphysics:density':material['density_kg_m3'],
                          'omniphysics:youngsModulus':material['youngs_modulus_Pa'],
                          'omniphysics:poissonsRatio':material['poissons_ratio'],
                          'physxDeformableMaterial:elasticityDamping':material['elasticity_damping']}.items():
            mat.GetPrim().GetAttribute(key).Set(value)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,materialPurpose='physics')
        mesh.CreatePurposeAttr('guide')
        display=UsdGeom.Mesh.Define(stage,'/World/Surface')
        display.CreatePointsAttr(mesh.GetPointsAttr().Get());display.CreateFaceVertexCountsAttr([3]*len(faces))
        display.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
        display.CreateSubdivisionSchemeAttr('none');display.CreateDoubleSidedAttr(True)
        display.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
        display.GetDisplayColorPrimvar().SetInterpolation('vertex')
        handle_paths=[]
        for i,(node,patch) in enumerate(zip(spec['load_nodes'],spec['load_patches'])):
            path=f'/World/Load_{i}';handle_paths.append(path)
            handle=UsdGeom.Sphere.Define(stage,path);handle.CreateRadiusAttr(.003)
            handle.AddTranslateOp().Set(Gf.Vec3d(*points[node]));handle.CreateVisibilityAttr('invisible')
            UsdPhysics.RigidBodyAPI.Apply(handle.GetPrim())
            mass=UsdPhysics.MassAPI.Apply(handle.GetPrim());mass.CreateMassAttr(spec['handle_mass_kg_each'])
            offsets=points[patch]-points[node]
            inertia=spec['handle_mass_kg_each']*np.mean(np.sum(offsets**2,axis=1)[:,None]-offsets**2,axis=0)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*np.maximum(inertia,.4*spec['handle_mass_kg_each']*.003**2)))
            ha=PhysxSchema.PhysxRigidBodyAPI.Apply(handle.GetPrim())
            ha.CreateDisableGravityAttr(True);ha.CreateSleepThresholdAttr(0.)
            ha.CreateStabilizationThresholdAttr(0.);ha.CreateLinearDampingAttr(0.);ha.CreateAngularDampingAttr(0.)
            attachment=stage.DefinePrim(f'/World/Attachment_{i}','OmniPhysicsVtxXformAttachment')
            attachment.GetRelationship('omniphysics:src0').SetTargets([mesh.GetPath()])
            attachment.GetRelationship('omniphysics:src1').SetTargets([handle.GetPath()])
            attachment.GetAttribute('omniphysics:vtxIndicesSrc0').Set(patch)
            attachment.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray.FromNumpy(offsets.astype(np.float32)))
        UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(1000.)
        camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera_op=camera.AddTransformOp()
        cam=spec['camera'];camera.CreateFocalLengthAttr(cam['focal_length_mm'])
        camera.CreateHorizontalApertureAttr(cam['horizontal_aperture_mm'])
        camera.CreateVerticalApertureAttr(cam['vertical_aperture_mm'])
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01,10.))
        camera_op.Set(Gf.Matrix4d(*camera_pose(spec,0).ravel().tolist()))
        product=rep.create.render_product(str(camera.GetPath()),tuple(cam['resolution']))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
        depth_annotator=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane');depth_annotator.attach([product])
        stage.GetRootLayer().Export(str(root/'scene.usda'))
        world.reset();body=DeformablePrim('/World/Block',deformable_type='volume')
        handles=[RigidPrim(p) for p in handle_paths]
        fixed=np.asarray(spec['fixture_nodes']);enabled=np.zeros((1,len(points),1),dtype=bool);enabled[0,fixed]=True
        body.set_nodal_kinematic_position_targets(points[None].astype(np.float32),enabled=enabled)
        _,actual_enabled=body.get_nodal_kinematic_position_targets()
        if not np.array_equal(np.flatnonzero(actual_enabled.numpy()[0,:,0]),fixed):
            raise RuntimeError('Kinematic boundary flags differ from requested node set')
        for _ in range(8):world.render()
        initial=body.get_nodal_positions()[0].numpy()[0]
        if not np.allclose(initial,points,atol=1e-6):raise RuntimeError('Initial nodal ordering/state mismatch')
        start_clock=world.current_time;rest_det=np.linalg.det(points[tets[:,1:]]-points[tets[:,:1]])
        times=[];measured=[];positions=[];velocities=[];poses=[];frame_forces=[];targets=[];handle_positions=[]
        step_times=[];commands=[];application_positions=[];net_moments=[];transmitted=[]
        min_volume=1.;max_constraint_error=0.;minimum_mask_pixels=640*480
        fx=640*cam['focal_length_mm']/cam['horizontal_aperture_mm'];fy=480*cam['focal_length_mm']/cam['vertical_aperture_mm']
        k=np.array([[fx,0,320],[0,fy,240],[0,0,1]],dtype=float)
        for step in range(round(case['seconds']*hz)+1):
            t=step/hz;target=points.copy();target[fixed]+=boundary_shift(spec,t)
            if step:
                force=forces_at(spec,t-.5/hz)
                body.set_nodal_kinematic_position_targets(target[None].astype(np.float32),enabled=enabled)
                before_v=np.stack([h.get_velocities()[0].numpy()[0] for h in handles])
                at=np.stack([h.get_world_poses()[0].numpy()[0] for h in handles])
                for h,f in zip(handles,force):h.apply_forces(f[None])
                world.step(render=False,update_fabric=True)
                if abs((world.current_time-start_clock)-t)>1e-7*max(1,t):raise RuntimeError('Simulation/force clock mismatch')
                after_v=np.stack([h.get_velocities()[0].numpy()[0] for h in handles])
                load=force-spec['handle_mass_kg_each']*(after_v-before_v)*hz
                step_times.append(t-.5/hz);commands.append(force.copy());application_positions.append(at.copy())
                net_moments.append(np.cross(at-np.asarray(cam['center_m']),force).sum(0))
            else:force=np.zeros((2,3),np.float32);load=force.copy()
            q=body.get_nodal_positions()[0].numpy()[0].copy();v=body.get_nodal_velocities().numpy()[0].copy()
            if not np.isfinite(q).all() or not np.isfinite(v).all():raise RuntimeError('Nonfinite FEM state')
            ratio=np.linalg.det(q[tets[:,1:]]-q[tets[:,:1]])/rest_det
            min_volume=min(min_volume,float(ratio.min()))
            error=float(np.linalg.norm(q[fixed]-target[fixed],axis=1).max());max_constraint_error=max(max_constraint_error,error)
            if ratio.min()<.1 or error>2e-4:raise RuntimeError('Rejected: compressed/inverted element or constraint drift')
            if step%(hz//fps):continue
            index=len(times);pose=camera_pose(spec,t)
            camera_op.Set(Gf.Matrix4d(*pose.ravel().tolist()))
            display.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(q))
            for _ in range(3):world.render()
            image=np.asarray(rgb.get_data());depth=np.asarray(depth_annotator.get_data()).squeeze()
            if image.shape[:2]!=(480,640) or depth.shape!=(480,640):raise RuntimeError('Invalid RGB/depth capture shape')
            mask=np.isfinite(depth)&(depth>.01)&(depth<10.)
            if mask.sum()<1000:raise RuntimeError('Object missing or too small in camera')
            minimum_mask_pixels=min(minimum_mask_pixels,int(mask.sum()))
            depth=np.where(mask,depth,0).astype(np.float32)
            name=f'{index:05d}'
            Image.fromarray(image[:,:,:3]).save(root/'truth/rgb'/f'{name}.png')
            Image.fromarray(mask.astype(np.uint8)*255).save(root/'truth/masks'/f'{name}.png')
            np.savez_compressed(root/'truth/depth'/f'{name}.npz',depth_m=depth)
            times.append(t);measured.append(world.current_time-start_clock);positions.append(q);velocities.append(v)
            poses.append(pose);frame_forces.append(force.copy());targets.append(target[fixed].copy());transmitted.append(load.copy())
            handle_positions.append(np.stack([h.get_world_poses()[0].numpy()[0] for h in handles]))
            print(json.dumps({'episode':case['episode_id'],'frame':index,'time_s':t,'min_volume':min_volume,
                              'constraint_error_m':error,'mask_pixels':int(mask.sum())}),flush=True)
        positions=np.asarray(positions);poses=np.asarray(poses);times=np.asarray(times)
        np.savez_compressed(root/'truth/trajectory.npz',times_s=times,simulated_times_s=measured,
                            positions_m=positions,velocities_m_s=velocities,boundary_target_positions_m=targets,
                            commanded_handle_forces_N=frame_forces,handle_positions_m=handle_positions,
                            inferred_transmitted_forces_N=transmitted,force_step_midpoints_s=step_times,
                            force_step_commands_N=commands,force_step_application_positions_m=application_positions,
                            force_step_net_moments_Nm=net_moments,
                            gaussian_means_m=np.einsum('ni,fnij->fnj',gaussians['barycentric_weights'],positions[:,gaussians['face_vertex_ids']]),
                            lumped_node_mass_kg=lumped_masses(points,tets,material['density_kg_m3']))
        indices=np.asarray(case['sparse_frames'],dtype=int)
        np.savez_compressed(root/'truth/cameras.npz',K=k,camera_to_world_usd_row=poses,times_s=times,frame_ids=np.arange(len(times)))
        np.savez_compressed(root/'observations/cameras.npz',K=k,camera_to_world_usd_row=poses[indices],times_s=times[indices],frame_ids=indices)
        for index in indices:
            for folder,suffix in [('rgb','.png'),('masks','.png'),('depth','.npz')]:
                shutil.copy2(root/'truth'/folder/f'{index:05d}{suffix}',root/'observations'/folder/f'{index:05d}{suffix}')
        (root/'observations/input_protocol.json').write_text(json.dumps({'observed_frame_ids':indices.tolist(),
            'image_size':[640,480],'axes':'USD camera -Z forward, +Y up; pixels +v down',
            'depth':'Positive camera-axis depth in metres; background zero',
            'inputs':'Only reference.npz and observations/ are model inputs; material/boundary labels and withheld frames are truth'},indent=2)+'\n')
        if any(Path(__file__).with_name(n).read_bytes()!=b for n,b in sources.items()):raise RuntimeError('Source changed during generation')
        result={'completed':True,'episode_id':case['episode_id'],'split':case['split'],'group_id':case['group_id'],
                'engine':'Isaac Sim '+importlib.metadata.version('isaacsim')+' / PhysX GPU',
                'recipe_sha256':case['recipe_sha256'],'source_sha256':hashes,'physics_hz':hz,'fps':fps,
                'frames':len(times),'observed_frames':len(indices),'minimum_tet_volume_ratio':min_volume,
                'maximum_constraint_error_m':max_constraint_error,'minimum_mask_pixels':minimum_mask_pixels,
                'material':material,'boundary_type':case['boundary_type'],'force_type':case['force_type'],
                'maximum_displacement_m':float(np.linalg.norm(positions-points,axis=-1).max()),
                'camera_arc_deg':cam['arc_deg'],'gravity_m_s2':0.,'numerical_convergence_established':False}
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    except BaseException as error:
        exit_code=1
        failure={'completed':False,'episode_id':args.episode,'time_s':locals().get('t'),
                 'exception':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
                 'source_sha256':hashes,'minimum_tet_volume_ratio':locals().get('min_volume')}
        (root/'failure.json').write_text(json.dumps(failure,indent=2)+'\n');traceback.print_exc()
        if 'q' in locals():np.savez_compressed(root/'truth/failure-state.npz',positions_m=q,
            saved_positions_m=positions,saved_times_s=times)
        raise
    finally:app.close(exit_code=exit_code)


if __name__=='__main__':main()
