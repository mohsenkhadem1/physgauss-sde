"""Isaac Sim 6.2 GPU FEM block, random fixtures and force-driven loading.

Run using an Isaac Sim Python environment. No dependencies are installed or
modified by this script. External forces act on small dynamic load handles
attached to selected surface nodes, rather than prescribing node motion.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=31001)
    parser.add_argument('--seconds', type=float, default=3.0)
    parser.add_argument('--hz', type=int, default=240)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--fixture-nodes', type=int, default=1)
    parser.add_argument('--load-patch-nodes', type=int, default=1)
    parser.add_argument('--force-scale', type=float, default=.25)
    parser.add_argument('--youngs-modulus', type=float, default=20000.)
    parser.add_argument('--cells', type=int, nargs=3, default=[12, 6, 4])
    parser.add_argument('--no-rgb', action='store_true')
    parser.add_argument('--keep-open', action='store_true', help='Pause the completed scene for GUI inspection')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a fresh output directory')
    if args.keep_open and args.headless: parser.error('--keep-open requires the GUI')
    if args.seconds <= 0 or args.hz <= 0 or args.fps <= 0 or args.hz % args.fps or args.youngs_modulus <= 0:
        parser.error('Positive duration/modulus and hz divisible by fps required')
    args.output.mkdir(parents=True)
    (args.output/'truth').mkdir(); (args.output/'observations').mkdir()
    provenance=args.output/'provenance';provenance.mkdir()
    sources={name:Path(__file__).with_name(name).read_bytes()
             for name in ['isaac_boundary_block.py','boundary_block.py']}
    source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in sources.items()}
    for name,data in sources.items(): (provenance/name).write_bytes(data)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless, 'width': 960, 'height': 720,
                         'active_gpu': args.gpu, 'physics_gpu': args.gpu, 'multi_gpu': False,
                         'disable_viewport_updates': args.headless, 'fast_shutdown': True})
    exit_code=0
    try:
        import carb
        import numpy as np
        from PIL import Image
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import DeformablePrim, RigidPrim
        from isaacsim.core.utils.viewports import set_camera_view
        from boundary_block import block_mesh, episode, forces_at, lumped_masses, reference_gaussians
        settings = carb.settings.get_settings()
        settings.set_bool('/persistent/physics/enableDeformableBeta', True)
        settings.set_bool('/physics/updateToUsd', True)
        settings.set_bool('/physics/updateVelocitiesToUsd', True)
        world = World(stage_units_in_meters=1., physics_dt=1/args.hz, rendering_dt=1/args.hz,
                      backend='warp', device=f'cuda:{args.gpu}')
        stage = world.stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Get(stage, '/physicsScene')
        scene.CreateGravityMagnitudeAttr(0.)
        physics = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physics.CreateEnableGPUDynamicsAttr(True); physics.CreateBroadphaseTypeAttr('GPU')
        points, tetrahedra, faces = block_mesh(cells=args.cells)
        spec = episode(points, faces, seed=args.seed, fixture_nodes=args.fixture_nodes,
                       seconds=args.seconds, force_scale=args.force_scale,load_patch_nodes=args.load_patch_nodes)
        masses = lumped_masses(points, tetrahedra, density=500.)
        mesh = UsdGeom.TetMesh.Define(stage, '/World/Block')
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(tetrahedra))
        mesh.GetSurfaceFaceVertexIndicesAttr().Set(Vt.Vec3iArray.FromNumpy(faces))
        mesh.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.zeros_like(points, dtype=np.float32)))
        prim = mesh.GetPrim()
        for api in ['OmniPhysicsDeformableBodyAPI', 'OmniPhysicsVolumeDeformableSimAPI', 'PhysxBaseDeformableBodyAPI']:
            if not prim.ApplyAPI(api): raise RuntimeError(f'Missing schema {api}')
        prim.GetAttribute('omniphysics:restShapePoints').Set(mesh.GetPointsAttr().Get())
        prim.GetAttribute('omniphysics:restTetVtxIndices').Set(mesh.GetTetVertexIndicesAttr().Get())
        prim.GetAttribute('physxDeformableBody:disableGravity').Set(True)
        prim.GetAttribute('physxDeformableBody:solverPositionIterationCount').Set(32)
        # Millimetre-scale motion must not be suppressed by the default 5cm/s
        # sleep threshold or 10/s settling damping. Use explicit physical drag.
        for name,value in {'sleepThreshold':0.,'settlingThreshold':0.,
                           'settlingDamping':0.,'linearDamping':.5}.items():
            prim.GetAttribute('physxDeformableBody:'+name).Set(value)
        UsdPhysics.CollisionAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(.0001)
        material = UsdShade.Material.Define(stage, '/World/BlockMaterial')
        for api in ['OmniPhysicsBaseMaterialAPI', 'OmniPhysicsDeformableMaterialAPI', 'PhysxDeformableMaterialAPI']:
            if not material.GetPrim().ApplyAPI(api): raise RuntimeError(f'Missing material schema {api}')
        for key, value in {'omniphysics:density': 500., 'omniphysics:youngsModulus': args.youngs_modulus,
                           'omniphysics:poissonsRatio': .35, 'physxDeformableMaterial:elasticityDamping': .02}.items():
            material.GetPrim().GetAttribute(key).Set(value)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
        mesh.CreatePurposeAttr('guide')
        display = UsdGeom.Mesh.Define(stage, '/World/BlockSurface')
        display.CreatePointsAttr(mesh.GetPointsAttr().Get())
        display.CreateFaceVertexCountsAttr([3]*len(faces))
        display.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
        display.CreateSubdivisionSchemeAttr('none'); display.CreateDoubleSidedAttr(True)
        colors = .2+.6*(points-points.min(0))/(points.max(0)-points.min(0))
        display.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
        display.GetDisplayColorPrimvar().SetInterpolation('vertex')
        anchor = UsdGeom.Xform.Define(stage, '/World/Fixture')
        UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim()).CreateKinematicEnabledAttr(True)

        def attach(name, ids, target, local):
            pin = stage.DefinePrim('/World/'+name, 'OmniPhysicsVtxXformAttachment')
            pin.GetRelationship('omniphysics:src0').SetTargets([mesh.GetPath()])
            pin.GetRelationship('omniphysics:src1').SetTargets([target])
            pin.GetAttribute('omniphysics:vtxIndicesSrc0').Set(list(map(int, ids)))
            pin.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray.FromNumpy(np.asarray(local, np.float32)))
        attach('FixtureAttachment', spec['fixture_nodes'], anchor.GetPath(), points[spec['fixture_nodes']])
        handle_mass = .001
        handle_paths = []
        for i, node in enumerate(spec['load_nodes']):
            path = f'/World/Load_{i}'; handle_paths.append(path)
            handle = UsdGeom.Sphere.Define(stage, path); handle.CreateRadiusAttr(.003)
            handle.AddTranslateOp().Set(Gf.Vec3d(*points[node]))
            handle.CreateVisibilityAttr('invisible')
            UsdPhysics.RigidBodyAPI.Apply(handle.GetPrim())
            mass = UsdPhysics.MassAPI.Apply(handle.GetPrim()); mass.CreateMassAttr(handle_mass)
            patch=spec['load_patches'][i];offsets=points[patch]-points[node]
            inertia=handle_mass*np.mean(np.sum(offsets**2,axis=1)[:,None]-offsets**2,axis=0)
            inertia=np.maximum(inertia,.4*handle_mass*.003**2)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
            handle_physics=PhysxSchema.PhysxRigidBodyAPI.Apply(handle.GetPrim())
            handle_physics.CreateDisableGravityAttr(True)
            handle_physics.CreateSleepThresholdAttr(0.)
            handle_physics.CreateStabilizationThresholdAttr(0.)
            handle_physics.CreateLinearDampingAttr(0.)
            handle_physics.CreateAngularDampingAttr(0.)
            # No collider: this handle transfers load only through its attachment.
            attach(f'LoadAttachment_{i}', patch, handle.GetPath(), offsets)
        UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1500.)
        set_camera_view(eye=[.42, -.55, .48], target=[0, 0, .25])
        camera = UsdGeom.Camera.Define(stage, '/World/ObservationCamera')
        eye = Gf.Vec3d(.42, -.55, .48); target = Gf.Vec3d(0, 0, .25)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
        camera.CreateFocalLengthAttr(35.); camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(27.); camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.))
        rgb = None
        if not args.no_rgb:
            import omni.replicator.core as rep
            product = rep.create.render_product(str(camera.GetPath()), (640, 480))
            rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product])
        gaussians=reference_gaussians(points, faces)
        np.savez_compressed(args.output/'reference.npz', rest_points_m=points, tetrahedra=tetrahedra,
                            surface_faces=faces, **gaussians)
        (args.output/'truth/boundaries.json').write_text(json.dumps(spec, indent=2)+'\n')
        camera_meta = {'resolution': [640, 480], 'K': [[640*35/36, 0, 320], [0, 480*35/27, 240], [0, 0, 1]],
                       'camera_to_world_usd_row_matrix': list(map(list, camera.GetLocalTransformation())),
                       'axes': 'USD camera looks along local -Z, local +Y up; pixel +v down',
                       'observation_schedule': 'all saved frames are dense development observations; subsample for future sparse-view tests'}
        (args.output/'observations/camera.json').write_text(json.dumps(camera_meta, indent=2)+'\n')
        stage.GetRootLayer().Export(str(args.output/'scene.usda'))
        world.reset()
        body = DeformablePrim('/World/Block', deformable_type='volume')
        handles = [RigidPrim(path) for path in handle_paths]
        # Warm the renderer while physics is paused; t=0 remains the rest state.
        for _ in range(8): world.render()
        initial = body.get_nodal_positions()[0].numpy()[0].copy()
        if not np.allclose(initial, points, atol=1e-6): raise RuntimeError('Simulation node ordering/rest mismatch')
        times=[]; positions=[]; velocities=[]; commands=[]; transmitted=[]; handle_positions=[]; step_times=[]; step_forces=[]; force_times=[]; measured_times=[]
        start_clock=world.current_time
        maximum_pin_error = 0.; minimum_volume_ratio = 1.
        for step in range(round(args.seconds*args.hz)+1):
            t = step/args.hz
            if step:
                force = forces_at(spec, t-.5/args.hz)
                before = np.stack([h.get_velocities()[0].numpy()[0] for h in handles])
                for h, f in zip(handles, force): h.apply_forces(f[None])
                # A Kit app update may advance a variable number of substeps.
                # Direct physics stepping gives each force exactly one dt.
                world.step(render=False, update_fabric=True)
                if abs((world.current_time-start_clock)-t)>1e-7*max(1,t):
                    raise RuntimeError('Isaac simulation clock differs from the declared force clock')
                after = np.stack([h.get_velocities()[0].numpy()[0] for h in handles])
                load = force-handle_mass*(after-before)*args.hz
                step_times.append(t-.5/args.hz); step_forces.append(force.copy())
            else:
                force=np.zeros((len(handles), 3), np.float32); load=force.copy()
            q = body.get_nodal_positions()[0].numpy()[0].copy()
            v = body.get_nodal_velocities().numpy()[0].copy()
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)): raise RuntimeError('Nonfinite FEM state')
            pin_error = float(np.linalg.norm(q[spec['fixture_nodes']]-points[spec['fixture_nodes']], axis=1).max())
            maximum_pin_error=max(maximum_pin_error, pin_error)
            ratio=np.linalg.det(q[tetrahedra[:, 1:]]-q[tetrahedra[:, :1]])/np.linalg.det(points[tetrahedra[:, 1:]]-points[tetrahedra[:, :1]])
            minimum_volume_ratio=min(minimum_volume_ratio, float(ratio.min()))
            if pin_error > .002 or ratio.min() <= 0: raise RuntimeError('Fixture drift or inverted tetrahedron')
            if step % (args.hz//args.fps): continue
            times.append(t); positions.append(q); velocities.append(v); commands.append(force.copy()); transmitted.append(load.copy())
            measured_times.append(world.current_time-start_clock)
            force_times.append(t-.5/args.hz if step else 0.)
            handle_positions.append(np.stack([h.get_world_poses()[0].numpy()[0] for h in handles]))
            display.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(q))
            if rgb is not None:
                # Explicit paused renders: image geometry and saved q have the same timestamp.
                for _ in range(3): world.render()
                frame = np.asarray(rgb.get_data())
                if frame.shape[:2] != (480, 640): raise RuntimeError('Empty RGB output')
                Image.fromarray(frame[:, :, :3]).save(args.output/'observations'/f'rgb-{len(times)-1:04d}.png')
            else: world.render()
            print(json.dumps({'frame':len(times)-1,'time_s':t,'maximum_displacement_m':float(np.linalg.norm(q-points,axis=1).max()),'pin_error_m':pin_error}),flush=True)
        positions=np.asarray(positions)
        np.savez_compressed(args.output/'truth/trajectory.npz', times_s=times, positions_m=positions,
                            simulated_times_s=measured_times,
                            velocities_m_s=velocities, commanded_handle_forces_N=commands,
                            inferred_transmitted_forces_N=transmitted, handle_positions_m=handle_positions,
                            saved_force_midpoints_s=force_times,
                            force_step_midpoints_s=step_times, force_step_commands_N=step_forces,
                            gaussian_means_m=np.einsum('ni,fnij->fnj',gaussians['barycentric_weights'],positions[:,gaussians['face_vertex_ids']]),
                            lumped_node_mass_kg=masses)
        # Remove the best rigid transform to distinguish elastic motion from rotation about a point pin.
        residual=[]
        reference=points-points.mean(0)
        for q in positions:
            moving=q-q.mean(0); u,_,vt=np.linalg.svd(reference.T@moving); rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
            residual.append(float(np.sqrt(np.mean((moving-reference@rotation)**2))))
        summary={'engine':'Isaac Sim '+importlib.metadata.version('isaacsim')+' / PhysX GPU FEM',
                 'seed':args.seed,'seconds':args.seconds,'physics_hz':args.hz,'observation_fps':args.fps,
                 'force_scale':args.force_scale,'cells':args.cells,
                 'nodes':len(points),'tetrahedra':len(tetrahedra),'fixture_nodes':spec['fixture_nodes'],
                 'load_nodes':spec['load_nodes'],'youngs_modulus_Pa':args.youngs_modulus,'poissons_ratio':.35,
                 'load_patches':spec['load_patches'],'load_patch_nodes':args.load_patch_nodes,
                 'density_kg_m3':500.,'block_mass_kg':float(masses.sum()),'handle_mass_kg_each':handle_mass,
                 'gravity_m_s2':0.,'maximum_pin_error_m':maximum_pin_error,'minimum_tet_volume_ratio':minimum_volume_ratio,
                 'deformable_sleep_threshold_m_s':0.,'settling_damping_s_inverse':0.,'linear_damping_s_inverse':.5,
                 'physics_clock':'one explicit physics step per force command; actual clock asserted every step',
                 'maximum_displacement_m':float(np.linalg.norm(positions-points,axis=-1).max()),
                 'maximum_elastic_coordinate_rms_m':max(residual),'completed':True,
                 'loading':'Newton-valued world forces on dynamic 1g surface-node handles; no prescribed load displacement',
                 'transmitted_force_estimate':'command minus handle_mass * finite-difference handle acceleration; discrete estimate, not a contact sensor',
                 'qualification':'Development simulator only; no boundary inference, calibration or continuum convergence claim',
                 'python':sys.executable,'gpu':args.gpu,'source_sha256':source_hashes}
        if any(Path(__file__).with_name(name).read_bytes()!=data for name,data in sources.items()):
            raise RuntimeError('Source changed during the run; preserve as development only')
        if args.force_scale and max(residual) < 1e-5: raise RuntimeError('No measurable elastic deformation')
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
        if args.keep_open:
            world.pause()
            while app.is_running(): app.update()
    except BaseException as error:
        exit_code=1
        failure={'completed':False,'exception':type(error).__name__,'message':str(error),
                 'traceback':traceback.format_exc(),'source_sha256':source_hashes,
                 'time_s':locals().get('t'),'minimum_tet_volume_ratio':locals().get('minimum_volume_ratio')}
        (args.output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        traceback.print_exc()
        if 'q' in locals():
            np.savez_compressed(args.output/'truth/failure-state.npz',positions_m=q,velocities_m_s=v,
                                saved_positions_m=positions,saved_times_s=times)
        raise
    finally:
        app.close(exit_code=exit_code)


if __name__ == '__main__': main()
