"""Independent Isaac episodes batched in one World/GPU, with local saved frames.

The vendor deformable tensor API requires a string pattern, not list[str].
Bodies are matched by documented tensor prim-path metadata and initial-state
readback. Coincident zero-offset bodies disable external collisions; one shared
camera renders one visible surface at a time. Translated multi-body preflights
failed equivalence and are preserved under runs/operator-dev; they are excluded
from training. Saved geometry/cameras stay in episode reference coordinates.

CLI output is a fresh batch directory containing episodes/<id>/ plus the shared
batch-scene.usda and batch-execution.json. A sensor failure is local to a case;
a physics/world failure records an explicit cohort-abort for every unfinished
case. No failed or partially generated path is silently overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np


def grid_offsets(count: int, spacing_m: float = 4.) -> np.ndarray:
    if not 1 <= count <= 8 or spacing_m < 4.:
        raise ValueError('Use 1-8 environments and spacing >=4m')
    return spacing_m*np.array([[i % 2, (i//2) % 2, i//4] for i in range(count)], dtype=float)


def translated_camera(local_pose: np.ndarray, offset: np.ndarray) -> np.ndarray:
    pose = np.array(local_pose, copy=True)
    pose[3, :3] += offset
    return pose


def match_rows(actual: np.ndarray, expected: np.ndarray, tolerance: float = 1e-6):
    """Return actual->case and case->actual permutations, validating all nodes."""
    if actual.shape != expected.shape or actual.ndim not in (2, 3):
        raise ValueError('Actual and expected tensor batches have incompatible shapes')
    axes = tuple(range(2, actual.ndim+1))
    distance = np.mean((actual[:, None]-expected[None])**2, axis=axes)
    actual_to_case = distance.argmin(1)
    if len(set(actual_to_case.tolist())) != len(expected):
        raise ValueError('Tensor rows do not uniquely match the declared cases')
    error = np.max(np.abs(actual-expected[actual_to_case]))
    if not np.isfinite(error) or error > tolerance:
        raise ValueError(f'Initial tensor values/order differ from scene: max error {error}')
    return actual_to_case, np.argsort(actual_to_case)



def match_prim_paths(actual_paths, expected_paths):
    """Map documented tensor prim paths to case order without geometric guesses."""
    actual_paths = list(map(str, actual_paths))
    expected_paths = list(map(str, expected_paths))
    if len(actual_paths) != len(expected_paths) or set(actual_paths) != set(expected_paths):
        raise ValueError(f'Tensor prim paths differ: actual={actual_paths}, expected={expected_paths}')
    actual_to_case = np.array([expected_paths.index(p) for p in actual_paths], dtype=int)
    return actual_to_case, np.argsort(actual_to_case)


def write_sensor(root, name, image, depth, mask):
    from PIL import Image
    Image.fromarray(image[:, :, :3]).save(root/'truth/rgb'/f'{name}.png', compress_level=3)
    Image.fromarray(mask.astype(np.uint8)*255).save(root/'truth/masks'/f'{name}.png', compress_level=3)
    np.savez_compressed(root/'truth/depth'/f'{name}.npz', depth_m=depth)


def create_environment(stage, case, root, offset, index, rep):
    """Build one scene with the same material/constraints/handles as single-run."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
    from isaac_dataset import geometry, appearance, camera_pose
    points, tets, faces, spec = geometry(case)
    material = case['material']; colors, gaussians = appearance(points, faces)
    world_points = (points+offset).astype(np.float32)
    prefix = f'/World/Env_{index:02d}'
    environment_xform = UsdGeom.Xform.Define(stage, prefix)
    environment_xform.AddTranslateOp().Set(Gf.Vec3d(*offset))
    (root/'truth/boundaries.json').write_text(json.dumps(spec, indent=2)+'\n')
    np.savez_compressed(root/'reference.npz', rest_points_m=points, tetrahedra=tets,
                        surface_faces=faces, vertex_colors=colors, **gaussians)
    mesh = UsdGeom.TetMesh.Define(stage, prefix+'/Block')
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    # All children share one translated environment; rest meshes, handles,
    # display vertices and camera transforms remain in object-local coordinates.
    mesh.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(tets))
    mesh.GetSurfaceFaceVertexIndicesAttr().Set(Vt.Vec3iArray.FromNumpy(faces))
    mesh.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.zeros_like(world_points)))
    prim = mesh.GetPrim()
    for schema in ['OmniPhysicsDeformableBodyAPI', 'OmniPhysicsVolumeDeformableSimAPI', 'PhysxBaseDeformableBodyAPI']:
        if not prim.ApplyAPI(schema):
            raise RuntimeError('Missing schema '+schema)
    prim.GetAttribute('omniphysics:restShapePoints').Set(mesh.GetPointsAttr().Get())
    prim.GetAttribute('omniphysics:restTetVtxIndices').Set(mesh.GetTetVertexIndicesAttr().Get())
    for key, value in {'disableGravity': True, 'solverPositionIterationCount': 32,
                       'sleepThreshold': 0., 'settlingThreshold': 0., 'settlingDamping': 0.,
                       'linearDamping': material['linear_damping_s_inverse']}.items():
        prim.GetAttribute('physxDeformableBody:'+key).Set(value)
    UsdPhysics.CollisionAPI.Apply(prim)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    contact.CreateContactOffsetAttr(.0001); contact.CreateRestOffsetAttr(0.)
    mat = UsdShade.Material.Define(stage, prefix+'/Material')
    for schema in ['OmniPhysicsBaseMaterialAPI', 'OmniPhysicsDeformableMaterialAPI', 'PhysxDeformableMaterialAPI']:
        if not mat.GetPrim().ApplyAPI(schema):
            raise RuntimeError('Missing schema '+schema)
    for key, value in {'omniphysics:density': material['density_kg_m3'],
                       'omniphysics:youngsModulus': material['youngs_modulus_Pa'],
                       'omniphysics:poissonsRatio': material['poissons_ratio'],
                       'physxDeformableMaterial:elasticityDamping': material['elasticity_damping']}.items():
        mat.GetPrim().GetAttribute(key).Set(value)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose='physics')
    mesh.CreatePurposeAttr('guide')
    display = UsdGeom.Mesh.Define(stage, prefix+'/Surface')
    display.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32))); display.CreateFaceVertexCountsAttr([3]*len(faces))
    display.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    display.CreateSubdivisionSchemeAttr('none'); display.CreateDoubleSidedAttr(True)
    display.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
    display.GetDisplayColorPrimvar().SetInterpolation('vertex')
    for handle_index, (node, patch) in enumerate(zip(spec['load_nodes'], spec['load_patches'])):
        path = prefix+f'/Load_{handle_index}'
        handle = UsdGeom.Sphere.Define(stage, path); handle.CreateRadiusAttr(.003)
        handle.AddTranslateOp().Set(Gf.Vec3d(*points[node].astype(float)))
        handle.CreateVisibilityAttr('invisible'); UsdPhysics.RigidBodyAPI.Apply(handle.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(handle.GetPrim()); mass.CreateMassAttr(spec['handle_mass_kg_each'])
        local_offsets = points[patch]-points[node]
        inertia = spec['handle_mass_kg_each']*np.mean(np.sum(local_offsets**2, axis=1)[:, None]-local_offsets**2, axis=0)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*np.maximum(inertia, .4*spec['handle_mass_kg_each']*.003**2)))
        ha = PhysxSchema.PhysxRigidBodyAPI.Apply(handle.GetPrim())
        ha.CreateDisableGravityAttr(True); ha.CreateSleepThresholdAttr(0.)
        ha.CreateStabilizationThresholdAttr(0.); ha.CreateLinearDampingAttr(0.); ha.CreateAngularDampingAttr(0.)
        attachment = stage.DefinePrim(prefix+f'/Attachment_{handle_index}', 'OmniPhysicsVtxXformAttachment')
        attachment.GetRelationship('omniphysics:src0').SetTargets([mesh.GetPath()])
        attachment.GetRelationship('omniphysics:src1').SetTargets([handle.GetPath()])
        attachment.GetAttribute('omniphysics:vtxIndicesSrc0').Set(patch)
        attachment.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray.FromNumpy(local_offsets.astype(np.float32)))
    cam = spec['camera']
    keys = ['times', 'measured', 'positions', 'velocities', 'poses', 'frame_forces', 'targets',
            'handle_positions', 'step_times', 'commands', 'application_positions', 'net_moments', 'transmitted']
    return dict(case=case, root=root, offset=offset, points=points, tets=tets, faces=faces,
                spec=spec, gaussians=gaussians, display=display, records={key: [] for key in keys}, active=True,
                min_volume=1., max_constraint_error=0., minimum_mask_pixels=640*480,
                rest_det=np.linalg.det(points[tets[:, 1:]]-points[tets[:, :1]]),
                k=np.array([[640*cam['focal_length_mm']/cam['horizontal_aperture_mm'], 0, 320],
                            [0, 480*cam['focal_length_mm']/cam['vertical_aperture_mm'], 240], [0, 0, 1.]], float))


def capture(env):
    image = np.asarray(env['rgb'].get_data())
    depth = np.asarray(env['depth'].get_data()).squeeze()
    if image.ndim != 3 or image.shape[:2] != (480, 640) or image.shape[2] < 3 or depth.shape != (480, 640):
        return None
    mask = np.isfinite(depth) & (depth > .01) & (depth < 10.)
    if mask.sum() < 1000:
        return None
    return image.copy(), np.where(mask, depth, 0).astype(np.float32), mask


def write_failure(env, error, hashes, time_s, phase, q=None):
    if (env['root']/'summary.json').exists() or (env['root']/'failure.json').exists():
        return
    env['active'] = False
    failure = {'completed': False, 'episode_id': env['case']['episode_id'], 'time_s': time_s,
               'phase': phase, 'exception': type(error).__name__, 'message': str(error),
               'traceback': traceback.format_exc(), 'source_sha256': hashes,
               'minimum_tet_volume_ratio': env['min_volume']}
    (env['root']/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
    records = env['records']
    np.savez_compressed(env['root']/'truth/failure-state.npz',
                        positions_m=np.asarray(q) if q is not None else np.empty((0, 3)),
                        saved_positions_m=np.asarray(records['positions']), saved_times_s=records['times'])


def save_completed(env, hashes, hz, fps, batch_count, elapsed):
    from boundary_block import lumped_masses
    case, root, spec, r = env['case'], env['root'], env['spec'], env['records']
    points, gaussians = env['points'], env['gaussians']
    positions, poses, times = np.asarray(r['positions']), np.asarray(r['poses']), np.asarray(r['times'])
    np.savez_compressed(root/'truth/trajectory.npz', times_s=times, simulated_times_s=r['measured'],
                        positions_m=positions, velocities_m_s=r['velocities'], boundary_target_positions_m=r['targets'],
                        commanded_handle_forces_N=r['frame_forces'], handle_positions_m=r['handle_positions'],
                        inferred_transmitted_forces_N=r['transmitted'], force_step_midpoints_s=r['step_times'],
                        force_step_commands_N=r['commands'], force_step_application_positions_m=r['application_positions'],
                        force_step_net_moments_Nm=r['net_moments'],
                        gaussian_means_m=np.einsum('ni,fnij->fnj', gaussians['barycentric_weights'], positions[:, gaussians['face_vertex_ids']]),
                        lumped_node_mass_kg=lumped_masses(points, env['tets'], case['material']['density_kg_m3']))
    indices = np.asarray(case['sparse_frames'], dtype=int)
    np.savez_compressed(root/'truth/cameras.npz', K=env['k'], camera_to_world_usd_row=poses,
                        times_s=times, frame_ids=np.arange(len(times)))
    np.savez_compressed(root/'observations/cameras.npz', K=env['k'], camera_to_world_usd_row=poses[indices],
                        times_s=times[indices], frame_ids=indices)
    for index in indices:
        for folder, suffix in [('rgb', '.png'), ('masks', '.png'), ('depth', '.npz')]:
            shutil.copy2(root/'truth'/folder/f'{index:05d}{suffix}', root/'observations'/folder/f'{index:05d}{suffix}')
    (root/'observations/input_protocol.json').write_text(json.dumps({
        'observed_frame_ids': indices.tolist(), 'image_size': [640, 480],
        'axes': 'USD camera -Z forward, +Y up; pixels +v down',
        'depth': 'Positive camera-axis depth in metres; background zero',
        'inputs': 'Only reference.npz and observations/ are model inputs; material/boundary labels and withheld frames are truth',
    }, indent=2)+'\n')
    result = {'completed': True, 'episode_id': case['episode_id'], 'split': case['split'], 'group_id': case['group_id'],
              'engine': 'Isaac Sim '+importlib.metadata.version('isaacsim')+' / PhysX GPU',
              'recipe_sha256': case['recipe_sha256'], 'source_sha256': hashes, 'physics_hz': hz, 'fps': fps,
              'frames': len(times), 'observed_frames': len(indices), 'minimum_tet_volume_ratio': env['min_volume'],
              'maximum_constraint_error_m': env['max_constraint_error'], 'minimum_mask_pixels': env['minimum_mask_pixels'],
              'material': case['material'], 'boundary_type': case['boundary_type'], 'force_type': case['force_type'],
              'maximum_displacement_m': float(np.linalg.norm(positions-points, axis=-1).max()),
              'camera_arc_deg': spec['camera']['arc_deg'], 'gravity_m_s2': 0.,
              'numerical_convergence_established': False,
              'batch': {'environments': batch_count, 'world_offset_m': env['offset'].tolist(),
                        'saved_coordinates': 'episode-local; world offset subtracted',
                        'camera_far_clip_m': 10., 'rendering': 'one shared camera; sequential isolated visible surfaces',
                        'inter_episode_contact_filtered': True, 'shared_process_elapsed_s': elapsed}}
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'episode': case['episode_id'], 'completed': True, 'frames': len(times)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--spacing-m', type=float, default=4.)
    parser.add_argument('--warmup-renders', type=int, default=60)
    parser.add_argument('--capture-retries', type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh batch output directory')
    if args.warmup_renders < 3 or args.capture_retries < 0:
        parser.error('Use at least3 warmup renders and nonnegative capture retries')
    offsets = np.zeros_like(grid_offsets(len(args.episodes), args.spacing_m))
    if len(set(args.episodes)) != len(args.episodes):
        parser.error('Repeated episode IDs are unsupported')
    plan = json.loads(args.plan.read_text()); by_id = {c['episode_id']: c for c in plan['episodes']}
    cases = [by_id[e] for e in args.episodes]
    for key in ['physics_hz', 'fps', 'seconds', 'cells']:
        if any(c[key] != cases[0][key] for c in cases):
            parser.error(f'Batched cases must share {key}')
    hz, fps, seconds = [cases[0][key] for key in ['physics_hz', 'fps', 'seconds']]
    if hz < fps or hz % fps:
        parser.error('physics rate must be a multiple of capture rate')
    args.output.mkdir(parents=True); (args.output/'episodes').mkdir()
    names = ['isaac_batch_dataset_worker.py', 'isaac_dataset.py', 'boundary_block.py']
    sources = {n: Path(__file__).with_name(n).read_bytes() for n in names}
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()}
    roots = []
    for case in cases:
        root = args.output/'episodes'/case['episode_id']; root.mkdir(); roots.append(root)
        for folder in ['truth/rgb', 'truth/depth', 'truth/masks', 'observations/rgb',
                       'observations/depth', 'observations/masks', 'provenance']:
            (root/folder).mkdir(parents=True)
        for name, content in sources.items():
            (root/'provenance'/name).write_bytes(content)
        (root/'truth/recipe.json').write_text(json.dumps(case, indent=2)+'\n')
    execution = {'completed': False, 'episodes': args.episodes, 'gpu': args.gpu,
                 'source_sha256': hashes, 'started_unix_s': time.time(), 'world_offsets_m': offsets.tolist()}
    (args.output/'batch-execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'active_gpu': args.gpu, 'physics_gpu': args.gpu,
                         'multi_gpu': False, 'disable_viewport_updates': True,
                         'fast_shutdown': True, 'width': 640, 'height': 480})
    exit_code = 0; environments = []; started = time.time(); t = None; phase = 'initialization'
    try:
        import carb
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, Vt
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import DeformablePrim, RigidPrim
        import omni.replicator.core as rep
        from isaac_dataset import boundary_shift, camera_pose, forces_at
        settings = carb.settings.get_settings()
        settings.set_bool('/persistent/physics/enableDeformableBeta', True)
        settings.set_bool('/physics/updateToUsd', True)
        settings.set_bool('/physics/updateVelocitiesToUsd', True)
        world = World(stage_units_in_meters=1., physics_dt=1/hz, rendering_dt=1/hz,
                      backend='warp', device=f'cuda:{args.gpu}')
        stage = world.stage; UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Get(stage, '/physicsScene'); scene.CreateGravityMagnitudeAttr(0.)
        api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        api.CreateEnableGPUDynamicsAttr(True); api.CreateBroadphaseTypeAttr('GPU')
        UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1000.)
        for index, (case, root, offset) in enumerate(zip(cases, roots, offsets)):
            environments.append(create_environment(stage, case, root, offset, index, rep))
        # Keep collision meshes enabled (PhysX volume views require them), but
        # explicitly filter each other independent body. Original episodes have
        # no inter-object contact, so no physical interaction is removed.
        body_prims = [stage.GetPrimAtPath(f'/World/Env_{i:02d}/Block') for i in range(len(environments))]
        for index, prim in enumerate(body_prims):
            UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(
                [other.GetPath() for i, other in enumerate(body_prims) if i != index])
        # Physics bodies coincide with inter-body contact filtered. The renderer shows
        # exactly one independent material surface through one shared camera.
        camera = UsdGeom.Camera.Define(stage, '/World/BatchCamera')
        camera_op = camera.AddTransformOp(); cam = environments[0]['spec']['camera']
        camera.CreateFocalLengthAttr(cam['focal_length_mm'])
        camera.CreateHorizontalApertureAttr(cam['horizontal_aperture_mm'])
        camera.CreateVerticalApertureAttr(cam['vertical_aperture_mm'])
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.))
        camera_op.Set(Gf.Matrix4d(*camera_pose(environments[0]['spec'], 0).ravel().tolist()))
        product = rep.create.render_product(str(camera.GetPath()), tuple(cam['resolution']))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product])
        depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane'); depth.attach([product])
        for index, env in enumerate(environments):
            env.update(rgb=rgb, depth=depth, camera_op=camera_op)
            if index:
                env['display'].MakeInvisible()
        stage.GetRootLayer().Export(str(args.output/'batch-scene.usda'))
        world.reset()
        # Installed6.2 explicitly rejects list[str] for deformable tensor paths.
        bodies = DeformablePrim('/World/Env_.*/Block', deformable_type='volume')
        handles = [RigidPrim(f'/World/Env_.*/Load_{i}') for i in range(2)]
        expected_world = np.array([e['points']+e['offset'] for e in environments])
        body_to_case, case_to_body = match_prim_paths(
            bodies.simulation_mesh_paths, [f'/World/Env_{i:02d}/Block' for i in range(len(environments))])
        np.testing.assert_allclose(bodies.get_nodal_positions()[0].numpy()[case_to_body], expected_world, atol=1e-6, rtol=0)
        handle_orders = []
        for i, handle in enumerate(handles):
            expected = np.array([e['points'][e['spec']['load_nodes'][i]]+e['offset'] for e in environments])
            # Access documented physics-tensor RigidBodyView.prim_paths via the
            # installed wrapper's view attribute; record the binding explicitly.
            order = match_prim_paths(handle._physics_rigid_body_view.prim_paths,
                                     [f'/World/Env_{j:02d}/Load_{i}' for j in range(len(environments))])
            np.testing.assert_allclose(handle.get_world_poses()[0].numpy()[order[1]], expected, atol=1e-6, rtol=0)
            handle_orders.append(order)
        execution['body_tensor_row_to_case'] = body_to_case.tolist()
        execution['handle_tensor_row_to_case'] = [order[0].tolist() for order in handle_orders]
        (args.output/'batch-execution.json').write_text(json.dumps(execution, indent=2)+'\n')
        enabled = np.zeros(expected_world.shape[:-1]+(1,), dtype=bool)
        for index, env in enumerate(environments):
            enabled[index, env['spec']['fixture_nodes']] = True
        bodies.set_nodal_kinematic_position_targets(expected_world[body_to_case].astype(np.float32),
                                                    enabled=enabled[body_to_case])
        _, actual_enabled = bodies.get_nodal_kinematic_position_targets()
        if not np.array_equal(actual_enabled.numpy()[case_to_body], enabled):
            raise RuntimeError('Batched kinematic flags differ from declared fixtures')
        phase = 'sensor warmup'
        ready = [False]*len(environments)
        for attempt in range(args.warmup_renders):
            world.render()
            if attempt >= 2:
                ready = [capture(env) is not None for env in environments]
                if all(ready):
                    break
        execution['warmup_renders'] = attempt+1
        for env, is_ready in zip(environments, ready):
            if not is_ready:
                write_failure(env, RuntimeError('Sensor not ready after bounded paused warmup'), hashes, 0., phase)
        if not any(e['active'] for e in environments):
            raise RuntimeError('No ready cameras in batch')
        start_clock = world.current_time
        phase = 'simulation'
        pending = []
        with ThreadPoolExecutor(max_workers=min(8, 2*len(environments))) as writers:
            for step in range(round(seconds*hz)+1):
                t = step/hz
                targets = expected_world.copy()
                for index, env in enumerate(environments):
                    fixed = env['spec']['fixture_nodes']
                    targets[index, fixed] += boundary_shift(env['spec'], t)
                if step:
                    forces = np.array([forces_at(e['spec'], t-.5/hz) for e in environments])
                    bodies.set_nodal_kinematic_position_targets(targets[body_to_case].astype(np.float32),
                                                                enabled=enabled[body_to_case])
                    before_velocity = np.stack([h.get_velocities()[0].numpy()[order[1]] for h, order in zip(handles, handle_orders)], 1)
                    applications = np.stack([h.get_world_poses()[0].numpy()[order[1]] for h, order in zip(handles, handle_orders)], 1)-offsets[:, None]
                    for i, (handle, order) in enumerate(zip(handles, handle_orders)):
                        handle.apply_forces(forces[order[0], i])
                    world.step(render=False, update_fabric=True)
                    if abs((world.current_time-start_clock)-t) > 1e-7*max(1, t):
                        raise RuntimeError('Shared simulation/force clock mismatch')
                    after_velocity = np.stack([h.get_velocities()[0].numpy()[order[1]] for h, order in zip(handles, handle_orders)], 1)
                    masses = np.array([e['spec']['handle_mass_kg_each'] for e in environments])
                    transmitted = forces-masses[:, None, None]*(after_velocity-before_velocity)*hz
                    for index, env in enumerate(environments):
                        r = env['records']; r['step_times'].append(t-.5/hz)
                        r['commands'].append(forces[index].copy()); r['application_positions'].append(applications[index].copy())
                        r['net_moments'].append(np.cross(applications[index]-env['spec']['camera']['center_m'], forces[index]).sum(0))
                else:
                    forces = np.zeros((len(environments), 2, 3), np.float32); transmitted = forces.copy()
                q_world = bodies.get_nodal_positions()[0].numpy()[case_to_body].copy()
                q_local = q_world-offsets[:, None]
                velocities = bodies.get_nodal_velocities().numpy()[case_to_body].copy()
                for index, env in enumerate(environments):
                    q = q_local[index]; fixed = env['spec']['fixture_nodes']
                    if not np.isfinite(q).all() or not np.isfinite(velocities[index]).all():
                        raise RuntimeError(f'Physics failure in {env["case"]["episode_id"]}: nonfinite nodal state; cohort aborted')
                    tets = env['tets']; ratio = np.linalg.det(q[tets[:, 1:]]-q[tets[:, :1]])/env['rest_det']
                    error = float(np.linalg.norm(q[fixed]-(targets[index, fixed]-offsets[index]), axis=-1).max())
                    env['min_volume'] = min(env['min_volume'], float(ratio.min()))
                    env['max_constraint_error'] = max(env['max_constraint_error'], error)
                    if ratio.min() < .1 or error > 2e-4:
                        raise RuntimeError(f'Physics failure in {env["case"]["episode_id"]}: compression/inversion or fixture drift; cohort aborted')
                if step % (hz//fps):
                    continue
                captures = [None]*len(environments)
                for env in environments:
                    env['display'].MakeInvisible()
                for index, env in enumerate(environments):
                    if not env['active']:
                        continue
                    pose = camera_pose(env['spec'], t)
                    camera_op.Set(Gf.Matrix4d(*pose.ravel().tolist()))
                    env['display'].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(q_local[index].astype(np.float32)))
                    env['display'].MakeVisible()
                    for _ in range(3):
                        world.render()
                    captures[index] = capture(env)
                    for _ in range(args.capture_retries):
                        if captures[index] is not None:
                            break
                        world.render(); captures[index] = capture(env)
                    env['display'].MakeInvisible()
                frame_handles = np.stack([h.get_world_poses()[0].numpy()[order[1]] for h, order in zip(handles, handle_orders)], 1)-offsets[:, None]
                for index, env in enumerate(environments):
                    if not env['active']:
                        continue
                    sensor = captures[index]
                    if sensor is None:
                        write_failure(env, RuntimeError('Invalid capture after bounded paused retries'), hashes, t, 'capture', q_local[index])
                        continue
                    image, depth, mask = sensor; r = env['records']; frame = len(r['times'])
                    env['minimum_mask_pixels'] = min(env['minimum_mask_pixels'], int(mask.sum()))
                    pending.append(writers.submit(write_sensor, env['root'], f'{frame:05d}', image, depth, mask))
                    if len(pending) >= 32:
                        pending.pop(0).result()
                    r['times'].append(t); r['measured'].append(world.current_time-start_clock)
                    r['positions'].append(q_local[index].copy()); r['velocities'].append(velocities[index].copy())
                    r['poses'].append(camera_pose(env['spec'], t)); r['frame_forces'].append(forces[index].copy())
                    r['targets'].append((targets[index, env['spec']['fixture_nodes']]-offsets[index]).copy())
                    r['handle_positions'].append(frame_handles[index].copy()); r['transmitted'].append(transmitted[index].copy())
                print(json.dumps({'batch_frame': step//(hz//fps), 'time_s': t,
                                  'active_episodes': [e['case']['episode_id'] for e in environments if e['active']],
                                  'elapsed_s': time.time()-started}), flush=True)
            for task in pending:
                task.result()
        phase = 'finalization'
        if any(Path(__file__).with_name(name).read_bytes() != content for name, content in sources.items()):
            raise RuntimeError('Source changed during batch generation')
        for env in environments:
            if env['active']:
                save_completed(env, hashes, hz, fps, len(environments), time.time()-started)
        if any(not e['active'] for e in environments):
            exit_code = 1
    except BaseException as error:
        exit_code = 1; traceback.print_exc()
        for index, env in enumerate(environments):
            write_failure(env, error, hashes, t, 'cohort abort during '+phase,
                          q_local[index] if 'q_local' in locals() else None)
        # If scene creation itself failed, record episodes not yet represented.
        for case, root in zip(cases, roots):
            if not (root/'summary.json').exists() and not (root/'failure.json').exists():
                (root/'failure.json').write_text(json.dumps({'completed': False, 'episode_id': case['episode_id'],
                    'phase': 'cohort abort during '+phase, 'exception': type(error).__name__, 'message': str(error),
                    'source_sha256': hashes, 'traceback': traceback.format_exc()}, indent=2)+'\n')
    finally:
        execution.update(completed=exit_code == 0, elapsed_s=time.time()-started,
                         outcomes=[{'episode_id': c['episode_id'], 'completed': (root/'summary.json').exists(),
                                    'failure': (root/'failure.json').exists()} for c, root in zip(cases, roots)])
        (args.output/'batch-execution.json').write_text(json.dumps(execution, indent=2)+'\n')
        app.close(exit_code=exit_code)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
