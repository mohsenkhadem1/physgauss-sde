"""Actual calibrated reset experiments with a declared load-handle interface.

Passive videos are followed by separate repeat experiments from rest. These
are not time-travel observations of the original rollout. Original material,
supports (including prescribed support motion), and BOTH rigid load patches
are retained. Original unknown loads are replaced by calibrated commands.

Public action queries/known handle centers are separate from private simulator
labels and unselected outcomes. All competing policies must receive the same
known intervention hardware. A policy records its selected ID before using
``load_selected_measurement``. Precomputed outcomes are counterfactual benchmark
branches, not information available to the selecting policy.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

CAPTURE_TIMES = np.linspace(0., 3., 7)


def calibrated_menu(centers, k, camera_pose, seconds=3.):
    """Public control/sensor queries; no material, supports, or response labels."""
    directions = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, -1., 0]])
    actions = [{'action_id': f'poke_{i}', 'kind': 'calibrated_reset_probe', 'handle_index': 0,
                'force_type': 'push_pulse', 'force_type_index': 0, 'peak_force_N': (.02*d).tolist(),
                'seconds': float(seconds), 'start_s': 0., 'cost': 1.} for i, d in enumerate(directions)]
    evaluation_direction = np.array([1., 1., .5]); evaluation_direction /= np.linalg.norm(evaluation_direction)
    evaluation = {'action_id': 'evaluation', 'kind': 'withheld_control_response', 'handle_index': 0,
                  'force_type': 'push_pulse', 'force_type_index': 0,
                  'peak_force_N': (.015*evaluation_direction).tolist(), 'seconds': float(seconds), 'start_s': 0.}
    return {'protocol': 'known-handle-reset-probes-v0.1', 'units': {'force': 'N', 'length': 'm', 'time': 's'},
            'known_handle_centers_m': np.asarray(centers).tolist(),
            'hardware': 'Two original 1g rigid handles and original nine-node patches retained; centers supplied as common side information',
            'sensor': {'K': np.asarray(k).tolist(), 'camera_to_world_usd_row': np.asarray(camera_pose).tolist(),
                       'times_s': (CAPTURE_TIMES*seconds/3.).tolist(), 'resolution': [640, 480]},
            'candidates': actions, 'evaluation_query': evaluation,
            'experiment': 'Independent reset at rest; original supports and prescribed support motion repeat; original loads replaced',
            'selection_rule': 'Choose using public queries and passive posterior before loading any candidate outcome'}


def command_at(action, time_s):
    from isaac_dataset import envelope
    force = np.zeros((2, 3), np.float32)
    force[0] = float(envelope('push_pulse', time_s-action.get('start_s', 0.), action['seconds']))*np.asarray(action['peak_force_N'])
    return force


def record_selection(root, policy, episode_id, action_id, posterior_digest='unspecified'):
    root = Path(root)
    menu_path = root/'public'/episode_id/'action-menu.json'; menu = json.loads(menu_path.read_text())
    if action_id not in {a['action_id'] for a in menu['candidates']}:
        raise ValueError('Select exactly one public candidate action')
    if any('/' in value or '\\' in value or value in ('', '.', '..') for value in [policy, episode_id]):
        raise ValueError('Invalid selection identifier')
    path = root/'selections'/policy/f'{episode_id}.json'; path.parent.mkdir(parents=True, exist_ok=True)
    value = {'policy': policy, 'episode_id': episode_id, 'action_id': action_id,
             'posterior_digest': posterior_digest, 'menu_sha256': hashlib.sha256(menu_path.read_bytes()).hexdigest(),
             'recorded_unix_s': time.time()}
    # Exclusive creation prevents replacing the choice after inspecting outcomes.
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
    return path


def load_selected_measurement(root, selection_path):
    """Load only selected RGB-D sensor paths/control metadata, never nodal truth."""
    root = Path(root).resolve(); selection_path = Path(selection_path).resolve()
    if root/'selections' not in selection_path.parents:
        raise ValueError('A recorded selection inside this experiment root is required')
    selected = json.loads(selection_path.read_text()); episode = selected['episode_id']
    menu_path = root/'public'/episode/'action-menu.json'; menu = json.loads(menu_path.read_text())
    if hashlib.sha256(menu_path.read_bytes()).hexdigest() != selected['menu_sha256']:
        raise ValueError('Action menu changed after selection')
    action = next((a for a in menu['candidates'] if a['action_id'] == selected['action_id']), None)
    if action is None:
        raise ValueError('Selected action is not a candidate')
    directory = root/'outcomes'/episode/action['action_id']/'observations'
    with np.load(directory/'cameras.npz') as raw:
        cameras = {k: raw[k] for k in raw.files}
    return {'action': action, 'known_handle_centers_m': np.asarray(menu['known_handle_centers_m']),
            'cameras': cameras, 'rgb_paths': sorted((directory/'rgb').glob('*.png')),
            'depth_paths': sorted((directory/'depth').glob('*.npz')),
            'mask_paths': sorted((directory/'masks').glob('*.png'))}


def capture(rgb, depth_annotator):
    image = np.asarray(rgb.get_data()); depth = np.asarray(depth_annotator.get_data()).squeeze()
    if image.ndim != 3 or image.shape[:2] != (480, 640) or depth.shape != (480, 640):
        return None
    mask = np.isfinite(depth) & (depth > .01) & (depth < 10.)
    if mask.sum() < 1000:
        return None
    return image[:, :, :3].copy(), np.where(mask, depth, 0).astype(np.float32), mask


def create_environment(stage, case, root):
    offset=np.zeros(3);index=0
    """Build one scene with the same material/constraints/handles as single-run."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
    from isaac_dataset import geometry, appearance, camera_pose
    points, tets, faces, spec = geometry(case)
    material = case['material']; colors, gaussians = appearance(points, faces)
    world_points = (points+offset).astype(np.float32)
    prefix = '/World/ProbeObject'
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


def validate_case(root, episode_id):
    """Independent CPU forces/clocks/rest/constraints/geometry/sensor/reset audit."""
    from PIL import Image
    from isaac_dataset import geometry, boundary_shift
    from validate_isaac_dataset import ray_depth
    root = Path(root); private = root/'private'/episode_id
    summary = json.loads((private/'summary.json').read_text())
    if not summary['completed'] or (private/'failure.json').exists():
        raise ValueError('Unsuccessful probe case')
    case = json.loads((private/'truth/recipe.json').read_text())
    points, tets, faces, spec = geometry(case)
    if json.loads((private/'truth/boundaries.json').read_text()) != spec:
        raise ValueError('Original support/handle geometry changed')
    menu = json.loads((root/'public'/episode_id/'action-menu.json').read_text())
    centers = points[spec['load_nodes']]
    np.testing.assert_allclose(menu['known_handle_centers_m'], centers, atol=1e-12)
    for name, expected in summary['source_sha256'].items():
        if hashlib.sha256((private/'provenance'/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Probe source snapshot hash mismatch')
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Probe source changed from frozen generation version')
    rest_det = np.linalg.det(points[tets[:, 1:]]-points[tets[:, :1]])
    results = []; saved = {}
    actions = menu['candidates']+[menu['evaluation_query'], {**menu['candidates'][0], 'action_id': 'reset_replay'}]
    for action in actions:
        action_id = action['action_id']; directory = root/'outcomes'/episode_id/action_id/'observations'
        with np.load(private/action_id/'trajectory.npz') as raw:
            a = {k: raw[k] for k in raw.files}
        saved[action_id] = a
        if not all(np.isfinite(v).all() for v in a.values()):
            raise ValueError('Nonfinite saved probe state')
        times = np.asarray(menu['sensor']['times_s']); hz = case['physics_hz']
        np.testing.assert_allclose(a['times_s'], times, atol=1e-12)
        np.testing.assert_allclose(a['simulated_times_s'], times, atol=3e-7, rtol=0)
        np.testing.assert_allclose(a['positions_m'][0], points, atol=1e-6)
        np.testing.assert_allclose(a['velocities_m_s'][0], 0, atol=1e-7)
        np.testing.assert_allclose(a['handle_positions_m'][0], centers, atol=1e-6)
        fixed = spec['fixture_nodes']; targets = np.array([points[fixed]+boundary_shift(spec, t) for t in times])
        np.testing.assert_allclose(a['boundary_target_positions_m'], targets, atol=1e-12)
        boundary_error = float(np.linalg.norm(a['positions_m'][:, fixed]-targets, axis=-1).max())
        ratio = np.linalg.det(a['positions_m'][:, tets[:, 1:]]-a['positions_m'][:, tets[:, :1]])/rest_det
        if ratio.min() < .1 or boundary_error > 2e-4:
            raise ValueError('Probe tetrahedron or fixture failure')
        patch_error = 0.
        for patch in spec['load_patches']:
            distances = np.linalg.norm(a['positions_m'][:, patch, None]-a['positions_m'][:, None, patch], axis=-1)
            rest_distances = np.linalg.norm(points[patch, None]-points[None, patch], axis=-1)
            patch_error = max(patch_error, float(abs(distances-rest_distances).max()))
        if patch_error > 5e-4:
            raise ValueError('Original rigid load patches were not retained')
        step_times = (np.arange(round(case['seconds']*hz))+.5)/hz
        np.testing.assert_allclose(a['force_step_midpoints_s'], step_times, atol=1e-12)
        commands = np.array([command_at(action, t) for t in step_times])
        np.testing.assert_allclose(a['force_step_commands_N'], commands, atol=1e-9)
        moments = np.cross(a['force_step_application_positions_m']-points.mean(0), commands).sum(1)
        np.testing.assert_allclose(a['force_step_net_moments_Nm'], moments, atol=1e-12)
        with np.load(directory/'cameras.npz') as raw:
            cameras = {k: raw[k] for k in raw.files}
        poses = np.broadcast_to(menu['sensor']['camera_to_world_usd_row'], (len(times), 4, 4))
        np.testing.assert_allclose(cameras['camera_to_world_usd_row'], poses, atol=1e-12)
        np.testing.assert_allclose(cameras['K'], menu['sensor']['K'], atol=1e-12)
        np.testing.assert_allclose(cameras['times_s'], times, atol=1e-12)
        max_depth = 0.; min_pixels = 640*480
        for folder, suffix in [('rgb', '.png'), ('masks', '.png'), ('depth', '.npz')]:
            if len(list((directory/folder).glob('*'+suffix))) != len(times):
                raise ValueError('Probe sensor frame count')
        for frame in range(len(times)):
            image = np.asarray(Image.open(directory/'rgb'/f'{frame:05d}.png'))
            mask = np.asarray(Image.open(directory/'masks'/f'{frame:05d}.png')) > 0
            with np.load(directory/'depth'/f'{frame:05d}.npz') as raw:
                depth = raw['depth_m']
            if image.shape != (480, 640, 3) or mask.shape != (480, 640) or depth.shape != (480, 640):
                raise ValueError('Probe sensor shape')
            if not np.isfinite(depth).all() or (depth[~mask] != 0).any() or (depth[mask] <= .01).any():
                raise ValueError('Probe depth/mask values')
            min_pixels = min(min_pixels, int(mask.sum()))
            if mask[[0, -1]].any() or mask[:, [0, -1]].any() or mask.sum() < 1000:
                raise ValueError('Probe object clipped or missing')
            ys, xs = np.where(mask); indices = np.linspace(0, len(xs)-1, 32).astype(int)
            xs, ys = xs[indices], ys[indices]
            ray = ray_depth(a['positions_m'][frame], faces, poses[frame], cameras['K'], xs, ys)
            valid = np.isfinite(ray)
            if valid.sum() < 30:
                raise ValueError('Probe depth silhouette disagreement')
            max_depth = max(max_depth, float(abs(ray[valid]-depth[ys[valid], xs[valid]]).max()))
        if max_depth > 2e-4:
            raise ValueError('Probe saved state/camera/depth disagreement')
        results.append({'action_id': action_id, 'minimum_saved_volume_ratio': float(ratio.min()),
                        'boundary_error_m': boundary_error, 'load_patch_distance_error_m': patch_error,
                        'maximum_ray_depth_error_m': max_depth,
                        'depth_rays_tested': 32*len(times), 'minimum_mask_pixels': min_pixels})
    reset_error = float(abs(saved['poke_0']['positions_m']-saved['reset_replay']['positions_m']).max())
    reset_velocity = float(abs(saved['poke_0']['velocities_m_s']-saved['reset_replay']['velocities_m_s']).max())
    if reset_error > 1e-6 or reset_velocity > 1e-5:
        raise ValueError(f'Reset replay differs: positions {reset_error}m, velocities {reset_velocity}m/s')
    return {'passed': True, 'episode_id': episode_id, 'actions': results,
            'reset_replay_max_position_error_m': reset_error,
            'reset_replay_max_velocity_error_m_s': reset_velocity,
            'qualification': 'Consistency at fixed simulator discretization; calibrated physical accuracy unverified'}


def run_worker(args):
    dataset, root = args.dataset, args.output; episode = args.episode
    private = root/'private'/episode
    if private.exists():
        raise ValueError('Use a fresh episode path; failed/partial runs are preserved')
    private.mkdir(parents=True)
    for folder in ['truth', 'provenance']:
        (private/folder).mkdir()
    original = dataset/'episodes'/episode
    case = json.loads((original/'truth/recipe.json').read_text())
    (private/'truth/recipe.json').write_text(json.dumps(case, indent=2)+'\n')
    names = ['isaac_calibrated_probes.py', 'isaac_dataset.py', 'boundary_block.py']
    sources = {name: Path(__file__).with_name(name).read_bytes() for name in names}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
    for name, content in sources.items():
        (private/'provenance'/name).write_bytes(content)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'active_gpu': args.gpu, 'physics_gpu': args.gpu,
                         'multi_gpu': False, 'disable_viewport_updates': True, 'fast_shutdown': True,
                         'width': 640, 'height': 480})
    exit_code = 0; action_id = None; t = None; started = time.time()
    try:
        import carb
        from PIL import Image
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, Vt
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import DeformablePrim, RigidPrim
        import omni.replicator.core as rep
        from isaac_dataset import boundary_shift
        settings = carb.settings.get_settings()
        settings.set_bool('/persistent/physics/enableDeformableBeta', True)
        settings.set_bool('/physics/updateToUsd', True)
        settings.set_bool('/physics/updateVelocitiesToUsd', True)
        hz = case['physics_hz']; world = World(stage_units_in_meters=1., physics_dt=1/hz,
            rendering_dt=1/hz, backend='warp', device=f'cuda:{args.gpu}')
        stage = world.stage; UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Get(stage, '/physicsScene'); scene.CreateGravityMagnitudeAttr(0.)
        scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        scene_api.CreateEnableGPUDynamicsAttr(True); scene_api.CreateBroadphaseTypeAttr('GPU')
        UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1000.)
        env = create_environment(stage, case, private)
        points, tets, spec = env['points'], env['tets'], env['spec']
        if json.loads((original/'truth/boundaries.json').read_text()) != spec:
            raise RuntimeError('Original handle/support geometry cannot be reproduced')
        with np.load(original/'observations/cameras.npz') as raw:
            k = raw['K']; pose = raw['camera_to_world_usd_row'][0]
        menu = calibrated_menu(points[spec['load_nodes']], k, pose, case['seconds'])
        public = root/'public'/episode; public.mkdir(parents=True)
        (public/'action-menu.json').write_text(json.dumps(menu, indent=2)+'\n')
        camera = UsdGeom.Camera.Define(stage, '/World/ProbeCamera'); camera_op = camera.AddTransformOp()
        camera.CreateFocalLengthAttr(35.); camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(27.); camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.))
        camera_op.Set(Gf.Matrix4d(*pose.ravel().tolist()))
        product = rep.create.render_product(str(camera.GetPath()), (640, 480))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product])
        depth_annotator = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane'); depth_annotator.attach([product])
        stage.GetRootLayer().Export(str(private/'initial-scene.usda'))
        fixed = np.array(spec['fixture_nodes']); enabled = np.zeros((1, len(points), 1), bool); enabled[0, fixed] = True
        mesh = UsdGeom.TetMesh.Get(stage, '/World/ProbeObject/Block')
        expected_handle_paths = [f'/World/ProbeObject/Load_{i}' for i in range(2)]
        action_list = menu['candidates']+[menu['evaluation_query'], {**menu['candidates'][0], 'action_id': 'reset_replay'}]
        summaries = []; body = None; handles = None
        for action in action_list:
            action_id = action['action_id']; directory = root/'outcomes'/episode/action_id/'observations'
            for folder in ['rgb', 'depth', 'masks']:
                (directory/folder).mkdir(parents=True)
            truth = private/action_id; truth.mkdir()
            (truth/'control.json').write_text(json.dumps(action, indent=2)+'\n')
            # A full stop/reset clears solver state and force accumulators.
            world.stop(); body = None; handles = None; gc.collect()
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.zeros_like(points, dtype=np.float32)))
            for index, path in enumerate(expected_handle_paths):
                prim = stage.GetPrimAtPath(path); xform = UsdGeom.Xformable(prim); xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(*points[spec['load_nodes'][index]]))
                xform.AddOrientOp().Set(Gf.Quatf(1., Gf.Vec3f(0.)))
                rigid_api = UsdPhysics.RigidBodyAPI(prim)
                rigid_api.GetVelocityAttr().Set(Gf.Vec3f(0.)); rigid_api.GetAngularVelocityAttr().Set(Gf.Vec3f(0.))
            world.reset()
            body = DeformablePrim('/World/ProbeObject/Block', deformable_type='volume')
            handles = RigidPrim(expected_handle_paths)
            paths = list(handles._physics_rigid_body_view.prim_paths)
            handle_order = np.array([expected_handle_paths.index(p) for p in paths])
            handle_inverse = np.argsort(handle_order)
            body.set_nodal_positions(points[None].astype(np.float32))
            body.set_nodal_velocities(np.zeros((1, len(points), 3), np.float32))
            body.set_nodal_kinematic_position_targets(points[None].astype(np.float32), enabled=enabled)
            handles.set_world_poses(positions=points[spec['load_nodes']][handle_order].astype(np.float32),
                                    orientations=np.tile([1., 0., 0., 0.], (2, 1)).astype(np.float32))
            handles.set_velocities(np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32))
            _, actual_enabled = body.get_nodal_kinematic_position_targets()
            if not np.array_equal(actual_enabled.numpy(), enabled):
                raise RuntimeError('Reset support flags mismatch')
            env['display'].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            for attempt in range(60):
                world.render()
                if attempt >= 2 and capture(rgb, depth_annotator) is not None:
                    break
            else:
                raise RuntimeError('Probe sensor not ready after bounded paused warmup')
            start_clock = world.current_time
            frames = {key: [] for key in ['times_s', 'simulated_times_s', 'positions_m', 'velocities_m_s',
                        'boundary_target_positions_m', 'handle_positions_m', 'force_step_midpoints_s',
                        'force_step_commands_N', 'force_step_application_positions_m', 'force_step_net_moments_Nm']}
            min_volume = 1.; max_bc_error = 0.; frame_index = 0
            capture_steps = set(np.rint(np.asarray(menu['sensor']['times_s'])*hz).astype(int))
            for step in range(round(case['seconds']*hz)+1):
                t = step/hz; target = points.copy(); target[fixed] += boundary_shift(spec, t)
                if step:
                    force = command_at(action, t-.5/hz)
                    if spec['boundary_type'] == 'moving_patch':
                        body.set_nodal_kinematic_position_targets(target[None].astype(np.float32), enabled=enabled)
                    applications = handles.get_world_poses()[0].numpy()[handle_inverse].copy()
                    handles.apply_forces(force[handle_order])
                    world.step(render=False, update_fabric=True)
                    if abs(world.current_time-start_clock-t) > 1e-7*max(1, t):
                        raise RuntimeError('Probe physics/command clock mismatch')
                    frames['force_step_midpoints_s'].append(t-.5/hz)
                    frames['force_step_commands_N'].append(force)
                    frames['force_step_application_positions_m'].append(applications)
                    frames['force_step_net_moments_Nm'].append(np.cross(applications-points.mean(0), force).sum(0))
                q = body.get_nodal_positions()[0].numpy()[0].copy()
                if not np.isfinite(q).all():
                    raise RuntimeError('Nonfinite probe nodal positions')
                volume = np.linalg.det(q[tets[:, 1:]]-q[tets[:, :1]])/env['rest_det']
                error = float(np.linalg.norm(q[fixed]-target[fixed], axis=-1).max())
                min_volume = min(min_volume, float(volume.min())); max_bc_error = max(max_bc_error, error)
                if volume.min() < .1 or error > 2e-4:
                    raise RuntimeError('Probe compression/inversion or support drift')
                if step not in capture_steps:
                    continue
                velocity = body.get_nodal_velocities().numpy()[0].copy()
                if not np.isfinite(velocity).all():
                    raise RuntimeError('Nonfinite probe nodal velocity')
                if step == 0:
                    np.testing.assert_allclose(q, points, atol=1e-6)
                    np.testing.assert_allclose(velocity, 0., atol=1e-7)
                env['display'].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(q))
                for _ in range(3): world.render()
                sensor = capture(rgb, depth_annotator)
                for _ in range(12):
                    if sensor is not None: break
                    world.render(); sensor = capture(rgb, depth_annotator)
                if sensor is None:
                    raise RuntimeError('Invalid probe sensor after bounded retries')
                image, depth, mask = sensor; name = f'{frame_index:05d}'
                Image.fromarray(image).save(directory/'rgb'/f'{name}.png', compress_level=3)
                Image.fromarray(mask.astype(np.uint8)*255).save(directory/'masks'/f'{name}.png', compress_level=3)
                np.savez_compressed(directory/'depth'/f'{name}.npz', depth_m=depth)
                frames['times_s'].append(t); frames['simulated_times_s'].append(world.current_time-start_clock)
                frames['positions_m'].append(q); frames['velocities_m_s'].append(velocity)
                frames['boundary_target_positions_m'].append(target[fixed])
                frames['handle_positions_m'].append(handles.get_world_poses()[0].numpy()[handle_inverse].copy())
                frame_index += 1
            np.savez_compressed(truth/'trajectory.npz', **frames)
            times = np.asarray(frames['times_s'])
            np.savez_compressed(directory/'cameras.npz', K=k, camera_to_world_usd_row=np.broadcast_to(pose, (len(times), 4, 4)),
                                times_s=times, frame_ids=np.arange(len(times)))
            summary = {'action_id': action_id, 'frames': len(times), 'minimum_runtime_volume_ratio': min_volume,
                       'maximum_runtime_boundary_error_m': max_bc_error, 'elapsed_s': time.time()-started}
            (truth/'summary.json').write_text(json.dumps(summary, indent=2)+'\n'); summaries.append(summary)
            print(json.dumps({'episode': episode, **summary}), flush=True)
        if any(Path(__file__).with_name(name).read_bytes() != content for name, content in sources.items()):
            raise RuntimeError('Probe source changed during simulation')
        summary = {'completed': True, 'episode_id': episode, 'source_sha256': hashes, 'actions': summaries,
                   'engine': 'Isaac Sim '+importlib.metadata.version('isaacsim')+' / PhysX GPU',
                   'reset': 'Full stop/reset; restore rest geometry and handle poses; zero qdot and rigid velocities',
                   'known_hardware_side_information': True, 'original_material_and_supports_retained': True,
                   'original_unknown_loads_replaced': True, 'physics_hz': hz, 'numerical_convergence_established': False,
                   'elapsed_s': time.time()-started}
        (private/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    except BaseException as error:
        exit_code = 1; traceback.print_exc()
        failure = {'completed': False, 'episode_id': episode, 'action_id': action_id, 'time_s': t,
                   'exception': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(),
                   'source_sha256': hashes}
        (private/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        if 'q' in locals():
            np.savez_compressed(private/'failure-state.npz', positions_m=q)
    finally:
        app.close(exit_code=exit_code)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--split', default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--episode', help='Internal single-case worker mode')
    parser.add_argument('--resume', action='store_true', help='Resume only cases that independently validate against identical source')
    parser.add_argument('--isaac-python', default='/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python')
    args = parser.parse_args()
    if args.episode:
        return run_worker(args)
    if args.output.exists() and not args.resume:
        parser.error('Use a fresh output root, or --resume for unchanged verified sources')
    args.output.mkdir(parents=True, exist_ok=True); (args.output/'logs').mkdir(exist_ok=True)
    import fcntl
    lock = (args.output/'.probe-generation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = json.loads((args.dataset/'plan.json').read_text())
    cases = [case for case in plan['episodes'] if case['split'] == args.split][:args.limit]
    results = []
    for case in cases:
        episode = case['episode_id']; started = time.time()
        command = [args.isaac_python, '-u', str(Path(__file__).resolve()), '--dataset', str(args.dataset),
                   '--output', str(args.output), '--episode', episode, '--gpu', str(args.gpu)]
        try:
            if (args.output/'private'/episode).exists():
                if not args.resume:
                    raise ValueError('Case output already exists')
                audit = validate_case(args.output, episode)
                result = {'episode_id': episode, 'passed': True, 'status': 'resumed_verified', 'elapsed_s': time.time()-started}
                results.append(result); print(json.dumps(result), flush=True)
                continue
            with (args.output/'logs'/f'{episode}.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES'}, timeout=1200)
            if result.returncode:
                raise RuntimeError(f'Probe worker exit {result.returncode}')
            audit = validate_case(args.output, episode)
            (args.output/'private'/episode/'validation.json').write_text(json.dumps(audit, indent=2)+'\n')
            result = {'episode_id': episode, 'passed': True, 'elapsed_s': time.time()-started}
        except Exception as error:
            result = {'episode_id': episode, 'passed': False, 'error': repr(error), 'elapsed_s': time.time()-started}
        results.append(result); print(json.dumps(result), flush=True)
        (args.output/'execution.json').write_text(json.dumps({'cases': results, 'completed': len(results) == len(cases) and all(r['passed'] for r in results)}, indent=2)+'\n')
        if not result['passed']:
            break
    completed = len(results) == len(cases) and all(r['passed'] for r in results)
    (args.output/'execution.json').write_text(json.dumps({'cases': results, 'completed': completed}, indent=2)+'\n')
    return not completed


if __name__ == '__main__':
    raise SystemExit(main())
