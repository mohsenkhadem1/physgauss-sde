"""Render exact saved independent-FEM states with Isaac; never run physics.

Primary observations and private candidate-view targets share saved physical
times. Cameras and appearance use public rest geometry only. Rendering a saved
state supplies a view target, not a forecast or an interpolated motion frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_completed_input(folder, expected_trajectory_sha256):
    """Open only one completed declared TRAIN response and its frozen inputs."""
    folder = Path(folder)
    protocol = json.loads((folder / 'protocol.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    execution = json.loads((folder / 'execution.json').read_text())
    episode = protocol['episode_id']
    if episode not in ('t0000', 't0001'):
        raise ValueError('This adapter smoke permits only TRAIN t0000/t0001')
    if not (execution['terminal'] and execution['completed'] and summary['completed']):
        raise ValueError('Incomplete source trajectory')
    if summary['case'] != episode or summary['response'] != protocol['response']:
        raise ValueError('Source identity mismatch')
    if protocol['native_equivalence'] or protocol['predictive_reserved_responses_opened']:
        raise ValueError('Expected independent TRAIN-only source')
    hashes = {}
    for original, expected in protocol['source_input_dependency_and_gate_sha256'].items():
        frozen = folder / 'provenance' / Path(original).name
        if sha256(frozen) != expected:
            raise ValueError('Frozen source/input hash mismatch: ' + str(frozen))
        hashes[str(frozen)] = expected
    fixture = json.loads((folder / 'provenance/gaussian_independent_diagnostic_inputs_v01.json').read_text())
    selected = next(row for row in fixture['cases'] if row['episode_id'] == episode)
    if selected['recipe']['split'] != 'train':
        raise ValueError('Non-TRAIN input is forbidden')
    trajectory_path = folder / 'trajectory.npz'
    actual = sha256(trajectory_path)
    if actual != expected_trajectory_sha256 or actual != summary['trajectory_sha256']:
        raise ValueError('Trajectory hash mismatch')
    static_path = folder / 'provenance/static-reference.npz'
    if sha256(static_path) != protocol['saved_regenerated_static_reference_sha256']:
        raise ValueError('Static reference hash mismatch')
    with np.load(static_path, allow_pickle=False) as raw:
        reference = {key: raw[key].copy() for key in ('rest_points_m', 'surface_faces', 'tetrahedra')}
    for row in selected['reference_static_arrays']:
        value = reference[row['key']]
        if (list(value.shape) != row['shape'] or value.dtype != np.dtype(row['dtype'])
                or array_sha256(value) != row['sha256']):
            raise ValueError('Canonical static array mismatch: ' + row['key'])
    with np.load(trajectory_path, allow_pickle=False) as raw:
        times = raw['times_s'].copy()
        actual_times = raw['actual_state_times_s'].copy()
        positions = raw['positions_m'].copy()
    expected_times = np.arange(31, dtype=np.float64) / 10
    if (protocol['output_hz'] != 10 or times.shape != (31,)
            or not np.array_equal(times, expected_times)
            or not np.allclose(actual_times, times, atol=1e-12, rtol=0)):
        raise ValueError('Require exact saved 31-state 10Hz clock; no interpolation')
    points, faces = reference['rest_points_m'], reference['surface_faces']
    if positions.shape != (31, len(points), 3) or not np.isfinite(positions).all():
        raise ValueError('Invalid saved positions')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(points):
        raise ValueError('Invalid exact static surface faces')
    for path in [trajectory_path, static_path, folder/'protocol.json', folder/'summary.json', folder/'execution.json']:
        hashes[str(path)] = sha256(path)
    return dict(episode=episode, protocol=protocol, times_s=times, positions_m=positions,
                reference=reference, input_sha256=hashes, source_directory=str(folder.resolve()))


def frame_indices(indices, frame_count):
    if not indices or len(set(indices)) != len(indices) or sorted(indices) != list(indices):
        raise ValueError('Declare unique increasing exact source frame indices')
    if indices[0] < 0 or indices[-1] >= frame_count:
        raise ValueError('Frame index outside saved trajectory')
    return np.asarray(indices, dtype=np.int32)


def rest_camera_plan(rest_points, seconds=3., candidate_offsets=((60., 10.),)):
    points = np.asarray(rest_points, dtype=np.float64)
    low, high = points.min(0), points.max(0)
    diameter = float(np.linalg.norm(high-low))
    if not np.isfinite(points).all() or diameter <= 0 or seconds <= 0:
        raise ValueError('Invalid public rest geometry/duration')
    offsets = np.asarray(candidate_offsets, dtype=float).reshape(-1, 2)
    if not np.isfinite(offsets).all() or np.any(abs(offsets[:, 1]) > 30):
        raise ValueError('Use finite fixed angular offsets, elevation offset within30deg')
    return {'center_m': ((low+high)/2).tolist(), 'radius_m': 2.6*diameter,
            'seconds': float(seconds), 'azimuth_start_deg': 25., 'arc_deg': 110.,
            'elevation_deg': 25., 'elevation_excursion_deg': 8.,
            'candidate_angular_offsets_deg': offsets.tolist(),
            'dependency': 'Public rest bounds and physical time only; no current/future geometry, material or boundary labels'}


def camera_pose(plan, time_s, offset=(0., 0.)):
    u = float(time_s) / plan['seconds']
    if not 0 <= u <= 1:
        raise ValueError('Camera time outside declared duration')
    az = np.deg2rad(plan['azimuth_start_deg'] + plan['arc_deg']*(3*u*u-2*u*u*u) + offset[0])
    el = np.deg2rad(plan['elevation_deg'] + plan['elevation_excursion_deg']*np.sin(np.pi*u) + offset[1])
    back = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    right = np.cross([0., 0., 1.], back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    pose = np.eye(4); pose[:3, :3] = np.stack([right, up, back])
    pose[3, :3] = np.asarray(plan['center_m']) + plan['radius_m']*back
    return pose


def camera_intrinsics(width, height, focal_length=35., horizontal_aperture=36.):
    if width <= 0 or height <= 0:
        raise ValueError('Positive resolution required')
    vertical_aperture = horizontal_aperture*height/width
    k = np.array([[width*focal_length/horizontal_aperture, 0., width/2],
                  [0., height*focal_length/vertical_aperture, height/2], [0., 0., 1.]])
    return k, vertical_aperture


def rest_vertex_colors(points):
    """Fixed appearance attached to rest vertices; no physical-label encoding."""
    points = np.asarray(points)
    extent = np.maximum(np.ptp(points, axis=0), 1e-12)
    return (.15 + .65*(points-points.min(0))/extent).astype(np.float32)


def project_points(points, pose, k):
    local = (np.asarray(points)-pose[3, :3]) @ pose[:3, :3].T
    z = -local[:, 2]
    pixels = np.stack([k[0, 0]*local[:, 0]/z+k[0, 2],
                       -k[1, 1]*local[:, 1]/z+k[1, 2]], axis=1)
    return pixels, z


def ray_surface(points, faces, pose, k, pixels_xy, chunk=256):
    """Independent two-sided triangle intersection, image-plane metric depth."""
    tri = np.asarray(points, dtype=float)[faces]
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    normal = np.cross(e1, e2); normal_norm = np.linalg.norm(normal, axis=1)
    s = pose[3, :3]-tri[:, 0]; q = np.cross(s, e1)
    output = []
    for start in range(0, len(pixels_xy), chunk):
        pixels = np.asarray(pixels_xy[start:start+chunk])
        rays = np.stack([(pixels[:, 0]-k[0, 2])/k[0, 0],
                         -(pixels[:, 1]-k[1, 2])/k[1, 1], -np.ones(len(pixels))], -1) @ pose[:3, :3]
        h = np.cross(rays[:, None], e2[None]); determinant = (e1[None]*h).sum(-1)
        inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=abs(determinant)>1e-12)
        u = inverse*(s[None]*h).sum(-1); v = inverse*(rays[:, None]*q[None]).sum(-1)
        depth = inverse*(e2*q).sum(-1)[None]
        valid = (u>=0)&(v>=0)&(u+v<=1)&(depth>0)&(abs(determinant)>1e-12)
        intersections = np.where(valid, depth, np.inf)
        closest = intersections.argmin(1); t = intersections[np.arange(len(rays)), closest]
        cosine = abs((normal[closest]*rays).sum(-1))/(normal_norm[closest]*np.linalg.norm(rays, axis=1))
        output.append((t, np.where(np.isfinite(t), cosine, 0.), closest))
    return tuple(np.concatenate([row[i] for row in output]) for i in range(3))


def semantic_object_mask(value, shape, label='independent_object'):
    if not isinstance(value, dict) or 'data' not in value or 'info' not in value:
        raise ValueError('Semantic annotator not ready')
    data = np.asarray(value['data']).squeeze()
    if data.shape != shape:
        raise ValueError('Semantic image shape mismatch')
    labels = value['info'].get('idToLabels', {})
    ids = [int(key) for key, item in labels.items()
           if label in (item.get('class', []) if isinstance(item.get('class'), list) else [item.get('class')])]
    if not ids:
        raise ValueError('Object semantic label missing: ' + str(labels))
    return np.isin(data, ids), labels


def validate_capture(rgb, depth, mask, points, faces, pose, k, depth_tolerance_m=2e-6):
    h, w = depth.shape
    if rgb.shape[:2] != (h, w) or rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or mask.shape != (h, w):
        raise ValueError('RGB/depth/mask shape mismatch')
    if rgb.dtype != np.uint8 or not np.isfinite(rgb).all():
        raise ValueError('Expected finite uint8 RGB annotator output')
    valid = np.isfinite(depth)&(depth>.01)&(depth<10.)
    union = (valid|mask).sum()
    depth_mask_iou = float((valid&mask).sum()/max(union, 1))
    if mask.sum() < 256 or depth_mask_iou < .995 or mask[[0, -1]].any() or mask[:, [0, -1]].any():
        raise ValueError('Missing/clipped object or semantic/depth disagreement')
    rgb_contrast = float(abs(rgb[mask, :3].mean(0)-rgb[~mask, :3].mean(0)).mean()/255.)
    if rgb_contrast < .005:
        raise ValueError('RGB object/background contrast is absent')
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (mask[1:-1, 1:-1]&mask[:-2, 1:-1]&mask[2:, 1:-1]
                              &mask[1:-1, :-2]&mask[1:-1, 2:])
    ys, xs = np.where(interior)
    if len(xs) < 64:
        raise ValueError('Too few interior object pixels')
    choose = np.linspace(0, len(xs)-1, min(len(xs), 512)).astype(int)
    xs, ys = xs[choose], ys[choose]
    pixels = np.stack([xs+.5, ys+.5], -1)
    expected, incidence, triangles = ray_surface(points, faces, pose, k, pixels)
    eligible = np.flatnonzero(np.isfinite(expected)&(incidence>=.2))
    if len(eligible) < 64:
        raise ValueError('Too few well-conditioned mesh rays')
    selected = eligible[np.linspace(0, len(eligible)-1, 64).astype(int)]
    error = abs(expected[selected]-depth[ys[selected], xs[selected]])
    if float(error.max()) > depth_tolerance_m:
        raise ValueError(f'Stale/wrong geometry, depth or camera: ray error {error.max():.9g}m')
    # Deterministic coarse silhouette check includes background rays.
    gy, gx = np.mgrid[3:h:6, 3:w:6]
    grid = np.stack([gx.ravel()+.5, gy.ravel()+.5], -1)
    reference_depth, _, _ = ray_surface(points, faces, pose, k, grid)
    expected_mask = np.isfinite(reference_depth)
    actual_mask = mask[gy.ravel(), gx.ravel()]
    iou = float((expected_mask&actual_mask).sum()/max((expected_mask|actual_mask).sum(), 1))
    if iou < .98:
        raise ValueError(f'Independent ray-mask IoU below .98: {iou:.9g}')
    depth_values = depth[ys[selected], xs[selected]]
    local = np.stack([(pixels[selected, 0]-k[0, 2])/k[0, 0]*depth_values,
                      -(pixels[selected, 1]-k[1, 2])/k[1, 1]*depth_values, -depth_values], -1)
    world = local @ pose[:3, :3] + pose[3, :3]
    reprojection, _ = project_points(world, pose, k)
    reprojection_error = float(abs(reprojection-pixels[selected]).max())
    projected, z = project_points(points[np.unique(faces)], pose, k)
    ys_mask, xs_mask = np.where(mask)
    observed_bounds = np.array([xs_mask.min()+.5, ys_mask.min()+.5, xs_mask.max()+.5, ys_mask.max()+.5])
    projected_bounds = np.r_[projected.min(0), projected.max(0)]
    bounds_error = float(abs(observed_bounds-projected_bounds).max())
    if reprojection_error > 1e-8 or bounds_error > 2. or z.min() <= .01:
        raise ValueError('Camera reprojection/projected silhouette guard failed')
    return {'mask_pixels': int(mask.sum()), 'semantic_depth_mask_iou': depth_mask_iou,
            'rgb_foreground_background_mean_contrast': rgb_contrast,
            'independent_coarse_ray_mask_iou': iou, 'coarse_rays': len(grid),
            'conditioned_depth_rays': 64, 'minimum_incidence_cosine': float(incidence[selected].min()),
            'maximum_ray_depth_error_m': float(error.max()), 'depth_tolerance_m': depth_tolerance_m,
            'maximum_unproject_reproject_error_px': reprojection_error,
            'projected_bbox_max_error_px': bounds_error,
            'mask_touches_image_edge': False}


def gpu_assignment(required_gpu=2):
    """Check graphics AND compute contexts for this Isaac process."""
    xml = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True))
    devices = []
    for gpu in xml.findall('gpu'):
        for process in gpu.findall('./processes/process_info'):
            if int(process.findtext('pid')) == os.getpid():
                devices.append({'index': int(gpu.findtext('minor_number')), 'uuid': gpu.findtext('uuid'),
                                'pid': os.getpid(), 'type': process.findtext('type')})
    if not devices or {row['index'] for row in devices} != {required_gpu}:
        raise RuntimeError('Isaac process must be assigned only to requested GPU: ' + str(devices))
    return devices


def physics_schema_guard(stage):
    invalid = []
    for prim in stage.Traverse():
        schemas = [str(prim.GetTypeName()), *map(str, prim.GetAppliedSchemas())]
        if any('physics' in schema.lower() or 'physx' in schema.lower() for schema in schemas):
            invalid.append(str(prim.GetPath()))
    if invalid:
        raise RuntimeError('Render-only scene contains physics schemas: ' + str(invalid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--expected-trajectory-sha256', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame-indices', type=int, nargs='+', required=True)
    parser.add_argument('--candidate-offset-deg', type=float, nargs=2, action='append')
    parser.add_argument('--resolution', type=int, nargs=2, default=[320, 240])
    parser.add_argument('--gpu', type=int, choices=[2], default=2)
    parser.add_argument('--warmup-frames', type=int, default=30)
    parser.add_argument('--capture-attempts', type=int, default=8)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        parser.error('Use validated Isaac physical GPU indices; unset CUDA_VISIBLE_DEVICES')
    if sha256(__file__) != args.expected_source_sha256:
        parser.error('Adapter source does not match frozen hash')
    if args.output.exists() or not 3 <= args.warmup_frames <= 60 or not 1 <= args.capture_attempts <= 12:
        parser.error('Use fresh output, warmup3..60 and attempts1..12')
    data = load_completed_input(args.input, args.expected_trajectory_sha256)
    indices = frame_indices(args.frame_indices, len(data['times_s']))
    width, height = args.resolution
    k, vertical_aperture = camera_intrinsics(width, height)
    plan = rest_camera_plan(data['reference']['rest_points_m'], candidate_offsets=args.candidate_offset_deg or [(60., 10.)])
    views = [{'view_id': 'primary', 'role': 'observation', 'offset_deg': [0., 0.]}]
    views += [{'view_id': f'candidate{i:02d}', 'role': 'held_view', 'offset_deg': offset}
              for i, offset in enumerate(plan['candidate_angular_offsets_deg'])]
    root = args.output; root.mkdir(parents=True)
    for name in ['provenance', 'public', 'private']:
        (root/name).mkdir()
    (root/'provenance'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    protocol = {'role': 'TRAIN render-only adapter smoke; no reference qualification or predictive result',
                'physics_backend': 'independent float64 finite-strain FEM, saved trajectory',
                'rendering_backend': 'Isaac Sim plain USD Mesh + RTX/Replicator; no physics scene or actor',
                'source_sha256': args.expected_source_sha256, 'input_sha256': data['input_sha256'],
                'source_directory': data['source_directory'], 'episode_id': data['episode'],
                'source_saved_times_s': data['times_s'].tolist(), 'source_output_hz': 10,
                'selected_source_frame_indices': indices.tolist(), 'selected_times_s': data['times_s'][indices].tolist(),
                'interpolation': False, 'forecasting_claim': False, 'view_plan': plan, 'views': views,
                'resolution_xy': [width, height], 'gpu': args.gpu, 'CUDA_VISIBLE_DEVICES': None,
                'renderer': 'RaytracedLighting', 'warmup_frames': args.warmup_frames,
                'capture_attempts': args.capture_attempts, 'physics_steps': 0,
                'metric_depth_semantics': 'distance_to_image_plane, metres; invalid pixels saved as zero',
                'mask_semantics': 'semantic class independent_object; checked against independent surface rays',
                'private_candidate_pixels': True, 'appearance': 'Fixed normalized public-rest-coordinate vertex RGB; no material/BC/force marker encoding'}
    write_json(root/'protocol.json', protocol)
    write_json(root/'public/view-plan.json', {'camera_plan': plan, 'views': views, 'intrinsics_K': k.tolist(),
        'pose_convention': 'USD row-vector camera-to-world: right/up/back rows, eye in final row; pixels at x+.5,y+.5',
        'resolution_xy': [width, height], 'source_output_hz': 10, 'interpolation': False})
    reference = data['reference']; faces = reference['surface_faces']; colors = rest_vertex_colors(reference['rest_points_m'])
    np.savez_compressed(root/'public/reference.npz', rest_points_m=reference['rest_points_m'], surface_faces=faces, vertex_colors=colors)
    np.savez_compressed(root/'private/source-state-subset.npz', source_frame_indices=indices,
                        times_s=data['times_s'][indices], positions_m=data['positions_m'][indices])
    execution = {'pid': os.getpid(), 'terminal': False, 'completed': False, 'phase': 'starting_isaac',
                 'started_unix_s': time.time(), 'completed_images': 0, 'planned_images': len(indices)*len(views)}
    write_json(root/'execution.json', execution)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'active_gpu': args.gpu, 'physics_gpu': args.gpu,
                         'multi_gpu': False, 'disable_viewport_updates': True, 'fast_shutdown': True,
                         'renderer': 'RaytracedLighting', 'width': width, 'height': height})
    code = 0
    try:
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux, Vt
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from isaacsim.core.experimental.utils import semantics
        write_json(root/'provenance/gpu-after-startup.json', gpu_assignment(args.gpu))
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.)
        surface = UsdGeom.Mesh.Define(stage, '/World/IndependentSurface')
        surface.CreateFaceVertexCountsAttr([3]*len(faces)); surface.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        surface.CreateSubdivisionSchemeAttr('none'); surface.CreateDoubleSidedAttr(True)
        surface.CreateOrientationAttr('rightHanded')
        surface.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors))
        surface.GetDisplayColorPrimvar().SetInterpolation('vertex')
        semantics.add_labels(surface.GetPrim(), labels=['independent_object'])
        UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1000.)
        camera = UsdGeom.Camera.Define(stage, '/World/Camera'); camera_op = camera.AddTransformOp()
        camera.CreateFocalLengthAttr(35.); camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(vertical_aperture); camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.))
        initial = data['positions_m'][indices[0]].astype(np.float32)
        surface.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(initial))
        camera_op.Set(Gf.Matrix4d(*camera_pose(plan, float(data['times_s'][indices[0]])).ravel().tolist()))
        product = rep.create.render_product(str(camera.GetPath()), (width, height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator('rgb'); rgb_annotator.attach([product])
        depth_annotator = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane'); depth_annotator.attach([product])
        mask_annotator = rep.AnnotatorRegistry.get_annotator('semantic_segmentation', init_params={'colorize': False})
        mask_annotator.attach([product])
        timeline = omni.timeline.get_timeline_interface(); timeline.pause(); timeline.set_current_time(0.)
        physics_schema_guard(stage)
        stage.GetRootLayer().Export(str(root/'provenance/render-scene-initial.usda'))
        for _ in range(args.warmup_frames):
            rep.orchestrator.step(rt_subframes=1, delta_time=0., pause_timeline=True, wait_for_render=True)
        write_json(root/'provenance/gpu-before-capture.json', gpu_assignment(args.gpu))
        if timeline.get_current_time() != 0.:
            raise RuntimeError('Render-only warmup advanced timeline')
        records, failures = [], []
        for frame in indices:
            physical_time = float(data['times_s'][frame]); points = data['positions_m'][frame].astype(np.float32)
            surface.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
            surface.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([points.min(0), points.max(0)])))
            if not np.array_equal(np.asarray(surface.GetPointsAttr().Get()), points):
                raise RuntimeError('USD mesh points differ from requested exact float32 state')
            if not np.array_equal(np.asarray(surface.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), faces):
                raise RuntimeError('USD topology differs from exact static reference')
            for view in views:
                pose = camera_pose(plan, physical_time, view['offset_deg'])
                camera_op.Set(Gf.Matrix4d(*pose.ravel().tolist()))
                for attempt in range(1, args.capture_attempts+1):
                    for _ in range(3):
                        rep.orchestrator.step(rt_subframes=2, delta_time=0., pause_timeline=True, wait_for_render=True)
                    rgb = np.asarray(rgb_annotator.get_data()).copy()
                    depth = np.asarray(depth_annotator.get_data()).squeeze().copy()
                    try:
                        if depth.shape != (height, width):
                            raise ValueError('Depth annotator not ready')
                        mask, labels = semantic_object_mask(mask_annotator.get_data(), (height, width))
                        metrics = validate_capture(rgb, depth, mask, points.astype(float), faces, pose, k)
                        break
                    except ValueError as error:
                        failures.append({'source_frame_index': int(frame), 'view_id': view['view_id'],
                                         'attempt': attempt, 'error': str(error)})
                        write_json(root/'readiness-failures.json', failures)
                else:
                    raise RuntimeError('Bounded capture readiness/QC failed: ' + str(failures[-1]))
                physics_schema_guard(stage)
                if timeline.get_current_time() != 0. or timeline.is_playing():
                    raise RuntimeError('Render-only capture advanced/played timeline')
                destination = (root/'public/observations' if view['role']=='observation'
                               else root/'private/held_views'/view['view_id'])
                for name in ['rgb', 'depth', 'masks']:
                    (destination/name).mkdir(parents=True, exist_ok=True)
                name = f'{int(frame):05d}'
                Image.fromarray(rgb[:, :, :3]).save(destination/'rgb'/f'{name}.png')
                Image.fromarray(mask.astype(np.uint8)*255).save(destination/'masks'/f'{name}.png')
                np.savez_compressed(destination/'depth'/f'{name}.npz', depth_m=np.where(mask, depth, 0).astype(np.float32))
                row = {'source_frame_index': int(frame), 'time_s': physical_time, **view, **metrics,
                       'camera_to_world_usd_row': pose.tolist(), 'K': k.tolist(), 'attempt': attempt,
                       'source_state_array_sha256': array_sha256(data['positions_m'][frame]),
                       'usd_float32_state_array_sha256': array_sha256(points),
                       'maximum_float32_vertex_roundoff_m': float(abs(points-data['positions_m'][frame]).max()),
                       'labels': labels, 'timeline_time_s': float(timeline.get_current_time()),
                       'relative_pixels_directory': str(destination.relative_to(root))}
                records.append(row); write_json(root/'captures.json', records)
                execution.update(phase='capturing', completed_images=len(records)); write_json(root/'execution.json', execution)
                print(json.dumps({'source_frame': int(frame), 'time_s': physical_time, 'view': view['view_id'], **metrics}), flush=True)
        for role, target in [('observation', root/'public/observations/cameras.npz'), ('held_view', root/'public/query_cameras.npz')]:
            rows = [row for row in records if row['role']==role]
            np.savez_compressed(target, K=k, resolution_xy=np.array([width, height]),
                source_frame_indices=np.array([row['source_frame_index'] for row in rows]),
                times_s=np.array([row['time_s'] for row in rows]),
                view_ids=np.array([row['view_id'] for row in rows]),
                camera_to_world_usd_row=np.array([row['camera_to_world_usd_row'] for row in rows]))
        if sha256(__file__) != args.expected_source_sha256 or any(sha256(p)!=h for p, h in data['input_sha256'].items()):
            raise RuntimeError('Source/input changed during rendering')
        write_json(root/'readiness-failures.json', failures)
        write_json(root/'summary.json', {'completed': True, 'captured_images': len(records),
            'selected_times_s': data['times_s'][indices].tolist(), 'source_saved_frame_count': 31,
            'source_saved_hz': 10, 'generated_intermediate_states': 0, 'physics_steps': 0,
            'backend': 'Independent FEM physics + Isaac render-only', 'forecasting_claim': False,
            'all_sensor_checks_passed': True, 'private_candidate_pixels': True,
            'maximum_ray_depth_error_m': max(row['maximum_ray_depth_error_m'] for row in records),
            'minimum_independent_mask_iou': min(row['independent_coarse_ray_mask_iou'] for row in records)})
        execution.update(terminal=True, completed=True, phase='complete', finished_unix_s=time.time())
    except BaseException as error:
        code = 1; traceback.print_exc()
        write_json(root/'failure.json', {'exception': type(error).__name__, 'error': str(error)})
        execution.update(terminal=True, completed=False, phase='failed', finished_unix_s=time.time())
    finally:
        write_json(root/'execution.json', execution)
        app.close()
    manifest = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}
    write_json(root/'file-manifest.json', manifest)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
