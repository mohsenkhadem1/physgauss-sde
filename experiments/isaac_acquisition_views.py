"""Render actual Isaac observations at new cameras of one unchanged saved state.

This generator may read simulator truth. Its public query loader exposes only
camera choices. Candidate pixels remain in truth/ until a policy has selected
one; independent evaluation cameras are never candidate acquisitions. No physics
is advanced and no existing episode is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def usd_look_at(center, azimuth_deg, elevation_deg, radius_m):
    """USD row-vector camera-to-world, local -Z forward and +Y up."""
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all() or radius_m <= 0:
        raise ValueError('Finite center and positive camera radius required')
    if not -89 < elevation_deg < 89:
        raise ValueError('Avoid singular polar camera poses')
    azimuth, elevation = np.deg2rad([azimuth_deg, elevation_deg])
    back = np.array([np.cos(elevation)*np.cos(azimuth),
                     np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    right = np.cross([0.,0.,1.], back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    pose = np.eye(4); pose[:3,:3] = np.stack((right,up,back))
    pose[3,:3] = center + radius_m * back
    return pose


def camera_choices(reference, time_s=1.5):
    """Fixed public choices depend only on known rest geometry, never motion."""
    points = np.asarray(reference['rest_points_m'])
    center = (points.min(0)+points.max(0))/2
    choices=[]
    for index, azimuth in enumerate([-135.,-45.,45.,135.]):
        choices.append({'view_id':f'candidate{index:02d}','role':'candidate',
                        'time_s':float(time_s),'azimuth_deg':azimuth,
                        'elevation_deg':25.,'radius_m':.92,
                        'camera_to_world_usd_row':usd_look_at(center,azimuth,25.,.92).tolist()})
    choices.append({'view_id':'evaluation00','role':'evaluation',
                    'time_s':float(time_s),'azimuth_deg':75.,'elevation_deg':43.,'radius_m':.97,
                    'camera_to_world_usd_row':usd_look_at(center,75.,43.,.97).tolist()})
    return choices


def saved_state(episode, time_s, tolerance_s=1e-6):
    """Generator-only state read, requiring an exact saved physical time."""
    with np.load(Path(episode)/'truth/trajectory.npz') as raw:
        times = raw['times_s']; index = int(np.argmin(abs(times-time_s)))
        if abs(float(times[index])-time_s) > tolerance_s:
            raise ValueError('Requested time is not a saved state; interpolation is forbidden')
        positions = raw['positions_m'][index].copy()
    if not np.isfinite(positions).all():
        raise ValueError('Nonfinite source state')
    return index, float(times[index]), positions


def load_public_view_queries(root, episode_id):
    """Public camera-only interface. It never opens state or rendered truth."""
    with np.load(Path(root)/'public'/episode_id/'cameras.npz') as raw:
        return {key:raw[key] for key in raw.files}


def selected_candidate_paths(root, episode_id, selected_view_id):
    """Reveal paths for exactly one already selected candidate, not evaluation."""
    queries = load_public_view_queries(root, episode_id)
    matches = np.flatnonzero(queries['view_ids'] == selected_view_id)
    if len(matches) != 1 or queries['roles'][matches[0]] != 'candidate':
        raise ValueError('Only a declared candidate can be acquired')
    folder = Path(root)/'truth'/episode_id/'views'/str(selected_view_id)
    return {'rgb_path':folder/'rgb.png','mask_path':folder/'mask.png',
            'depth_path':folder/'depth.npz','query_index':int(matches[0])}


def validate_capture(rgb, depth, positions, faces, pose, k, *, max_depth_error_m=2e-4):
    """CPU ray audit proves saved image depth and new camera match this state."""
    try:
        from validate_isaac_dataset import ray_depth
    except ImportError:
        from experiments.validate_isaac_dataset import ray_depth
    rgb, depth = np.asarray(rgb), np.asarray(depth).squeeze()
    if rgb.shape not in [(480,640,3),(480,640,4)] or depth.shape != (480,640):
        raise ValueError(f'Invalid RGB/depth readback shapes {rgb.shape}/{depth.shape}')
    mask = np.isfinite(depth) & (depth>.01) & (depth<10.)
    if mask.sum() < 1000:
        raise ValueError('Object absent or too small in sensor')
    # Exclude silhouette samples where raster filtering changes ray ownership.
    interior=mask.copy()
    for offset in [-2,-1,1,2]:
        interior &= np.roll(mask,offset,0) & np.roll(mask,offset,1)
    ys,xs=np.nonzero(interior)
    if len(xs)<64:
        raise ValueError('Insufficient interior pixels for camera/state verification')
    chosen=np.random.default_rng(9517).choice(len(xs),64,replace=False)
    xs,ys=xs[chosen],ys[chosen]
    expected=ray_depth(positions,faces,pose,k,xs,ys)
    if not np.isfinite(expected).all():
        raise ValueError('Sensor foreground differs from saved mesh/camera')
    error=float(np.max(abs(expected-depth[ys,xs])))
    if error>max_depth_error_m:
        raise ValueError(f'Stale/misaligned sensor: ray error {error:.6g}m')
    return {'mask_pixels':int(mask.sum()),'ray_depth_max_error_m':error,
            'depth_rays_tested':64,'touches_image_edge':bool(mask[[0,-1]].any() or mask[:,[0,-1]].any())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',nargs='*')
    parser.add_argument('--split',default='validation',choices=['train','validation','test','all'])
    parser.add_argument('--time',type=float,default=1.5)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--warmup-frames',type=int,default=60)
    parser.add_argument('--capture-attempts',type=int,default=12)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh output path; failed attempts are preserved')
    if not 30<=args.warmup_frames<=120 or not 1<=args.capture_attempts<=30:
        parser.error('Bounded warmup 30..120 and capture attempts 1..30 required')
    plan=json.loads((args.dataset/'plan.json').read_text())
    cases=[c for c in plan['episodes'] if args.split=='all' or c['split']==args.split]
    if args.episodes:
        cases=[c for c in cases if c['episode_id'] in args.episodes]
        if set(args.episodes)!={c['episode_id'] for c in cases}:
            parser.error('Requested episode absent from selected split')
    if not cases:parser.error('No episodes selected')
    root=args.output;root.mkdir(parents=True)
    (root/'provenance').mkdir()
    source=Path(__file__).read_bytes();source_hash=hashlib.sha256(source).hexdigest()
    (root/'provenance'/Path(__file__).name).write_bytes(source)
    spec={'dataset':'isaac-same-state-camera-acquisition-v0.1','source_dataset':str(args.dataset.resolve()),
          'source_plan_sha256':sha256_file(args.dataset/'plan.json'),
          'episode_ids':[c['episode_id'] for c in cases],'time_s':args.time,
          'camera_rule':'4 fixed candidate azimuths at25deg elevation; independent75deg/43deg evaluation camera',
          'state_policy':'Exact saved nodal state; no interpolation or physics step',
          'observation_policy':'Public camera choices; acquire one selected candidate; evaluation images remain withheld',
          'renderer':'Isaac colored surface mesh, vertex displayColor, no subdivision, double sided, DomeLight intensity1000',
          'gpu':args.gpu,'warmup_frames':args.warmup_frames,'capture_attempts':args.capture_attempts,
          'source_sha256':source_hash}
    (root/'acquisition-plan.json').write_text(json.dumps(spec,indent=2)+'\n')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':args.gpu,'physics_gpu':args.gpu,
                       'multi_gpu':False,'disable_viewport_updates':True,'fast_shutdown':True,
                       'width':640,'height':480})
    exit_code=0
    try:
        import carb
        from PIL import Image
        from pxr import Gf,UsdGeom,UsdLux,Vt
        from isaacsim.core.api import World
        import omni.replicator.core as rep
        import omni.timeline
        world=World(stage_units_in_meters=1.,physics_dt=1/60,rendering_dt=1/60,
                    backend='numpy',device='cpu')
        stage=world.stage;UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        surface=UsdGeom.Mesh.Define(stage,'/World/Surface')
        surface.CreateSubdivisionSchemeAttr('none');surface.CreateDoubleSidedAttr(True)
        UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(1000.)
        camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera_op=camera.AddTransformOp()
        camera.CreateFocalLengthAttr(35.);camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(27.);camera.CreateClippingRangeAttr(Gf.Vec2f(.01,10.))
        product=rep.create.render_product(str(camera.GetPath()),(640,480))
        rgb_annotator=rep.AnnotatorRegistry.get_annotator('rgb');rgb_annotator.attach([product])
        depth_annotator=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane');depth_annotator.attach([product])
        # This scene contains no dynamic actors. Paused rendering never advances q.
        world.reset();world.pause();omni.timeline.get_timeline_interface().pause()
        results=[];readiness=[]
        for episode_number,case in enumerate(cases):
            name=case['episode_id'];episode=args.dataset/'episodes'/name
            with np.load(episode/'reference.npz') as raw:
                reference={k:raw[k] for k in raw.files}
            frame,time_s,positions=saved_state(episode,args.time)
            with np.load(episode/'truth/cameras.npz') as raw:
                k=raw['K'].copy();original_pose=raw['camera_to_world_usd_row'][frame].copy()
            faces=reference['surface_faces'];colors=reference['vertex_colors']
            choices=camera_choices(reference,time_s)
            audit={'view_id':'source_replay','role':'audit','time_s':time_s,
                   'camera_to_world_usd_row':original_pose.tolist()}
            choices_with_audit=choices+[audit]
            folder=root/'truth'/name;folder.mkdir(parents=True)
            public=root/'public'/name;public.mkdir(parents=True)
            np.savez_compressed(public/'cameras.npz',K=k,
                                camera_to_world_usd_row=np.asarray([c['camera_to_world_usd_row'] for c in choices]),
                                times_s=np.full(len(choices),time_s),
                                view_ids=np.asarray([c['view_id'] for c in choices]),
                                roles=np.asarray([c['role'] for c in choices]))
            np.savez_compressed(folder/'state.npz',positions_m=positions,time_s=time_s,source_frame_id=frame)
            surface.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
            surface.CreateFaceVertexCountsAttr([3]*len(faces))
            surface.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
            surface.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
            surface.GetDisplayColorPrimvar().SetInterpolation('vertex')
            if episode_number==0:
                camera_op.Set(Gf.Matrix4d(*np.asarray(choices[0]['camera_to_world_usd_row']).ravel().tolist()))
                stage.GetRootLayer().Export(str(root/'render-scene.usda'))
                for _ in range(args.warmup_frames):world.render()
            state_sha=hashlib.sha256(np.ascontiguousarray(positions).tobytes()).hexdigest()
            captures=[]
            for choice in choices_with_audit:
                pose=np.asarray(choice['camera_to_world_usd_row'])
                camera_op.Set(Gf.Matrix4d(*pose.ravel().tolist()))
                last_error=None
                for attempt in range(1,args.capture_attempts+1):
                    for _ in range(4):world.render()
                    rgb=np.asarray(rgb_annotator.get_data());depth=np.asarray(depth_annotator.get_data()).squeeze()
                    try:
                        checks=validate_capture(rgb,depth,positions,faces,pose,k)
                        break
                    except ValueError as error:
                        last_error=str(error)
                        readiness.append({'episode_id':name,'view_id':choice['view_id'],
                                          'attempt':attempt,'error':last_error})
                else:
                    raise RuntimeError(f'Bounded sensor readiness failed: {name}/{choice["view_id"]}: {last_error}')
                mask=np.isfinite(depth)&(depth>.01)&(depth<10.)
                destination=folder/'views'/choice['view_id'];destination.mkdir(parents=True)
                Image.fromarray(rgb[:,:,:3]).save(destination/'rgb.png')
                Image.fromarray(mask.astype(np.uint8)*255).save(destination/'mask.png')
                np.savez_compressed(destination/'depth.npz',depth_m=np.where(mask,depth,0).astype(np.float32))
                entry={**choice,**checks,'capture_attempts':attempt,'state_array_sha256':state_sha}
                if choice['role']=='audit':
                    original_rgb=np.asarray(Image.open(episode/'truth/rgb'/f'{frame:05d}.png').convert('RGB'))
                    original_mask=np.asarray(Image.open(episode/'truth/masks'/f'{frame:05d}.png'))>0
                    difference=(rgb[:,:,:3].astype(float)-original_rgb)/255
                    entry['source_replay_rgb_rmse']=float(np.sqrt(np.mean(difference**2)))
                    entry['source_replay_mask_iou']=float((mask&original_mask).sum()/max((mask|original_mask).sum(),1))
                captures.append(entry)
                print(json.dumps({'episode_id':name,'view_id':choice['view_id'],**checks}),flush=True)
            source_files=['reference.npz','truth/trajectory.npz','truth/cameras.npz',
                          f'truth/rgb/{frame:05d}.png',f'truth/masks/{frame:05d}.png']
            record={'episode_id':name,'time_s':time_s,'source_frame_id':frame,
                    'state_array_sha256':state_sha,'captures':captures,
                    'source_sha256':{p:sha256_file(episode/p) for p in source_files}}
            (folder/'capture.json').write_text(json.dumps(record,indent=2)+'\n')
            results.append(record)
            (root/'progress.json').write_text(json.dumps({'completed_episodes':len(results),'episode_count':len(cases)},indent=2)+'\n')
        if sha256_file(__file__)!=source_hash:raise RuntimeError('Worker source changed during generation')
        (root/'readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
        manifest={str(p.relative_to(root)):sha256_file(p) for p in sorted(root.rglob('*')) if p.is_file()}
        (root/'file-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result={'completed':True,'episodes':len(results),'candidate_views_per_state':4,
                'independent_evaluation_views_per_state':1,'source_replay_audits_per_state':1,
                'time_s':args.time,'physics_steps':0,'sensor_retry_count':len(readiness),
                'maximum_ray_depth_error_m':max(c['ray_depth_max_error_m'] for r in results for c in r['captures']),
                'maximum_source_replay_rgb_rmse':max(c['source_replay_rgb_rmse'] for r in results for c in r['captures'] if c['role']=='audit'),
                'minimum_source_replay_mask_iou':min(c['source_replay_mask_iou'] for r in results for c in r['captures'] if c['role']=='audit')}
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    except BaseException as error:
        exit_code=1
        (root/'failure.json').write_text(json.dumps({'type':type(error).__name__,'error':str(error),
                                                    'traceback':traceback.format_exc()},indent=2)+'\n')
        traceback.print_exc()
    finally:
        app.close(exit_code=exit_code)


if __name__=='__main__':
    main()
