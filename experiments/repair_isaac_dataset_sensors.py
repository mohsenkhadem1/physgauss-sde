"""Repair stale Isaac sensor frames from exact saved states, never re-run physics.

Original rejected episodes are renamed intact before a byte-verified working
copy is made. Only failed sensor frames are replaced; full independent physics
and sensor validation is required before atomic promotion. Original worker
source attribution stays intact; replacement pixels have separate provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np
from PIL import Image, ImageDraw


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tree_hashes(root):
    root=Path(root)
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def ray_depth(*args):
    # Same independent Moller-Trumbore camera-axis audit as the repository
    # validator, kept dependency-free so public workflow tests need no simulator.
    points,faces,pose,k,xs,ys=args
    ray=np.stack([(xs+.5-k[0,2])/k[0,0],-(ys+.5-k[1,2])/k[1,1],-np.ones(len(xs))],-1)@pose[:3,:3]
    tri=points[faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    h=np.cross(ray[:,None],e2[None]);det=(e1[None]*h).sum(-1)
    inv=np.divide(1,det,out=np.zeros_like(det),where=abs(det)>1e-10)
    s=pose[3,:3]-tri[:,0];u=inv*(s[None]*h).sum(-1);q=np.cross(s,e1)
    v=inv*(ray[:,None]*q[None]).sum(-1);t=inv*(e2*q).sum(-1)[None]
    return np.where((u>=0)&(v>=0)&(u+v<=1)&(t>0)&(abs(det)>1e-10),t,np.inf).min(1)


def interior(mask,margin=3):
    result=mask.copy()
    for offset in range(1,margin+1):
        for axis in [0,1]:result &= np.roll(mask,offset,axis)&np.roll(mask,-offset,axis)
    result[:margin]=False;result[-margin:]=False;result[:,:margin]=False;result[:,-margin:]=False
    return result


def rest_projection_mask(points,faces,pose,k,shape=(480,640)):
    local=(np.asarray(points)-pose[3,:3])@pose[:3,:3].T
    depth=-local[:,2]
    if np.any(depth<=0):raise ValueError('Reference vertices behind camera')
    pixels=np.stack((k[0,0]*local[:,0]/depth+k[0,2]-.5,-k[1,1]*local[:,1]/depth+k[1,2]-.5),-1)
    image=Image.new('1',(shape[1],shape[0]));draw=ImageDraw.Draw(image)
    for face in faces:draw.polygon([tuple(v) for v in pixels[face]],fill=1)
    return np.asarray(image).astype(bool)


def audit_frame(rgb,mask,depth,q,reference,pose,k):
    """32 known-rest interior rays AND 32 current-mask rays, 0.2mm bound."""
    rgb=np.asarray(rgb);mask=np.asarray(mask)>0;depth=np.asarray(depth)
    errors=[];details=[]
    if rgb.shape!=(480,640,3) or depth.shape!=(480,640) or mask.shape!=(480,640):
        return {'passed':False,'error':'Sensor shapes','rays_tested':0}
    if not np.isfinite(depth).all() or np.any(depth[~mask]!=0) or np.any(depth[mask]<=.01):errors.append('Depth/mask mismatch')
    if mask.sum()<1000 or mask[[0,-1]].any() or mask[:,[0,-1]].any():errors.append('Mask small/clipped')
    faces=reference['surface_faces']
    rest=interior(rest_projection_mask(reference['rest_points_m'],faces,pose,k),6)
    for name,region in [('known_rest_interior',rest),('observed_mask_interior',interior(mask,3))]:
        ys,xs=np.where(region)
        if len(xs)<32:errors.append(name+': insufficient interior');continue
        selected=np.random.default_rng(530053).choice(len(xs),32,replace=False);xs=xs[selected];ys=ys[selected]
        expected=ray_depth(q,faces,pose,k,xs,ys);finite=np.isfinite(expected)
        valid=finite&mask[ys,xs]&(depth[ys,xs]>.01)
        # Rest projections can leave the moving silhouette: assess only rays
        # hitting current q, and require all such hits visible in saved depth.
        if name=='known_rest_interior':
            if np.any(finite&~valid):errors.append(name+': missing saved foreground')
            if np.any(~finite&mask[ys,xs]):errors.append(name+': foreground where current mesh has no hit')
        elif not finite.all():errors.append(name+': foreground not on saved mesh')
        maximum=float(np.max(np.abs(depth[ys[valid],xs[valid]]-expected[valid]))) if valid.any() else (0. if name=='known_rest_interior' and not finite.any() else None)
        if maximum is None or maximum>=2e-4:errors.append(name+': stale/misaligned depth')
        details.append({'region':name,'rays':32,'finite_mesh_hits':int(finite.sum()),'valid_depth_hits':int(valid.sum()),'maximum_error_m':maximum})
    return {'passed':not errors,'errors':errors,'regions':details,'rays_tested':sum(d['rays'] for d in details),'mask_pixels':int(mask.sum())}


def load_arrays(episode):
    def read(path):
        with np.load(path) as raw:return {k:raw[k].copy() for k in raw.files}
    episode=Path(episode)
    reference=read(episode/'reference.npz');trajectory=read(episode/'truth/trajectory.npz');cameras=read(episode/'truth/cameras.npz')
    n=len(trajectory['times_s'])
    np.testing.assert_array_equal(cameras['frame_ids'],np.arange(n))
    np.testing.assert_allclose(cameras['times_s'],trajectory['times_s'],atol=1e-12,rtol=0)
    if trajectory['positions_m'].shape!=(n,len(reference['rest_points_m']),3):raise ValueError('Trajectory/camera alignment')
    return reference,trajectory,cameras


def audit_episode(episode):
    episode=Path(episode);reference,trajectory,cameras=load_arrays(episode);frames=[]
    for frame in range(len(trajectory['times_s'])):
        stem=f'{frame:05d}'
        rgb=np.asarray(Image.open(episode/'truth/rgb'/(stem+'.png')).convert('RGB'))
        mask=np.asarray(Image.open(episode/'truth/masks'/(stem+'.png')))
        with np.load(episode/'truth/depth'/(stem+'.npz')) as raw:depth=raw['depth_m']
        check=audit_frame(rgb,mask,depth,trajectory['positions_m'][frame],reference,cameras['camera_to_world_usd_row'][frame],cameras['K'])
        frame_hashes={str(p.relative_to(episode)):sha(p) for p in [episode/'truth/rgb'/(stem+'.png'),episode/'truth/masks'/(stem+'.png'),episode/'truth/depth'/(stem+'.npz')]}
        frames.append({'frame_id':frame,'time_s':float(trajectory['times_s'][frame]),'sensor_sha256':frame_hashes,**check})
    return {'episode_id':episode.name,'passed':all(f['passed'] for f in frames),'bad_frames':[f['frame_id'] for f in frames if not f['passed']],'frames':frames,'threshold_m':2e-4,'physics_resampled':False,'input_sha256':{n:sha(episode/n) for n in ['reference.npz','truth/trajectory.npz','truth/cameras.npz']},'audit_source_sha256':sha(__file__)}


def assert_no_episode_worker(episode_id,episode):
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:parts=proc.read_bytes().decode().split('\0')
        except (OSError,UnicodeError):continue
        if not any(Path(p).name in ['isaac_dataset_worker.py','isaac_batch_dataset_worker.py'] for p in parts):continue
        if episode_id in parts or str(episode.resolve()) in parts:raise RuntimeError('Original episode worker still active')


def prepare_repair(dataset,episode_id):
    dataset=Path(dataset);episode=dataset/'episodes'/episode_id
    status_path=dataset/'logs'/(episode_id+'.status.json');status=json.loads(status_path.read_text())
    if status['status'] not in ['failed','generated_verified','resumed_verified','sensor_repaired_verified']:raise ValueError('Only terminal completed episodes may be repaired')
    if not json.loads((episode/'summary.json').read_text())['completed'] or (episode/'failure.json').exists():raise ValueError('Only completed physics may be sensor repaired')
    assert_no_episode_worker(episode_id,episode)
    rejected=dataset/'rejected-attempts/stale-sensor'/episode_id
    working=dataset/'sensor-repair-working'/episode_id
    if rejected.exists() or working.exists():raise FileExistsError('Original/working repair already exists; never overwrite')
    original_hashes=tree_hashes(episode)
    rejected.parent.mkdir(parents=True,exist_ok=True);working.parent.mkdir(parents=True,exist_ok=True)
    episode.rename(rejected)
    shutil.copytree(rejected,working)
    if tree_hashes(rejected)!=original_hashes or tree_hashes(working)!=original_hashes:raise RuntimeError('Copy preservation failed')
    metadata=dataset/'rejected-attempts/stale-sensor'/(episode_id+'-lineage');metadata.mkdir()
    (metadata/'original-file-sha256.json').write_text(json.dumps(original_hashes,indent=2)+'\n')
    shutil.copy2(status_path,metadata/'original-status.json')
    log=dataset/'logs'/(episode_id+'.log')
    if log.exists():shutil.copy2(log,metadata/'original-worker.log')
    status_path.write_text(json.dumps({'episode_id':episode_id,'status':'sensor_repair_pending','original_status':str(metadata/'original-status.json'),'original_preserved':str(rejected),'working_copy':str(working)},indent=2)+'\n')
    return rejected,working,metadata,original_hashes


def refresh_sparse_copies(working,frames):
    working=Path(working)
    with np.load(working/'observations/cameras.npz') as raw:sparse=set(map(int,raw['frame_ids']))
    refreshed=[]
    for frame in frames:
        if frame not in sparse:continue
        for folder,suffix in [('rgb','.png'),('masks','.png'),('depth','.npz')]:
            path=Path(folder)/(f'{frame:05d}'+suffix);shutil.copy2(working/'truth'/path,working/'observations'/path);refreshed.append(str(path))
    return refreshed


def render_repairs(working,bad_frames,gpu=3,warmup=60,attempts=20,jobs=None):
    """One static Isaac world; exact saved q/pose and audited bounded readback."""
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':gpu,'physics_gpu':gpu,'multi_gpu':False,'disable_viewport_updates':True,'fast_shutdown':True,'width':640,'height':480})
    code=0
    try:
        from pxr import Gf,UsdGeom,UsdLux,Vt
        from isaacsim.core.api import World
        import omni.replicator.core as rep
        import omni.timeline
        import carb
        world=World(stage_units_in_meters=1.,physics_dt=1/60,rendering_dt=1/60,backend='numpy',device='cpu');stage=world.stage;UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        surface=UsdGeom.Mesh.Define(stage,'/World/Surface');surface.CreateSubdivisionSchemeAttr('none');surface.CreateDoubleSidedAttr(True)
        UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(1000.)
        camera=UsdGeom.Camera.Define(stage,'/World/Camera');op=camera.AddTransformOp()
        camera.CreateFocalLengthAttr(35.);camera.CreateHorizontalApertureAttr(36.);camera.CreateVerticalApertureAttr(27.);camera.CreateClippingRangeAttr(Gf.Vec2f(.01,10.))
        product=rep.create.render_product(str(camera.GetPath()),(640,480));rgb_a=rep.AnnotatorRegistry.get_annotator('rgb');rgb_a.attach([product]);depth_a=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane');depth_a.attach([product])
        world.reset();world.pause();omni.timeline.get_timeline_interface().pause()
        all_captures=[]
        for job_index,job in enumerate(jobs or [{'working':str(working),'bad_frames':bad_frames}]):
            working=Path(job['working']);bad_frames=job['bad_frames']
            reference,trajectory,cameras=load_arrays(working);faces=reference['surface_faces']
            surface.CreateFaceVertexCountsAttr([3]*len(faces));surface.CreateFaceVertexIndicesAttr(faces.flatten().tolist());surface.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(reference['vertex_colors'].astype(np.float32)));surface.GetDisplayColorPrimvar().SetInterpolation('vertex')
            captures=[]
            for index,frame in enumerate(bad_frames):
                q=trajectory['positions_m'][frame];pose=cameras['camera_to_world_usd_row'][frame]
                surface.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(q.astype(np.float32)));op.Set(Gf.Matrix4d(*pose.ravel().tolist()))
                if index==0 and job_index==0:
                    for _ in range(warmup):world.render()
                failures=[]
                for attempt in range(1,attempts+1):
                    for _ in range(4):world.render()
                    clock=float(world.current_time)
                    settings=carb.settings.get_settings();previous=settings.get('/app/player/playSimulations')
                    settings.set_bool('/app/player/playSimulations',False)
                    try:rep.orchestrator.step(rt_subframes=2,pause_timeline=False,delta_time=0.,wait_for_render=True)
                    finally:settings.set_bool('/app/player/playSimulations',bool(previous))
                    if float(world.current_time)!=clock:raise RuntimeError('Paused renderer advanced simulation clock')
                    rgb=np.asarray(rgb_a.get_data(do_array_copy=True));depth=np.asarray(depth_a.get_data(do_array_copy=True)).squeeze()
                    if rgb.shape!=(480,640,4) and rgb.shape!=(480,640,3):failures.append({'attempt':attempt,'error':'RGB readiness '+str(rgb.shape)});continue
                    if depth.shape!=(480,640):failures.append({'attempt':attempt,'error':'Depth readiness '+str(depth.shape)});continue
                    mask=np.isfinite(depth)&(depth>.01)&(depth<10);depth=np.where(mask,depth,0).astype(np.float32)
                    check=audit_frame(rgb[:,:,:3],mask,depth,q,reference,pose,cameras['K'])
                    if check['passed']:break
                    failures.append({'attempt':attempt,'audit':check})
                else:raise RuntimeError(f'Bounded recapture failed frame{frame}: {failures}')
                stem=f'{frame:05d}';Image.fromarray(rgb[:,:,:3]).save(Path(working)/'truth/rgb'/(stem+'.png'));Image.fromarray(mask.astype(np.uint8)*255).save(Path(working)/'truth/masks'/(stem+'.png'));np.savez_compressed(Path(working)/'truth/depth'/(stem+'.npz'),depth_m=depth)
                captures.append({'frame_id':frame,'time_s':float(trajectory['times_s'][frame]),'attempts':attempt,'rejected_readbacks':failures,'state_sha256':hashlib.sha256(q.tobytes()).hexdigest(),'camera_sha256':hashlib.sha256(pose.tobytes()).hexdigest(),'audit':check})
                print(json.dumps({'recaptured_frame':frame,'attempts':attempt,'audit':check}),flush=True)
            (Path(working)/'sensor-recapture.json').write_text(json.dumps(captures,indent=2)+'\n')
            all_captures.append({'working':str(working),'captures':captures})
        return all_captures
    except BaseException:
        traceback.print_exc()
        if working is not None:(Path(working)/'sensor-recapture-failure.json').write_text(json.dumps({'traceback':traceback.format_exc(),'physics_resampled':False},indent=2))
        code=1;raise
    finally:app.close(wait_for_replicator=True,exit_code=code)


def repair(dataset,episode_id,gpu=3,isaac_python='/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python'):
    started=time.monotonic();dataset=Path(dataset);rejected,working,metadata,original_hashes=prepare_repair(dataset,episode_id)
    before=audit_episode(working);(metadata/'before-audit.json').write_text(json.dumps(before,indent=2)+'\n')
    if not before['bad_frames']:raise ValueError('No independently detected stale frames; do not replace pixels')
    # Physics/camera/reference bytes are immutable throughout this operation.
    immutable=['reference.npz','truth/trajectory.npz','truth/cameras.npz','truth/recipe.json','truth/boundaries.json']
    # SimulationApp fast shutdown terminates its process. Keep preservation and
    # final independent validation in this CPU parent, not after app.close().
    command=[isaac_python,'-u',str(Path(__file__).resolve()),'--render-working',str(working.resolve()),'--gpu',str(gpu),'--bad-frames',*map(str,before['bad_frames'])]
    with (metadata/'recapture-worker.log').open('w') as log:
        subprocess.run(command,env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES'},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    captures=json.loads((working/'sensor-recapture.json').read_text())
    return finalize_repair(dataset,episode_id,rejected,working,metadata,original_hashes,before,captures,started,gpu)


def finalize_repair(dataset,episode_id,rejected,working,metadata,original_hashes,before,captures,started,gpu):
    immutable=['reference.npz','truth/trajectory.npz','truth/cameras.npz','truth/recipe.json','truth/boundaries.json']
    refreshed=refresh_sparse_copies(working,before['bad_frames'])
    after=audit_episode(working);(metadata/'after-audit.json').write_text(json.dumps(after,indent=2)+'\n')
    if not after['passed']:raise RuntimeError('Full-frame repair audit failed')
    for name in immutable:
        if sha(working/name)!=original_hashes[name]:raise RuntimeError('Immutable physics/reference/camera changed')
    if tree_hashes(rejected)!=original_hashes:raise RuntimeError('Original rejected episode changed')
    provenance=working/'sensor-repair-provenance';provenance.mkdir();source=Path(__file__);shutil.copy2(source,provenance/source.name)
    lineage={'kind':'sensor-only exact-state recapture','original_rejected_episode':str(rejected.resolve()),'original_file_sha256':original_hashes,'bad_frames':before['bad_frames'],'captures':captures,'refreshed_sparse_files':refreshed,'immutable_file_sha256':{n:original_hashes[n] for n in immutable},'repair_source_sha256':sha(source),'original_worker_source_sha256_unchanged':True,'physics_resampled':False,'depth_threshold_m':2e-4,'seconds_elapsed':time.monotonic()-started}
    summary_path=working/'summary.json';summary=json.loads(summary_path.read_text());summary['minimum_mask_pixels']=min(f['mask_pixels'] for f in after['frames']);summary['sensor_repair']={'lineage':'sensor-repair-provenance/lineage.json','source_sha256':sha(source),'frames':before['bad_frames'],'physics_resampled':False};summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    # Frozen validator uses script-local imports. Expose that directory only
    # while calling it, including when this repair module is imported as a package.
    import sys
    source_directory=str(Path(__file__).resolve().parent);added=source_directory not in sys.path
    if added:sys.path.insert(0,source_directory)
    try:
        from validate_isaac_dataset import validate
        audit=validate(working)
    finally:
        if added:sys.path.remove(source_directory)
    (working/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    lineage['full_existing_validator']=audit;lineage['replaced_file_sha256']={str(p.relative_to(working)):sha(p) for p in sorted((working/'truth').rglob('*')) if p.is_file() and str(p.relative_to(working)) in original_hashes and sha(p)!=original_hashes[str(p.relative_to(working))]}
    (provenance/'lineage.json').write_text(json.dumps(lineage,indent=2)+'\n');(metadata/'repair-lineage.json').write_text(json.dumps(lineage,indent=2)+'\n')
    destination=dataset/'episodes'/episode_id
    if destination.exists():raise FileExistsError('Unexpected original episode path reappeared; no promotion')
    working.rename(destination)
    status={'episode_id':episode_id,'status':'sensor_repaired_verified','gpu':gpu,'seconds_elapsed':time.monotonic()-started,'original_status':str(metadata/'original-status.json'),'sensor_repair_lineage':str(destination/'sensor-repair-provenance/lineage.json'),'physics_resampled':False}
    (dataset/'logs'/(episode_id+'.status.json')).write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True);return status


def repair_batch(spec,output,gpu=3,isaac_python='/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python'):
    """Preserve each case, use one static Isaac app, validate each before promotion."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=json.loads(Path(spec).read_text());contexts=[];jobs=[]
    for row in rows:
        dataset=Path(row['dataset']);episode_id=row['episode_id'];started=time.monotonic()
        before=audit_episode(dataset/'episodes'/episode_id)
        if not before['bad_frames']:raise ValueError('Batch includes a case without failing sensor rays')
        rejected,working,metadata,original_hashes=prepare_repair(dataset,episode_id)
        (metadata/'before-audit.json').write_text(json.dumps(before,indent=2)+'\n')
        contexts.append((dataset,episode_id,rejected,working,metadata,original_hashes,before,started))
        jobs.append({'working':str(working.resolve()),'bad_frames':before['bad_frames']})
    jobfile=output/'prepared-render-jobs.json';jobfile.write_text(json.dumps(jobs,indent=2)+'\n')
    command=[isaac_python,'-u',str(Path(__file__).resolve()),'--render-jobs',str(jobfile.resolve()),'--gpu',str(gpu)]
    with (output/'recapture-worker.log').open('w') as log:
        subprocess.run(command,env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES'},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    results=[]
    for dataset,episode_id,rejected,working,metadata,original_hashes,before,started in contexts:
        shutil.copy2(output/'recapture-worker.log',metadata/'recapture-worker.log')
        captures=json.loads((working/'sensor-recapture.json').read_text())
        results.append(finalize_repair(dataset,episode_id,rejected,working,metadata,original_hashes,before,captures,started,gpu))
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n');return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path);p.add_argument('--episode');p.add_argument('--gpu',type=int,default=3);p.add_argument('--scan',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--render-jobs',type=Path);p.add_argument('--batch-spec',type=Path);p.add_argument('--render-working',type=Path);p.add_argument('--bad-frames',type=int,nargs='+');p.add_argument('--isaac-python',default='/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python')
    a=p.parse_args()
    if a.render_jobs:
        render_repairs(None,None,a.gpu,jobs=json.loads(a.render_jobs.read_text()));return
    if a.batch_spec:
        if a.output is None:p.error('--output required for batch repair')
        repair_batch(a.batch_spec,a.output,a.gpu,a.isaac_python);return
    if a.render_working:
        if not a.bad_frames:p.error('--bad-frames required for internal renderer')
        render_repairs(a.render_working,a.bad_frames,a.gpu);return
    if a.dataset is None:p.error('--dataset required')
    if a.scan:
        rows=[]
        for status in sorted((a.dataset/'logs').glob('*.status.json')):
            entry=json.loads(status.read_text());ep=a.dataset/'episodes'/entry['episode_id']
            if entry['status']=='failed' and (ep/'summary.json').exists():
                result=audit_episode(ep);rows.append({'status':entry,'audit':result});print(json.dumps({'episode':ep.name,'bad_frames':result['bad_frames']}),flush=True)
        if a.output:a.output.write_text(json.dumps(rows,indent=2)+'\n')
    else:
        if not a.episode:p.error('--episode required for repair')
        repair(a.dataset,a.episode,a.gpu,a.isaac_python)

if __name__=='__main__':main()
