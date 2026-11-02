"""Known-appearance development rendering of saved physical time steps.
No physical interpolation, camera fitting or displacement exaggeration.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scripts.physical_refinement_v15.observations import Camera,box_mesh,render_scene


def material_surface(frame):
    a=np.asarray(frame,float)
    if a.ndim!=4 or a.shape[0]!=6 or a.shape[1]!=a.shape[2] or a.shape[-1]!=3 or not np.isfinite(a).all():
        raise ValueError('Finite six-face sampled physical surface required')
    n=a.shape[1];uv=np.stack(np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n),indexing='ij'),-1).reshape(-1,2)
    q=[];tri=[]
    for face in range(6):
        x=np.empty((n*n,3));axis=face//2;x[:,axis]=face%2;x[:,[j for j in range(3)if j!=axis]]=uv;q.append(x)
        offset=face*n*n
        for i in range(n-1):
            for j in range(n-1):
                k=offset+i*n+j;tri.extend([(k,k+n,k+n+1),(k,k+n+1,k+1)])
    q=np.concatenate(q);pattern=.5+.5*np.cos(8*np.pi*q[:,0])*np.cos(6*np.pi*q[:,1])*np.cos(6*np.pi*q[:,2])
    color=np.array([.12,.42,.70])[None,:]*(.55+.45*pattern[:,None])+.14*q
    return dict(vertices=a.reshape(-1,3),triangles=np.asarray(tri),color=np.clip(color,0,1))


def render_frame(surface,camera,rigid_meshes=(),*,time_s=None):
    body=render_scene([material_surface(surface)],camera)
    rigid=render_scene(list(rigid_meshes),camera)
    visible=body['depth']<rigid['depth']
    rgb=np.where(visible[...,None],body['rgb'],rigid['rgb'])
    return dict(time_s=time_s,rgb=rgb,depth=np.minimum(body['depth'],rigid['depth']),body_visible=visible,
                body_visible_depth=np.where(visible,body['depth'],np.inf),camera=body['camera'])


def main():
    p=argparse.ArgumentParser();p.add_argument('trajectory');p.add_argument('--output',required=True);p.add_argument('--stride',type=int,default=5);p.add_argument('--fps',type=int,default=20);args=p.parse_args()
    path=Path(args.trajectory);out=Path(args.output)
    if out.exists():raise ValueError('Preserve previous video')
    if args.stride<1 or args.fps<1:raise ValueError('Positive sampling stride/fps required')
    with np.load(path)as a:times=a['times_s'].copy();surfaces=a['surface_frames'].copy()
    indices=np.arange(0,len(times),args.stride)
    if len(indices)<2:raise ValueError('At least two actual physical frames required')
    elapsed=np.diff(times[indices]);step=float(elapsed[0])
    if not np.allclose(elapsed,step,atol=1e-12,rtol=0):raise ValueError('Uniform saved-frame cadence required; no interpolation')
    meta=json.loads(path.with_name('complete.json').read_text());rigids=[]
    for c in meta['colliders']:
        if c['type']!='plane':raise ValueError('Prototype renderer supports the declared plane only')
        z=c['point'][2];rigids.append(box_mesh([0,0,z-.001],[.32,.25,.002],[.75,.77,.79]))
    cam=Camera((.23,-.35,.20),(0,0,-.01),512,384,550.)
    from PIL import Image,ImageDraw
    import imageio_ffmpeg
    out.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio_ffmpeg.write_frames(str(out),(512,432),fps=args.fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','20']);writer.send(None)
    preview=None
    try:
        for i in indices:
            rgb=render_frame(surfaces[i],cam,rigids,time_s=float(times[i]))['rgb'];frame=Image.new('RGB',(512,432),'white');frame.paste(Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)),(0,48));draw=ImageDraw.Draw(frame)
            draw.text((8,4),f"DEVELOPMENT physics prototype | {meta['case']} | n={meta['n']}",fill='black')
            draw.text((8,20),f"Simulated t={times[i]:.3f}s | {1/(step*args.fps):.1f}x slow motion | physical scale",fill='black')
            writer.send(np.asarray(frame));preview=frame if preview is None else preview
    finally:writer.close()
    preview_path=out.with_suffix('.png');preview.save(preview_path)
    report=dict(trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),video_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frame_indices=indices.tolist(),actual_times_s=times[indices].tolist(),fps=args.fps,slow_motion_factor=1/(step*args.fps),scope='Development trajectory from actual integrator states; fixed common camera, material texture, physical geometry. No prediction comparison or numerical qualification claim.',camera=dict(eye=cam.eye,target=cam.target,width=cam.width,height=cam.height,focal=cam.focal),preview=str(preview_path))
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(video=str(out),frames=len(indices),slow_motion_factor=report['slow_motion_factor'])))

if __name__=='__main__':main()
