"""Synchronized physical-scale videos of recorded trajectories, never interpolated."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.render_sequence import render_frame
from scripts.physical_refinement_v15.observations import Camera, box_mesh


def sphere_mesh(center, radius, color=(.85,.27,.13), rings=16, sectors=32):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Positive sphere radius required')
    phi, theta = np.meshgrid(np.linspace(0,np.pi,rings+1), np.arange(sectors)*2*np.pi/sectors, indexing='ij')
    vertices = np.stack([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)],-1).reshape(-1,3)*radius+center
    triangles=[]
    for i in range(rings):
        for j in range(sectors):
            a=i*sectors+j;b=i*sectors+(j+1)%sectors;c=a+sectors;d=b+sectors
            triangles.extend([(a,c,d),(a,d,b)])
    return dict(vertices=vertices,triangles=np.asarray(triangles),color=np.asarray(color))


def load(path):
    with np.load(path) as a:
        data={k:a[k].copy() for k in ('times_s','surface_frames')}
        if 'sphere_centers_m' in a:
            data['sphere_centers_m']=a['sphere_centers_m'].copy()
    data['metadata']=json.loads(path.with_name('complete.json').read_text())
    return data


def rigids(data,index):
    result=[]; sphere_index=0
    for collider in data['metadata']['colliders']:
        if collider['type']=='plane':
            if not np.allclose(collider.get('normal',[0,0,1]),[0,0,1]):
                raise ValueError('Only horizontal plane visualization supported')
            z=collider['point'][2]
            result.append(box_mesh([0,0,z-.001],[.45,.4,.002],[.75,.77,.79]))
        elif collider['type']=='sphere':
            if 'sphere_centers_m' not in data:
                raise ValueError('Sphere motion must be recorded, not reconstructed')
            result.append(sphere_mesh(data['sphere_centers_m'][index,sphere_index],collider['radius']))
            sphere_index+=1
        else:
            raise ValueError('Unsupported collider visualization')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('left');p.add_argument('right');p.add_argument('--output',required=True)
    p.add_argument('--left-label',required=True);p.add_argument('--right-label',required=True)
    p.add_argument('--scope',required=True);p.add_argument('--stride',type=int,default=5)
    p.add_argument('--fps',type=int,default=20)
    p.add_argument('--focal',type=float,default=550.)
    p.add_argument('--eye',type=float,nargs=3,default=(.25,-.38,.23))
    p.add_argument('--target',type=float,nargs=3,default=(0,0,-.01))
    args=p.parse_args();paths=[Path(args.left),Path(args.right)];out=Path(args.output)
    if out.exists():raise ValueError('Preserve existing video')
    if args.stride<1 or args.fps<1:raise ValueError('Positive cadence required')
    data=[load(path) for path in paths];times=data[0]['times_s']
    if not np.array_equal(times,data[1]['times_s']):
        raise ValueError('Recorded timestamps must match exactly; no interpolation')
    indices=np.arange(0,len(times),args.stride)
    if len(indices)<2:raise ValueError('At least two actual recorded frames required')
    dt=np.diff(times[indices]);step=float(dt[0])
    if step<=0 or not np.allclose(dt,step,atol=1e-12,rtol=0):raise ValueError('Uniform recorded cadence required')
    if not np.isfinite(args.focal) or args.focal<=0:raise ValueError('Positive focal length required')
    cam=Camera(tuple(args.eye),tuple(args.target),512,384,args.focal)
    from PIL import Image,ImageDraw
    import imageio_ffmpeg
    out.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio_ffmpeg.write_frames(str(out),(1024,448),fps=args.fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','20'])
    writer.send(None);samples=[]
    try:
        for ordinal,index in enumerate(indices):
            frame=Image.new('RGB',(1024,448),'white');draw=ImageDraw.Draw(frame)
            draw.text((8,3),args.scope,fill='black')
            draw.text((8,19),f'Simulated time {times[index]:.3f}s | {1/(step*args.fps):.1f}x slow motion | physical scale',fill='black')
            for side,(item,label) in enumerate(zip(data,[args.left_label,args.right_label])):
                rgb=render_frame(item['surface_frames'][index],cam,rigids(item,index),time_s=float(times[index]))['rgb']
                frame.paste(Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)),(side*512,64))
                draw.text((side*512+8,43),label,fill='black')
            draw.line([(512,40),(512,448)],fill=(100,100,100))
            writer.send(np.asarray(frame))
            if ordinal in {0,len(indices)//3,2*len(indices)//3,len(indices)-1}:samples.append(frame.copy())
    finally:writer.close()
    sheet=Image.new('RGB',(1024,448*len(samples)),'white')
    for i,frame in enumerate(samples):sheet.paste(frame,(0,i*448))
    sheet.save(out.with_suffix('.png'))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    metadata=dict(scope=args.scope,labels=[args.left_label,args.right_label],
                  trajectory_sha256={str(path):sha(path) for path in paths},
                  source_sha256={str(path):sha(path) for path in [Path(__file__),Path('scripts/compositional_dynamics_v17/render_sequence.py'),Path('scripts/physical_refinement_v15/observations.py')]},
                  video_sha256=sha(out),actual_times_s=times[indices].tolist(),frame_indices=indices.tolist(),
                  fps=args.fps,slow_motion_factor=1/(step*args.fps),camera=vars(cam),
                  note='Same fixed camera and material texture. No time alignment, trajectory interpolation or displacement magnification. Labels state comparison scope; this renderer does not certify physical accuracy.')
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(video=str(out),frames=len(indices))))


if __name__=='__main__':main()
