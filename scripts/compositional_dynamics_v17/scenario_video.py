"""Single recorded dynamic scene with explicit support annotations and provenance."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17.comparison_video import load,rigids
from scripts.compositional_dynamics_v17.render_sequence import render_frame
from scripts.physical_refinement_v15.observations import Camera


def main():
    p=argparse.ArgumentParser();p.add_argument('trajectory');p.add_argument('--output',required=True)
    p.add_argument('--title',required=True);p.add_argument('--stride',type=int,default=2);p.add_argument('--fps',type=int,default=20)
    p.add_argument('--eye',type=float,nargs=3,default=(.24,-.32,.20));p.add_argument('--target',type=float,nargs=3,default=(.06,0,.03));p.add_argument('--focal',type=float,default=850.)
    args=p.parse_args();path=Path(args.trajectory);out=Path(args.output)
    if out.exists():raise ValueError('Preserve previous video')
    data=load(path)
    with np.load(path) as raw:
        if 'tool_poses_m_rad' in raw:data['tool_poses_m_rad']=raw['tool_poses_m_rad'].copy()
    times=data['times_s'];indices=np.arange(0,len(times),args.stride)
    if len(indices)<2 or args.stride<1 or args.fps<1:raise ValueError('Positive cadence and >=2 frames required')
    delta=np.diff(times[indices]);dt=float(delta[0])
    if dt<=0 or not np.allclose(delta,dt,rtol=0,atol=1e-12):raise ValueError('No interpolation; uniform saved cadence required')
    cam=Camera(tuple(args.eye),tuple(args.target),768,512,args.focal)
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    font=ImageFont.truetype('DejaVuSans.ttf',14)
    out.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio_ffmpeg.write_frames(str(out),(768,592),fps=args.fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','19']);writer.send(None)
    selected=[];eye,R=cam.pose()
    try:
        for ordinal,index in enumerate(indices):
            rgb=render_frame(data['surface_frames'][index],cam,rigids(data,index),time_s=float(times[index]))['rgb']
            frame=Image.new('RGB',(768,592),'white');frame.paste(Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)),(0,80));draw=ImageDraw.Draw(frame)
            draw.text((9,4),args.title,font=font,fill='black')
            draw.text((9,24),f't = {times[index]:.3f} s | {1/(dt*args.fps):.1f}x slow motion | physical scale',font=font,fill='black')
            draw.text((9,44),'DEVELOPMENT simulation; numerical reference not yet qualified',font=font,fill='black')
            if data['metadata'].get('support_specs'):
                moving='tool_poses_m_rad' in data
                draw.text((9,64),'Crosses mark moving compliant tools' if moving else 'Crosses mark fixed compliant end supports; they are not rigid clamps',font=font,fill='black')
                for side,support in enumerate(data['metadata']['support_specs']):
                    origin=np.asarray(support['face_origin_m'])+(data['tool_poses_m_rad'][index,side,:3] if moving else 0.)
                    q=R@(origin-eye);uv=q[:2]/q[2]*cam.focal+[cam.width/2,cam.height/2+80]
                    x,y=uv;draw.line([(x-7,y),(x+7,y)],fill=(20,20,20),width=2);draw.line([(x,y-7),(x,y+7)],fill=(20,20,20),width=2)
            writer.send(np.asarray(frame))
            if ordinal in {0,len(indices)//3,2*len(indices)//3,len(indices)-1}:selected.append(frame.copy())
    finally:writer.close()
    gallery=Image.new('RGB',(1536,1184),'white')
    for i,frame in enumerate(selected):gallery.paste(frame,((i%2)*768,(i//2)*592))
    gallery.save(out.with_suffix('.png'))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=dict(title=args.title,trajectory=str(path),trajectory_sha256=sha(path),video_sha256=sha(out),source_sha256={str(p):sha(p) for p in [Path(__file__),Path('scripts/compositional_dynamics_v17/comparison_video.py'),Path('scripts/compositional_dynamics_v17/render_sequence.py'),Path('scripts/physical_refinement_v15/observations.py')]},frame_indices=indices.tolist(),actual_times_s=times[indices].tolist(),slow_motion_factor=1/(dt*args.fps),fps=args.fps,camera=vars(cam),physical_scale=True,trajectory_completed=data['metadata']['completed'],scope='Recorded integrator states only. Support crosses are annotations at recorded tool origins, not added collision geometry. No interpolation, learned prediction or qualified physical-reference claim.')
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(video=str(out),frames=len(indices))))


if __name__=='__main__':main()
