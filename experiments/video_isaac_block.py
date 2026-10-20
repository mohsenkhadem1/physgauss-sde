"""Video from actual Isaac RGB, with an optional matched material comparison.

Both panels use the same crop. Optional mesh edges are projected from measured
simulation nodes, with no displacement amplification or frame interpolation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw


def project(points, camera):
    pose=np.asarray(camera['camera_to_world_usd_row_matrix'])
    xyz=np.concatenate([points,np.ones(points.shape[:-1]+(1,))],axis=-1)@np.linalg.inv(pose)
    k=np.asarray(camera['K']);depth=-xyz[...,2]
    return np.stack([k[0,0]*xyz[...,0]/depth+k[0,2],-k[1,1]*xyz[...,1]/depth+k[1,2]],axis=-1)


def video(episodes, output, grid=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[]
    for root in map(Path,episodes):
        record={'root':root,'summary':json.loads((root/'summary.json').read_text()),
                'spec':json.loads((root/'truth/boundaries.json').read_text()),
                'camera':json.loads((root/'observations/camera.json').read_text())}
        with np.load(root/'truth/trajectory.npz') as raw:
            record.update(q=raw['positions_m'],times=raw['times_s'],forces=raw['force_step_commands_N'])
        with np.load(root/'reference.npz') as ref:record['faces']=ref['surface_faces']
        record['uv']=project(record['q'],record['camera']);records.append(record)
    for r in records[1:]:
        assert r['camera']==records[0]['camera'],'Comparison cameras differ'
        assert r['spec']==records[0]['spec'],'Comparison boundaries/forces differ'
        np.testing.assert_array_equal(r['times'],records[0]['times'])
        np.testing.assert_array_equal(r['forces'],records[0]['forces'])
    uv=np.concatenate([r['uv'].reshape(-1,2) for r in records])
    low=np.maximum(np.floor(uv.min(0)-25),[0,0]).astype(int)
    high=np.minimum(np.ceil(uv.max(0)+25),[640,480]).astype(int)
    crop=tuple(map(int,[*low,*high]));panel=(640,480);size=(640*len(records),540)
    fps=records[0]['summary']['observation_fps'];frames=[]
    writer=imageio_ffmpeg.write_frames(str(output/'simulation.mp4'),size,fps=fps,codec='libx264',
        output_params=['-crf','18','-movflags','+faststart'],macro_block_size=2)
    writer.send(None)
    try:
        for i,t in enumerate(records[0]['times']):
            canvas=Image.new('RGB',size,'white');draw=ImageDraw.Draw(canvas)
            for j,r in enumerate(records):
                frame=Image.open(r['root']/'observations'/f'rgb-{i:04d}.png').convert('RGBA')
                if grid:
                    overlay=Image.new('RGBA',frame.size,(0,0,0,0));lines=ImageDraw.Draw(overlay)
                    faces=r['faces'];tri=r['q'][i,faces]
                    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                    eye=np.asarray(r['camera']['camera_to_world_usd_row_matrix'])[3,:3]
                    visible=np.einsum('ij,ij->i',normal,eye-tri.mean(1))>0
                    for face in faces[visible]:
                        xy=[tuple(v) for v in r['uv'][i,face]]
                        lines.line(xy+[xy[0]],fill=(35,45,55,105),width=1)
                    frame=Image.alpha_composite(frame,overlay)
                frame=frame.convert('RGB').crop(crop).resize(panel,Image.Resampling.LANCZOS)
                canvas.paste(frame,(640*j,30))
                label=f'Young\'s modulus {r["summary"]["youngs_modulus_Pa"]/1000:g} kPa'
                draw.text((640*j+15,9),label,fill='black')
            draw.text((15,518),f'Actual Isaac motion | t={t:.2f} s | same crop; no deformation amplification'+(' | measured mesh overlay' if grid else ''),fill='black')
            writer.send(np.asarray(canvas).tobytes());frames.append(canvas)
    finally:writer.close()
    gif_frames=[]
    for frame in frames[::2]:
        thumbnail=frame.copy();thumbnail.thumbnail((800,540),Image.Resampling.LANCZOS)
        gif_frames.append(thumbnail.convert('P',palette=Image.Palette.ADAPTIVE,colors=96,dither=Image.Dither.NONE))
    gif_frames[0].save(output/'preview.gif',save_all=True,append_images=gif_frames[1:],duration=round(2000/fps),loop=0)
    frames[0].save(output/'poster.png')
    # Independent decoder verifies that every recorded frame reached the MP4.
    reader=imageio_ffmpeg.read_frames(str(output/'simulation.mp4'));metadata=next(reader)
    count=sum(1 for _ in reader);assert count==len(frames)
    info={'episodes':episodes,'fps':fps,'frames':count,'crop_xyxy':crop,'grid_overlay':grid,
          'deformation_amplification':1,'gif_frame_stride':2,'gif_maximum_width':800,
          'material_and_boundary_settings':[r['summary'] for r in records],
          'mp4_sha256':hashlib.sha256((output/'simulation.mp4').read_bytes()).hexdigest()}
    (output/'video.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes',nargs='+');parser.add_argument('--output',required=True)
    parser.add_argument('--grid',action='store_true')
    args=parser.parse_args();video(args.episodes,args.output,args.grid)
