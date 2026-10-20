"""Actual orbiting-camera RGB montage; fixed display crop, no motion amplification."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def single_preview(episode,output):
    episode=Path(episode);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((episode/'summary.json').read_text())
    low=np.array([640,480]);high=np.zeros(2,dtype=int)
    for path in sorted((episode/'truth/masks').glob('*.png')):
        y,x=np.where(np.asarray(Image.open(path))>0)
        low=np.minimum(low,[x.min(),y.min()]);high=np.maximum(high,[x.max()+1,y.max()+1])
    crop=tuple(map(int,[*np.maximum(low-20,0),*np.minimum(high+20,[640,480])]))
    frames=[];writer=imageio_ffmpeg.write_frames(str(output/'simulation.mp4'),(640,528),fps=20,
        codec='libx264',output_params=['-crf','18','-movflags','+faststart'],macro_block_size=2)
    writer.send(None)
    try:
        for i in range(61):
            image=Image.open(episode/'truth/rgb'/f'{i:05d}.png').convert('RGB').crop(crop)
            factor=min(640/image.width,480/image.height)
            image=image.resize((round(image.width*factor),round(image.height*factor)),Image.Resampling.LANCZOS)
            canvas=Image.new('RGB',(640,528),'white');canvas.paste(image,((640-image.width)//2,24+(480-image.height)//2))
            draw=ImageDraw.Draw(canvas)
            draw.text((8,6),f'{episode.name} | E={summary["material"]["youngs_modulus_Pa"]/1000:g} kPa | '+summary['force_type']+' / '+summary['boundary_type'],fill='black')
            draw.text((8,510),f't={i/20:.2f} s | actual orbiting-camera RGB | fixed crop | no motion amplification',fill='black')
            writer.send(np.asarray(canvas).tobytes());frames.append(canvas)
    finally:writer.close()
    gif=[f.convert('P',palette=Image.Palette.ADAPTIVE,colors=96,dither=Image.Dither.NONE) for f in frames[::2]]
    gif[0].save(output/'preview.gif',save_all=True,append_images=gif[1:],duration=100,loop=0)
    frames[30].save(output/'poster.png')
    decoder=imageio_ffmpeg.read_frames(str(output/'simulation.mp4'));next(decoder)
    assert sum(1 for _ in decoder)==61
    (output/'video.json').write_text(json.dumps({'episode':episode.name,'frames':61,'fps':20,
        'fixed_crop_xyxy':crop,'deformation_amplification':1,
        'mp4_sha256':hashlib.sha256((output/'simulation.mp4').read_bytes()).hexdigest()},indent=2)+'\n')


def make_preview(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # A paired material triplet per force family, with all four boundary types.
    ids=[16*material+5*load for load in range(4) for material in range(3)]
    episodes=[root/'episodes'/f'e{i:04d}' for i in ids]
    summaries=[json.loads((e/'summary.json').read_text()) for e in episodes]
    poses=[np.load(e/'truth/cameras.npz')['camera_to_world_usd_row'] for e in episodes]
    for row in range(4):
        for col in [1,2]:np.testing.assert_array_equal(poses[row*3],poses[row*3+col])
    # One crop across every frame/episode: camera changes are real, crop never tracks.
    low=np.array([640,480]);high=np.zeros(2,dtype=int)
    for episode in episodes:
        for path in sorted((episode/'truth/masks').glob('*.png')):
            y,x=np.where(np.asarray(Image.open(path))>0)
            low=np.minimum(low,[x.min(),y.min()]);high=np.maximum(high,[x.max()+1,y.max()+1])
    low=np.maximum(low-15,0);high=np.minimum(high+15,[640,480]);crop=tuple(map(int,[*low,*high]))
    panel=(320,240);size=(960,1104);frames=[]
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    except OSError:font=ImageFont.load_default()
    writer=imageio_ffmpeg.write_frames(str(output/'simulation.mp4'),size,fps=20,codec='libx264',
        output_params=['-crf','18','-movflags','+faststart'],macro_block_size=2)
    writer.send(None)
    try:
        for frame in range(61):
            canvas=Image.new('RGB',size,'white');draw=ImageDraw.Draw(canvas)
            draw.text((12,6),f'Isaac Sim | orbiting camera | t={frame/20:.2f} s | paired materials in each row',fill='black',font=font)
            for j,(episode,summary) in enumerate(zip(episodes,summaries)):
                row,col=divmod(j,3);x=col*320;y=28+row*264
                label=f'{summary["episode_id"]}  E={summary["material"]["youngs_modulus_Pa"]/1000:g} kPa'
                draw.text((x+7,y),label,fill='black',font=font)
                image=Image.open(episode/'truth/rgb'/f'{frame:05d}.png').convert('RGB').crop(crop)
                # Aspect ratio maintained with padding; no varying camera or object scaling.
                image.thumbnail(panel,Image.Resampling.LANCZOS)
                canvas.paste(image,(x+(320-image.width)//2,y+20+(240-image.height)//2))
                draw.text((x+7,y+244),summary['force_type']+' / '+summary['boundary_type'],fill='black',font=font)
            draw.text((12,1088),'Actual RGB; same fixed crop; no deformation amplification. Full 48-episode dataset has RGB, depth, masks and truth.',fill='black',font=font)
            writer.send(np.asarray(canvas).tobytes());frames.append(canvas)
    finally:writer.close()
    gif=[]
    for frame in frames[::2]:
        small=frame.resize((720,828),Image.Resampling.LANCZOS)
        gif.append(small.convert('P',palette=Image.Palette.ADAPTIVE,colors=96,dither=Image.Dither.NONE))
    gif[0].save(output/'preview.gif',save_all=True,append_images=gif[1:],duration=100,loop=0)
    frames[30].save(output/'poster.png')
    decoder=imageio_ffmpeg.read_frames(str(output/'simulation.mp4'));metadata=next(decoder)
    count=sum(1 for _ in decoder);assert count==61
    (output/'video.json').write_text(json.dumps({'episodes':[e.name for e in episodes],
        'source':'Actual dense Isaac RGB','fixed_crop_xyxy':crop,'display_panel_pixels':panel,
        'frames':count,'fps':20,'deformation_amplification':1,'decoded_size':metadata['size'],
        'mp4_sha256':hashlib.sha256((output/'simulation.mp4').read_bytes()).hexdigest()},indent=2)+'\n')
    single_preview(root/'episodes/e0007',output/'single')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset');parser.add_argument('--output',required=True)
    args=parser.parse_args();make_preview(args.dataset,args.output)
