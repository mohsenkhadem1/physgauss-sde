"""Actual Isaac RGB video exports with exact input/source manifests.

No simulator rerun, synthesized motion, frame interpolation or displacement
amplification. Optional fixed spatial crop enlarges body and motion equally.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg

BC={'point_pin':'Point pin','fixed_patch':'Fixed patch','two_patches':'Two patches','moving_patch':'Moving patch'}
FORCE={'push_pulse':'Push pulse','pull_hold':'Pull hold','shear_oscillation':'Shear oscillation','twist_couple':'Twist couple'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):return {'path':str(path),'sha256':sha(path)}
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def episode_inputs(dataset,eid):
    episode=dataset/'episodes'/eid
    if not json.loads((episode/'validation.json').read_text()).get('passed'):raise ValueError('Only independently verified episodes')
    case=json.loads((episode/'truth/recipe.json').read_text())
    with np.load(episode/'truth/trajectory.npz') as a:q=a['positions_m'];times=a['times_s']
    with np.load(episode/'reference.npz') as a:rest=a['rest_points_m']
    with np.load(episode/'truth/cameras.npz') as a:camera_times=a['times_s']
    if not np.array_equal(times,camera_times):raise ValueError('Camera/physics sample mismatch')
    if len(times)!=61 or not np.allclose(times,np.arange(61)/20,atol=1e-12):raise ValueError('Expected61actual20Hz frames')
    images=[episode/'truth/rgb'/f'{i:05d}.png' for i in range(len(times))]
    summary=json.loads((episode/'summary.json').read_text())
    provenance={name:record(episode/'provenance'/name) for name in summary['source_sha256']}
    return case,times,images,float(np.linalg.norm(q-rest,axis=-1).max()),{
        'episode_id':eid,'split':case['split'],'recipe':record(episode/'truth/recipe.json'),
        'declared_recipe_sha256':case['recipe_sha256'],'physics_trajectory':record(episode/'truth/trajectory.npz'),
        'camera_trajectory':record(episode/'truth/cameras.npz'),'reference':record(episode/'reference.npz'),
        'validation':record(episode/'validation.json'),'generation_source':provenance,
        'sensor_repair_metadata':[record(p) for p in episode.glob('sensor*.json')],
        'input_frames':[record(p) for p in images]}


def scenarios(dataset,output,ids):
    sources=[episode_inputs(dataset,eid) for eid in ids]
    if len(sources)!=4:raise ValueError('Four predeclared scenario IDs required')
    output.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio_ffmpeg.write_frames(str(output),(1280,1152),fps=20,codec='libx264',pix_fmt_in='rgb24',
        output_params=['-crf','19','-preset','fast','-movflags','+faststart'])
    writer.send(None)
    for frame in range(61):
        canvas=Image.new('RGB',(1280,1152),'#111722');draw=ImageDraw.Draw(canvas)
        draw.text((16,5),'Actual Isaac Sim: four sealed TEST scenarios',font=font(22),fill='white')
        draw.text((16,34),'3 s simulated interval | 61 captured frames at 20 fps | native pixels | actual camera orbits | no motion scaling',font=font(16),fill='#c8d1df')
        for j,(case,times,images,maximum,metadata) in enumerate(sources):
            x=(j%2)*640;y=64+(j//2)*544
            draw.text((x+12,y+5),f"{case['episode_id']} | E={case['material']['youngs_modulus_Pa']/1000:.2f} kPa | {BC[case['boundary_type']]}",font=font(19),fill='white')
            draw.text((x+12,y+32),f"{FORCE[case['force_type']]} | max nodal displacement {maximum*1000:.2f} mm | t={times[frame]:.2f} s",font=font(16),fill='#c8d1df')
            image=Image.open(images[frame]).convert('RGB')
            if image.size!=(640,480):raise ValueError('Native image dimensions changed')
            canvas.paste(image,(x,y+64))
        writer.send(np.asarray(canvas))
        if frame==30:canvas.save(output.with_suffix('.poster.png'))
    writer.close()
    manifest={'kind':'Actual Isaac captured RGB montage','episode_ids':ids,'physical_interval_s':[0,3],
        'captured_frames_per_episode':61,'video_fps':20,'encoded_duration_s':61/20,
        'spatial_transform':'Four native640x480images pasted into2x2canvas; only added header strips',
        'frame_interpolation':False,'displacement_amplification':False,'camera_motion':'Actual saved per-frame simulator cameras',
        'source_script':record(Path(__file__)),'imageio_ffmpeg_version':importlib.metadata.version('imageio-ffmpeg'),
        'cases':[s[4] for s in sources],'video':record(output)}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def probe(dataset,pool,output,eid='z0003',action_id='poke_3'):
    private=pool/'private'/eid
    if not json.loads((private/'validation.json').read_text()).get('passed'):raise ValueError('Only verified actual probe')
    case=json.loads((private/'truth/recipe.json').read_text());menu=json.loads((pool/'public'/eid/'action-menu.json').read_text())
    action=next(a for a in menu['candidates'] if a['action_id']==action_id)
    if action['force_type_index']!=2 or not np.array_equal(action['peak_force_N'],[0,0,.02]):raise ValueError('Expectedpredeclared+zoscil')
    with np.load(private/action_id/'trajectory.npz') as a:
        q=a['positions_m'];times=a['times_s'];step_times=a['force_step_midpoints_s'];forces=a['force_step_commands_N'];targets=a['boundary_target_positions_m']
    with np.load(private/'reference.npz') as a:rest=a['rest_points_m']
    directory=pool/'outcomes'/eid/action_id/'observations'
    with np.load(directory/'cameras.npz') as a:
        np.testing.assert_allclose(a['camera_to_world_usd_row'],np.broadcast_to(a['camera_to_world_usd_row'][0],(31,4,4)),atol=1e-12)
        np.testing.assert_allclose(a['times_s'],times,atol=1e-12)
    if len(times)!=31 or not np.allclose(times,np.arange(31)/10,atol=1e-12):raise ValueError('Expectedactual10Hz samples')
    image_paths=[directory/'rgb'/f'{i:05d}.png' for i in range(31)]
    mask=np.asarray(Image.open(directory/'masks/00000.png'))>0;ys,xs=np.where(mask)
    cx=int(round((xs.min()+xs.max())/2));cy=int(round((ys.min()+ys.max())/2));left=max(0,min(320,cx-160));top=max(0,min(240,cy-120));crop=(left,top,left+320,top+240)
    for frame in range(31):
        m=np.asarray(Image.open(directory/'masks'/f'{frame:05d}.png'))>0;y,x=np.where(m)
        if x.min()<crop[0] or x.max()>=crop[2] or y.min()<crop[1] or y.max()>=crop[3]:raise ValueError('Body leaves fixed illustrative crop')
    maximum=float(np.linalg.norm(q-rest,axis=-1).max());motion=float(np.linalg.norm(targets-targets[:1],axis=-1).max())
    output.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio_ffmpeg.write_frames(str(output),(1280,720),fps=10,codec='libx264',pix_fmt_in='rgb24',
        output_params=['-crf','18','-preset','fast','-movflags','+faststart']);writer.send(None)
    # Plot the actual960Hz logged command, not a reconstructed force head.
    xx=60+step_times/3*1160;yy=654-forces[:,0,2]/.02*35
    force_line=list(zip(xx.tolist(),yy.tolist()))
    for frame,t in enumerate(times):
        canvas=Image.new('RGB',(1280,720),'#111722');draw=ImageDraw.Draw(canvas)
        draw.text((14,4),f'Actual Isaac reset probe {eid}: +world z oscillation | t={t:.1f} s',font=font(21),fill='white')
        draw.text((14,32),f"E={case['material']['youngs_modulus_Pa']/1000:.2f} kPa | {BC[case['boundary_type']]} | 0.020 N command coefficient | max nodal motion {maximum*1000:.2f} mm",font=font(17),fill='#c8d1df')
        draw.text((14,57),f'Independent rest reset; same supports/material/handles. Prescribed support motion reaches {motion*1000:.2f} mm.',font=font(16),fill='#c8d1df')
        draw.text((14,80),'Left: native fixed camera. Right: fixed 2x spatial crop; body and motion enlarged equally, no deformation scaling.',font=font(15),fill='#c8d1df')
        image=Image.open(image_paths[frame]).convert('RGB');canvas.paste(image,(0,112));canvas.paste(image.crop(crop).resize((640,480),Image.Resampling.LANCZOS),(640,112))
        draw.text((14,595),'Logged handle0 Fz at 960 Hz; handle1 and other force components zero',font=font(15),fill='#c8d1df')
        draw.line((60,654,1220,654),fill='#65738b',width=1);draw.line(force_line,fill='#66d3ff',width=3)
        draw.line((60+t/3*1160,615,60+t/3*1160,694),fill='#ffcf6c',width=2)
        draw.text((8,615),'+.02N',font=font(12),fill='white');draw.text((8,678),'-.02N',font=font(12),fill='white')
        for value in [0,1,2,3]:draw.text((60+value/3*1160-8,697),str(value)+'s',font=font(12),fill='white')
        writer.send(np.asarray(canvas))
        if frame==15:canvas.save(output.with_suffix('.poster.png'))
    writer.close();summary=json.loads((private/'summary.json').read_text())
    manifest={'kind':'Actual Isaac calibrated repeat-experiment RGB','episode_id':eid,'split':case['split'],'action':action,
        'physical_interval_s':[0,3],'captured_frames':31,'video_fps':10,'encoded_duration_s':3.1,
        'camera':'Actual fixed saved camera; independent reset at rest, not online continuation',
        'spatial_transform':{'left':'native640x480','right':'fixed320x240crop uniformly2x','crop_xyxy':crop},
        'frame_interpolation':False,'displacement_amplification':False,'maximum_actual_node_displacement_m':maximum,
        'maximum_actual_prescribed_support_motion_m':motion,'recipe':record(private/'truth/recipe.json'),
        'declared_recipe_sha256':case['recipe_sha256'],'camera_metadata':record(directory/'cameras.npz'),
        'force_and_state_log':record(private/action_id/'trajectory.npz'),'known_action_menu':record(pool/'public'/eid/'action-menu.json'),
        'validation':record(private/'validation.json'),'source_script':record(Path(__file__)),
        'generation_source':{name:record(private/'provenance'/name) for name in summary['source_sha256']},
        'input_frames':[record(p) for p in image_paths],'video':record(output)}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--pool',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();montage=scenarios(a.dataset,a.output/'isaac-test-scenarios.mp4',['z0000','z0005','z0010','z0015']);single=probe(a.dataset,a.pool,a.output/'isaac-test-calibrated-probe.mp4')
    print(json.dumps({'montage':montage['video'],'probe':single['video']}))

if __name__=='__main__':main()
