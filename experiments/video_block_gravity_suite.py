"""Orbit videos from recorded Isaac nodes; true scale, no simulated-frame interpolation.

Left: actual world configuration, camera follows the mass centre. Right: the
same recorded configuration aligned to rest by a proper mass-weighted rigid fit.
Only playback is slowed. This renderer is not a prediction or native RGB capture.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def align_to_rest(rest,positions,weights):
    w=np.asarray(weights,dtype=float);w=w/w.sum();a=np.asarray(rest);b=np.asarray(positions)
    ca=w@a;cb=w@b;U,_,Vt=np.linalg.svd((a-ca).T@(w[:,None]*(b-cb)))
    fix=np.eye(3);fix[-1,-1]=np.linalg.det(U@Vt);R=U@fix@Vt
    aligned=(b-cb)@R.T+ca
    return aligned,float(np.sqrt(np.sum(w[:,None]*(aligned-a)**2)/3))


def volume_weights(rest,tets):
    tri=rest[tets];v=np.linalg.det(np.swapaxes(tri[:,1:]-tri[:,:1],1,2))/6
    if np.any(v<=0):raise ValueError('Positive reference volumes required')
    w=np.zeros(len(rest));np.add.at(w,tets.ravel(),np.repeat(v/4,4));return w/w.sum()


def camera(points,target,phase,origin,scale):
    az=np.deg2rad(-65+45*phase);el=np.deg2rad(28)
    forward=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right=np.cross([0,0,1],forward);right/=np.linalg.norm(right);up=np.cross(forward,right)
    delta=points-target;uv=delta@np.stack([right,-up],axis=1)*scale+origin
    return uv,delta@forward,forward


def draw_panel(canvas,q,rest,faces,fixed,target,phase,box,floor_z=None):
    x0,y0,x1,y1=box;panel=Image.new('RGB',(x1-x0,y1-y0),'#f6f8fa');draw=ImageDraw.Draw(panel)
    origin=np.array([(x1-x0)/2,(y1-y0)/2+10]);scale=(x1-x0)/.39
    uv,depth,view=camera(q,target,phase,origin,scale)
    if floor_z is not None:
        floor=np.array([[-.23,-.18,floor_z],[.23,-.18,floor_z],[.23,.18,floor_z],[-.23,.18,floor_z]])
        fuv,_,_=camera(floor,target,phase,origin,scale);draw.polygon([tuple(p) for p in fuv],fill='#dce1e4',outline='#9ca8af')
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    ru,_,_=camera(rest,target,phase,origin,scale)
    for e in edges:draw.line([tuple(p) for p in ru[e]],fill='#c5cbd0',width=1)
    # Reference-coordinate striping is attached to material; no deformation
    # magnitude amplification or re-colouring by an inferred field.
    col=(rest[faces].mean(1)[:,0]-rest[:,0].min())/np.ptp(rest[:,0])
    triangle=q[faces];normal=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30)
    light=np.array([.3,-.5,1.]);light/=np.linalg.norm(light);shade=.58+.42*np.abs(normal@light)
    for i in np.argsort(depth[faces].mean(1)):
        base=np.array([48+135*col[i],127+45*col[i],185-95*col[i]])
        rgb=tuple(np.clip(base*shade[i],0,255).astype(int));draw.polygon([tuple(p) for p in uv[faces[i]]],fill=rgb,outline='#344c5a')
    for p in uv[fixed]:draw.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill='#e65020')
    canvas.paste(panel,(x0,y0))


def load(root):
    root=Path(root);paths=list(root.glob('trajectory.npz'))+list(root.glob('truth/trajectory.npz'))
    if len(paths)!=1:raise ValueError('One recorded trajectory required at '+str(root))
    with np.load(paths[0],allow_pickle=False) as a:d={k:a[k] for k in a.files}
    # The native suite emits these explicit fields; no invented trajectories.
    for k in ['positions_m','times_s','rest_m','tets','faces','fixed_node_ids']:
        if k not in d:raise ValueError('Missing '+k)
    if len(d['positions_m'])!=len(d['times_s']) or not np.isfinite(d['positions_m']).all():raise ValueError('Aligned finite recorded frames required')
    summaries=list(root.glob('summary.json'))+list(root.glob('analysis.json'))
    meta=json.loads(summaries[0].read_text()) if summaries else {}
    if meta.get('schema')!='isaac-block-gravity-case-v0.1' or not meta.get('completed') or 'native_verification' not in meta:raise ValueError('Completed native Isaac case provenance required')
    if hashlib.sha256(paths[0].read_bytes()).hexdigest()!=meta['trajectory_sha256']:raise ValueError('Changed recorded native trajectory')
    return d,meta,paths[0]


def render(root,output,*,slow=5,fps=30):
    d,meta,source=load(root);q=d['positions_m'];rest=d['rest_m'];weights=volume_weights(rest,d['tets']);faces=d['faces'];fixed=d['fixed_node_ids'].astype(int)
    floor=float(d['floor_z_m']) if 'floor_z_m' in d and np.isfinite(d['floor_z_m']).all() else None
    out=Path(output);out.mkdir(parents=True,exist_ok=False);size=(1280,620)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    writer=imageio_ffmpeg.write_frames(str(out/'actual-and-deformation.mp4'),size,fps=fps,codec='libx264',output_params=['-crf','19','-threads','2','-movflags','+faststart'],macro_block_size=2);writer.send(None)
    count=0;peak=0.;preview=None;name=Path(root).name
    try:
        for i,t in enumerate(d['times_s']):
            aligned,rms=align_to_rest(rest,q[i],weights);peak=max(peak,rms)
            for repeat in range(slow):
                phase=(i*slow+repeat)/max(len(q)*slow-1,1);canvas=Image.new('RGB',size,'#f6f8fa');draw=ImageDraw.Draw(canvas)
                draw.text((18,12),f"{name} | E = {meta['case']['young_Pa']/1000:g} kPa | actual Isaac FEM nodes",font=font,fill='#142b3b')
                draw.text((18,40),'World motion; camera follows mass centre',font=font,fill='#142b3b');draw.text((660,40),'Same frame after rigid-motion removal',font=font,fill='#142b3b')
                draw_panel(canvas,q[i],rest,faces,fixed,weights@q[i],phase,(10,70,630,535),floor)
                draw_panel(canvas,aligned,rest,faces,fixed,weights@rest,phase,(650,70,1270,535))
                draw.text((18,552),f't = {float(t):.3f} s | nonrigid coordinate RMS = {rms*1e3:.3f} mm | COM z = {(weights@q[i])[2]:.3f} m',font=font,fill='#142b3b')
                draw.text((18,582),f'True geometric scale | {slow}x slow playback | recorded states held, no motion interpolation | orange: fixed nodes',font=small,fill='#344c5a')
                writer.send(np.asarray(canvas).tobytes());count+=1
                if i==len(q)//2 and repeat==0:preview=canvas.copy()
    finally:writer.close()
    if preview is not None:preview.save(out/'preview.png')
    reader=imageio_ffmpeg.read_frames(str(out/'actual-and-deformation.mp4'));next(reader);decoded=sum(1 for _ in reader);assert decoded==count
    info={'schema':'recorded-Isaac-world-and-rigidremoved-video-v0.1','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frames':count,'decoded_frames':decoded,'fps':fps,'playback_slow_factor':slow,'deformation_scale':1,'camera_orbit_degrees':45,'camera_tracks_mass_center':True,'interpolated_simulation_states':False,'is_prediction':False,'peak_nonrigid_coordinate_RMS_m':peak,'mp4_sha256':hashlib.sha256((out/'actual-and-deformation.mp4').read_bytes()).hexdigest(),'native_case_metadata':meta}
    (out/'video.json').write_text(json.dumps(info,indent=2)+'\n');return info

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',required=True);p.add_argument('--output',required=True);p.add_argument('--slow',type=int,default=5);a=p.parse_args();print(json.dumps({k:v for k,v in render(a.episode,a.output,slow=a.slow).items() if k!='native_case_metadata'}))
