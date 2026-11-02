"""Illustrative shared-texture orbit video from sealed development predictions.
This is a geometry visualization, not native learned-appearance quality evidence.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from scripts.compositional_dynamics_v17.matched_forecast_evaluation import committed_predictions,PUBLIC,TRUTH
from scripts.compositional_dynamics_v17.prototype import ROOT,face_grid
from scripts.compositional_dynamics_v17.render_sequence import render_frame
from scripts.physical_refinement_v15.observations import Camera,box_mesh


def texture_subdivision(surface):
    # Exact subdivision of existing planar triangles, not a new deformation model.
    dense=np.empty((6,17,17,3))
    for i in range(17):
        u=i/4;ii=min(int(u),3);f=u-ii
        for j in range(17):
            v=j/4;jj=min(int(v),3);g=v-jj
            a,b,c,d=surface[:,ii,jj],surface[:,ii+1,jj],surface[:,ii+1,jj+1],surface[:,ii,jj+1]
            dense[:,i,j]=(1-f)*a+(f-g)*b+g*c if g<=f else (1-g)*a+f*c+(g-f)*d
    return dense


def main():
    records,evidence,selected=committed_predictions()
    # Require the already-published score as proof this is after evaluation release.
    score=ROOT/'reports/compositional-dynamics-v17/matched-forecast-development-results.json'
    if not score.exists():raise ValueError('Scored development experiment required')
    output=ROOT/'runs/compositional-dynamics-v17/development-orbit-actual-ours-native-v2.mp4'
    if output.exists():raise ValueError('Preserve existing video')
    meta=json.loads(PUBLIC.read_text());q=np.array(meta['material_queries']);fq=face_grid(5);dense=face_grid(17)
    surface=[np.flatnonzero(np.all(q==p,axis=1))[0]for p in fq]
    target_ids=[np.flatnonzero(np.all(dense==p,axis=1))[0]for p in q]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    truth_status=json.loads((TRUTH/'complete.json').read_text())
    if not truth_status['completed']or truth_status['trajectory_sha256']!=sha(TRUTH/'trajectory.npz'):raise ValueError('Reference integrity')
    with np.load(TRUTH/'trajectory.npz')as f:
        ids=np.arange(0,len(f['times_s']),10);times=f['times_s'][ids];reference=f['surface_frames'][ids].reshape(len(ids),-1,3)[:,target_ids]
    clouds=[reference]
    for wanted in ['ours_fine',selected]:
        name,status,path,key=next(r for r in records if r[0]==wanted)
        if not status['completed']:raise ValueError('Video requires complete declared trajectories; no prefix substitution')
        with np.load(path)as f:
            if not np.allclose(f['times_s'],times,atol=1e-12,rtol=0):raise ValueError('Simultaneous frames required')
            clouds.append(f[key].copy())
    labels=['Simulation reference','Our inferred model','Native PhysTwin (selected)'];font=ImageFont.truetype('DejaVuSans.ttf',18);small=ImageFont.truetype('DejaVuSans.ttf',14)
    floor_z=meta['forecast_public_setup']['floor_z_m'];floor=box_mesh([.06,0,floor_z-.001],[.45,.4,.002],[.75,.77,.79]);fps=8
    writer=imageio_ffmpeg.write_frames(str(output),(1536,464),fps=fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','19']);writer.send(None);cameras=[];previews=[]
    try:
        for i,t in enumerate(times):
            theta=np.deg2rad(-75+50*i/(len(times)-1));camera=Camera((.06+.36*np.cos(theta),.36*np.sin(theta),.16),(.06,0,-.025),512,384,750.)
            cameras.append(vars(camera));frame=Image.new('RGB',(1536,464),'white');draw=ImageDraw.Draw(frame)
            for j,(label,points)in enumerate(zip(labels,clouds)):
                rgb=render_frame(texture_subdivision(points[i,surface].reshape(6,5,5,3)),camera,[floor],time_s=float(t))['rgb'];frame.paste(Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)),(j*512,80));draw.text((j*512+10,8),label,font=font,fill='black');draw.text((j*512+10,34),f't = {t:.2f} s | 12.5x slow motion | physical scale',font=small,fill='black');draw.text((j*512+10,56),'Shared texture | camera orbit: 50 degrees',font=small,fill='black')
            writer.send(np.asarray(frame))
            if i in [0,10,18,25]:previews.append(frame.copy())
    finally:writer.close()
    sheet=Image.new('RGB',(1536,464*len(previews)),'white')
    for i,frame in enumerate(previews):sheet.paste(frame,(0,464*i))
    sheet.save(output.with_suffix('.png'))
    report=dict(video=str(output.relative_to(ROOT)),video_sha256=sha(output),source_sha256=sha(__file__),score_sha256=sha(score),reference_sha256=sha(TRUTH/'trajectory.npz'),frame_count=len(times),times_s=times.tolist(),fps=fps,cameras=cameras,selected_native=selected,scope='Illustrative identical procedural appearance on matched98point material surfaces; exact planar-triangle subdivision only improves texture sampling, not geometry; not native GS image scores. Same moving camera in all panels. No displacement magnification or trajectory interpolation. One finite-resolution development reference.')
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'video':str(output),'frames':len(times)}))

if __name__=='__main__':main()
