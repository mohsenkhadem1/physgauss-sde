"""True five-times-slow playback of ALL recorded 240 Hz impact states.

A 60 fps video can display each 240 Hz source sample at 5x slow playback.
States are selected by nearest actual recorded timestamp; no q interpolation.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from experiments.video_block_gravity_suite import load,volume_weights,align_to_rest,draw_panel


def frame_schedule(times,slow=5,fps=60):
    t=np.asarray(times,dtype=float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or slow<=0 or fps<=0:raise ValueError('Increasing recorded timestamps required')
    desired=np.linspace(t[0],t[-1],round((t[-1]-t[0])*slow*fps)+1)
    distances=abs(desired[:,None]-t[None]);indices=np.argmin(distances,axis=1)
    if not np.array_equal(np.unique(indices),np.arange(len(t))):raise ValueError('Video sampling must display every recorded impact state')
    return indices,desired


def render(episode,output):
    d,meta,source=load(episode);times=d.get('actual_times_s',d['times_s']);idx,desired=frame_schedule(times)
    rest=d['rest_m'];q=d['positions_m'];faces=d['faces'];fixed=d['fixed_node_ids'].astype(int);w=volume_weights(rest,d['tets'])
    floor=float(d['floor_z_m']) if np.isfinite(d['floor_z_m']).all() else None
    out=Path(output);out.mkdir(parents=True,exist_ok=False);size=(1280,620)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    states=[align_to_rest(rest,v,w) for v in q];rms=np.array([r[1] for r in states]);peak=int(np.argmax(rms));preview=None
    writer=imageio_ffmpeg.write_frames(str(out/'impact-dense.mp4'),size,fps=60,codec='libx264',output_params=['-crf','19','-threads','2','-movflags','+faststart'],macro_block_size=2);writer.send(None)
    try:
        for frame,i in enumerate(idx):
            phase=frame/max(len(idx)-1,1);canvas=Image.new('RGB',size,'#f6f8fa');draw=ImageDraw.Draw(canvas)
            draw.text((18,12),f"{meta['case']['id']} | E = {meta['case']['young_Pa']/1000:g} kPa | all recorded impact states",font=font,fill='#142b3b')
            draw.text((18,40),'World motion; camera follows mass centre',font=font,fill='#142b3b');draw.text((660,40),'Same frame after rigid-motion removal',font=font,fill='#142b3b')
            draw_panel(canvas,q[i],rest,faces,fixed,w@q[i],phase,(10,70,630,535),floor)
            draw_panel(canvas,states[i][0],rest,faces,fixed,w@rest,phase,(650,70,1270,535))
            draw.text((18,552),f't = {times[i]:.4f} s | nonrigid RMS = {1000*rms[i]:.3f} mm | COM z = {(w@q[i])[2]:.3f} m',font=font,fill='#142b3b')
            draw.text((18,582),'True geometric scale | 5x slow | 240 Hz recorded states, 60 fps video | no motion interpolation | contact demo',font=small,fill='#344c5a')
            writer.send(np.asarray(canvas).tobytes())
            if i==peak and preview is None:preview=canvas.copy()
    finally:writer.close()
    if preview is not None:preview.save(out/'peak-impact.png')
    reader=imageio_ffmpeg.read_frames(str(out/'impact-dense.mp4'));next(reader);count=sum(1 for _ in reader);assert count==len(idx)
    info={'schema':'all-recorded-impact-states-video-v0.1','native_source':str(source),'native_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'recorded_frames':len(q),'video_frames':count,'fps':60,'slow_factor':5,'every_recorded_state_shown':True,'deformation_scale':1,'motion_interpolation':False,'peak_nonrigid_coordinate_RMS_m':float(rms[peak]),'peak_time_s':float(times[peak]),'maximum_timestamp_selection_error_s':float(np.max(abs(times[idx]-desired))),'is_prediction':False,'contact_accuracy_qualified':False,'mp4_sha256':hashlib.sha256((out/'impact-dense.mp4').read_bytes()).hexdigest()}
    (out/'video.json').write_text(json.dumps(info,indent=2)+'\n');return info

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(render(a.episode,a.output)))
