"""Assemble the prescribed eight scenarios; retain full-rate impact clips."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from PIL import Image
from experiments.video_block_gravity_suite import load,align_to_rest,volume_weights


def assemble(base,output):
    base=Path(base);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    names=['gravity_left','gravity_right','gravity_bothends','gravity_bottom','gravity_two_bottom_patches','freefall_300k','drop_300k-dense','drop_600k-dense']
    clips=[];expected=0
    for name in names:
        p=base/name;meta=json.loads((p/'video.json').read_text());movie=p/('impact-dense.mp4' if name.endswith('-dense') else 'actual-and-deformation.mp4')
        assert hashlib.sha256(movie.read_bytes()).hexdigest()==meta['mp4_sha256']
        frames=meta.get('video_frames',meta.get('decoded_frames'));expected+=frames*60/meta['fps']
        clips.append({'case':name,'path':str(movie.resolve()),'sha256':meta['mp4_sha256'],'native_source_sha256':meta.get('native_source_sha256',meta.get('source_sha256')),'frames':frames,'fps':meta['fps']})
    # These are task-authored paths, passed through an ffconcat file, never a shell.
    listing=out/'clips.ffconcat';listing.write_text('ffconcat version 1.0\n'+''.join("file '"+r['path'].replace("'","'\\''")+"'\n" for r in clips))
    movie=out/'supports-gravity-and-drop.mp4'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(listing),'-vf','fps=60','-c:v','libx264','-threads','2','-crf','19','-movflags','+faststart',str(movie)]
    subprocess.run(command,check=True,capture_output=True)
    decoder=imageio_ffmpeg.read_frames(str(movie));next(decoder);count=sum(1 for _ in decoder)
    if abs(count-round(expected))>1:raise ValueError('Unexpected overview frame count')
    first=json.loads((base/'gravity_left/video.json').read_text());source=Path(first['source']).parent;d,_,_=load(source);w=volume_weights(d['rest_m'],d['tets'])
    rms=[align_to_rest(d['rest_m'],q,w)[1] for q in d['positions_m']];peak=int(np.argmax(rms));frame_index=peak*first['playback_slow_factor']
    decoder=imageio_ffmpeg.read_frames(str(base/'gravity_left/actual-and-deformation.mp4'));format=next(decoder)
    for i,frame in enumerate(decoder):
        if i==frame_index:
            Image.frombytes('RGB',format['size'],frame).save(out/'peak-gravity.png');break
    info={'schema':'prescribed-block-support-gravity-drop-overview-v0.1','clips':clips,'fps':60,'decoded_frames':count,'expected_frames_before_timestamp_rounding':round(expected),'motion_interpolation':False,'resampling':'Presentation-frame duplication for30fpsclips; nativeq never interpolated','camera_orbit_and_mass_tracking':True,'deformation_scale':1,'contact_accuracy_qualified':False,'native_time_refinement_passed':False,'is_prediction':False,'mp4_sha256':hashlib.sha256(movie.read_bytes()).hexdigest(),'peak_gravity_preview_time_s':float(d['times_s'][peak]),'peak_gravity_nonrigid_RMS_m':float(rms[peak])}
    (out/'overview.json').write_text(json.dumps(info,indent=2)+'\n');return {'video':str(movie),'frames':count,'clips':len(clips)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(assemble(a.base,a.output)))
