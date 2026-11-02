"""Fixed-camera actual/ours/native video from committed images. Source-only until release.
No simulation, fitting, image warping, motion interpolation or metric-based case choice.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,tempfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/compositional-dynamics-v17/matched-appearance-native-resolution'
BASE=ROOT/'runs/compositional-dynamics-v17/matched-appearance-render'
PLAN=REPORT/'dynamic-plan.json'
CAMERA='held-0'
FPS=10

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def five_forecasts():
    cases=[('coarse','matched-material-forecast','prediction-commit.json','surface-prediction.npz','surface_prediction_sha256'),('fine','matched-material-fine-forecast','prediction-commit.json','surface-prediction.npz','surface_prediction_sha256'),('native125','native-phystwin-graph-recovery','recovery-complete.json','forecast.npz','forecast_sha256'),('native729','native-phystwin-capacity-forecast','complete.json','prediction.npz','prediction_sha256'),('native_extended','native-phystwin-budget-forecast','complete.json','prediction.npz','prediction_sha256')]
    evidence={}
    for _,folder,name,array,key in cases:
        b=ROOT/'runs/compositional-dynamics-v17'/folder;s=read(b/name)
        if s.get('completed')is not True or sha(b/array)!=s[key]:raise ValueError('All five completed sealed physical forecasts required')
        evidence[str((b/name).relative_to(ROOT))]=sha(b/name);evidence[str((b/array).relative_to(ROOT))]=sha(b/array)
    return evidence

def projection(points,camera):
    eye=np.asarray(camera['eye']);forward=np.asarray(camera['target'])-eye;forward/=np.linalg.norm(forward)
    # Use the same declared camera implementation, without reading any scene target.
    from scripts.physical_refinement_v15.observations import Camera
    cam=Camera(tuple(camera['eye']),tuple(camera['target']),camera['width'],camera['height'],camera['focal']);eye,R=cam.pose();x=(np.asarray(points)-eye)@R.T
    return x[:,:2]/x[:,2,None]*camera['focal']+[camera['width']/2,camera['height']/2]

def build():
    evidence=five_forecasts() # Before any saved reference image access.
    p=read(PLAN);commit=read(BASE/'prediction-commit.json');results_path=REPORT/'dynamic-results.json';results=read(results_path)
    if commit['plan_sha256']!=sha(PLAN)or results['plan_sha256']!=sha(PLAN)or results['prediction_commit_sha256']!=sha(BASE/'prediction-commit.json'):raise ValueError('Dynamic render/scoring commitment mismatch')
    if len(commit['records'])!=260 or len({(r['method'],r['frame'],r['camera'])for r in commit['records']})!=260:raise ValueError('Complete unique image census required')
    for group in ['source_sha256','input_sha256']:
        for name,h in p[group].items():
            if sha(ROOT/name)!=h:raise ValueError('Dynamic source/input changed')
    for row in commit['records']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Predicted image changed')
    for path,h in results['reference_images_sha256'].items():
        if sha(ROOT/path)!=h:raise ValueError('Reference image changed')
    cameras=[c for c in p['cameras']if c['name']==CAMERA]
    if len(cameras)!=1:raise ValueError('Prespecified held camera missing')
    camera=cameras[0];times=np.asarray(p['times_s']);
    if len(times)!=26 or not np.allclose(times,np.arange(26)*.01,atol=1e-12,rtol=0):raise ValueError('Declared development26frame clock')
    evidence.update({str(PLAN.relative_to(ROOT)):sha(PLAN),str((BASE/'prediction-commit.json').relative_to(ROOT)):sha(BASE/'prediction-commit.json'),str(results_path.relative_to(ROOT)):sha(results_path),str(Path(__file__).relative_to(ROOT)):sha(__file__)})
    output=REPORT/'videos';output.mkdir(exist_ok=True);video=output/'actual-ours-native-held0.mp4'
    if video.exists():raise ValueError('Preserve video artifact')
    forecast_plan=read(ROOT/p['ours_forecast_plan']);floor=float(forecast_plan['setup']['floor_z_m'])
    corners=np.array([[-.03,-.07,floor],[.15,-.07,floor],[.15,.07,floor],[-.03,.07,floor],[-.03,-.07,floor]])
    line=projection(corners,camera)
    rows={(x['method'],x['frame'],x['camera']):x for x in commit['records']}
    width,height=camera['width'],camera['height'];panel_w=width;header=54;footer=60
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    except OSError:font=ImageFont.load_default()
    sheets=[]
    with tempfile.TemporaryDirectory(prefix='v17-video-')as tmp:
        for i,t in enumerate(times):
            paths=[BASE/'references'/f'{i:03d}-{CAMERA}.npz',ROOT/rows['shared_q2',i,CAMERA]['path'],ROOT/rows['native',i,CAMERA]['path']]
            canvas=Image.new('RGB',(3*panel_w,header+height+footer),'white');draw=ImageDraw.Draw(canvas)
            titles=['Actual simulation','Shared-law prediction','Native PhysTwin prediction']
            for j,(title,path)in enumerate(zip(titles,paths)):
                with np.load(path)as f:rgb=f['rgb']
                image=Image.fromarray(np.uint8(np.round(np.clip(rgb,0,1)*255)))
                if image.size!=(width,height):raise ValueError('Same physical camera/viewport required')
                canvas.paste(image,(j*panel_w,header));draw.text((j*panel_w+12,8),title,font=font,fill='black')
                # Public floor outline is a display guide, not part of image metrics.
                draw.line([(float(x)+j*panel_w,float(y)+header)for x,y in line],fill=(125,125,125),width=2)
                evidence[str(path.relative_to(ROOT))]=sha(path)
            draw.text((12,header+height+5),f'Physical time {t:.2f} s | one-tenth-speed playback | fixed held camera | no motion magnification',font=font,fill='black')
            draw.text((12,header+height+30),'Gray outline: known floor guide. Same learned initial appearance for both predictions; transport methods differ.',font=font,fill='black')
            canvas.save(Path(tmp)/f'{i:03d}.png')
            if i in [0,13,25]:sheets.append(canvas.resize((1272,297)))
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg,'-v','error','-y','-framerate',str(FPS),'-i',str(Path(tmp)/'%03d.png'),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(video)],check=True)
        reader=imageio_ffmpeg.read_frames(str(video));decoded_metadata=next(reader);decoded_frames=sum(1 for _ in reader)
        if decoded_frames!=26 or abs(decoded_metadata['fps']-FPS)>1e-9 or tuple(decoded_metadata['size'])!=(3*panel_w,header+height+footer):raise ValueError('Encoded frame/fps/dimension mismatch')
    sheet=Image.new('RGB',(1272,297*3),'white')
    for i,img in enumerate(sheets):sheet.paste(img,(0,297*i))
    preview=output/'actual-ours-native-held0-preview.png';sheet.save(preview)
    metadata=dict(scope='True dynamic forecast frames; fixed camera, one-tenth-speed playback.26frames span0..0.25s physical, encoded10fps. Not an orbit or interpolated dynamics.',camera=CAMERA,selection='Fixed prospective held-0; not selected by forecast/image error.',frames=26,fps=FPS,encoded_duration_s=26/FPS,physical_duration_s=.25,motion_magnification=1,video_sha256=sha(video),preview_sha256=sha(preview),evidence_sha256=evidence,decode_passed=True,decoded_frames=decoded_frames,decoded_metadata=decoded_metadata,visual_review='pending',floor_guide='Public plane outline annotation only; absent from all scored image arrays.')
    (output/'actual-ours-native-held0-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true',help='Only after root authorizes all-five-forecast release');args=a.parse_args()
    if not args.execute:raise SystemExit('Source prepared only; execute after explicit release and completed render/scoring artifacts.')
    build()
