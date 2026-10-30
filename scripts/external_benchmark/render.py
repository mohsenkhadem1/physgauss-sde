"""Shared-scale point-cloud actual/predicted videos, with a gentle camera orbit."""
import argparse,json,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,cv2,imageio_ffmpeg
from scripts.external_benchmark.common import *
from scripts.external_benchmark.data import phystwin,softvt
from scripts.external_benchmark.evaluate import expand_prediction,baselines

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['phystwin','softvt'],required=True);p.add_argument('--case',default='single_lift_sloth');p.add_argument('--nodes',type=int,default=240)
    p.add_argument('--subset',default='object-soft');p.add_argument('--task',type=int,default=0);p.add_argument('--episode',type=int,default=0);p.add_argument('--tag',default='refined');a=p.parse_args()
    data=phystwin(a.case,a.nodes) if a.dataset=='phystwin' else softvt(a.subset,a.task,a.episode)
    folder=BASE/'fits'/a.tag/data['name'];fit=json.loads((folder/'fit.json').read_text())
    raw=np.load(folder/'prediction.npz')['predicted'];pred=expand_prediction(data,raw)[:,:len(data['evaluation']['points'][0])]
    truth=data['evaluation']['points'][:len(pred)];masks=data['evaluation']['mask'][:len(pred)]
    control=baselines(data,fit['fit_end'])['control_rigid'][:len(pred),:len(truth[0])]
    # Camera framing may use evaluator geometry; no predictor receives it.
    finite=np.concatenate([truth.reshape(-1,3),pred.reshape(-1,3)]);finite=finite[np.isfinite(finite).all(1)]
    center=(finite.min(0)+finite.max(0))/2;radius=np.linalg.norm(finite-center,axis=1).max();scale=240/max(radius,1e-3)
    initial=truth[0];colors=np.zeros((len(initial),3));colors[:,0]=(initial[:,0]-initial[:,0].min())/(np.ptp(initial[:,0])+1e-9)
    colors[:,1]=(initial[:,1]-initial[:,1].min())/(np.ptp(initial[:,1])+1e-9);colors[:,2]=.5
    color=(60+150*colors).astype(np.uint8)
    video=REPORT/f'{a.tag}-{data["name"]}.mp4';temp=video.with_suffix('.raw.mp4');video.parent.mkdir(exist_ok=True,parents=True)
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),20,(1536,640));assert writer.isOpened()
    sample_ids=set(np.linspace(0,len(pred)-1,4,dtype=int).tolist())
    for t in range(len(pred)):
        img=np.full((640,1536,3),248,np.uint8)
        angle=.35+.25*np.sin(2*np.pi*t/max(len(pred)-1,1));up=.3
        right=np.array([np.cos(angle),np.sin(angle),0]);vertical=np.array([-np.sin(angle)*np.sin(up),np.cos(angle)*np.sin(up),np.cos(up)])
        basis=np.c_[right,vertical];depth=np.cross(right,vertical)
        for panel,(points,title) in enumerate([(truth[t],'Recorded geometry'),(pred[t],'Material + world FEM'),(control[t],'Control-driven rigid baseline')]):
            offset=panel*512;cv2.putText(img,title,(offset+18,35),0,.65,(30,30,30),2)
            use=masks[t]&np.isfinite(points).all(1)
            if use.any():
                ids=np.flatnonzero(use);order=np.argsort((points[ids]-center)@depth)
                uv=(points-center)@basis*scale;pixels=np.c_[uv[:,0]+256,330-uv[:,1]].astype(int)
                for i in ids[order]:
                    x,y=pixels[i]
                    if 5<x<507 and 55<y<585:cv2.circle(img,(x+offset,y),2,tuple(int(c) for c in color[i]),-1,cv2.LINE_AA)
            bar=.02 if radius<.15 else .05;px=int(scale*bar)
            cv2.line(img,(offset+25,575),(offset+25+px,575),(30,30,30),2);cv2.putText(img,f'{bar*1000:g} mm',(offset+25,563),0,.42,(30,30,30),1)
        label='FITTING PREFIX' if t<fit['fit_end'] else 'HELD-OUT PREDICTION'
        cv2.putText(img,f'{data["name"]} | frame {t} | {label}',(20,608),0,.6,(20,20,20),1)
        cv2.putText(img,'Shared scale, no deformation exaggeration. Recorded geometry is data, not raw RGB.',(20,633),0,.5,(50,50,50),1)
        writer.write(img)
        if t in sample_ids:cv2.imwrite(str(video.with_name(video.stem+f'-frame{t}.png')),img)
    writer.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(temp),'-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True);temp.unlink()
    write_json(video.with_suffix('.json'),dict(source_prediction=str(folder/'prediction.npz'),sha256=sha256(folder/'prediction.npz'),frames=len(pred),fps=20,fit_end=fit['fit_end'],scope='Point-cloud geometry playback; camera orbits; FEM predicts quasistatic response to time-indexed tool positions.'))
    print(video)
