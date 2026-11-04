"""Real held-camera footage with reference/predicted geometry overlays."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,subprocess
import numpy as np,cv2
from .protocol import ROOT,BASE,REPORT,DATA,write
from .data import Sequence

def project(points,C,K):
    x=(points-C[:3,3])@C[:3,:3];p=x@K.T;return p[:,:2]/p[:,2,None]

def main():
    plan=json.load(open(REPORT/'visualization-plan.json'));inventory=[]
    for case in plan['cases']:
        s=Sequence(case);directory=BASE/'predictions'/case
        with np.load(directory/'predictions.npz')as f:pred={m:f[m].copy()for m in ('visual','image_blind')}
        labels=s.labels['tracks'];ids=np.linspace(0,len(labels[0])-1,min(70,len(labels[0])),dtype=int)
        width,height=s.meta['WH'];scale=min(480/width,300/height);w=round(width*scale);h=round(height*scale)
        # Fit all original image content inside fixed panels; no score-based crop.
        W=1440;H=360;movie=directory/'held-camera-reconstruction.mp4'
        ffmpeg=ROOT/'runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg'
        with (directory/'video.log').open('w')as log:
            proc=subprocess.Popen([str(ffmpeg),'-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','15','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(movie)],stdin=subprocess.PIPE,stderr=log)
            for t in range(len(labels)):
                raw=cv2.imread(str(DATA/case/f'color/2/{t}.png'));assert raw is not None
                raw=cv2.cvtColor(cv2.resize(raw,(w,h)),cv2.COLOR_BGR2RGB);canvas=np.full((H,W,3),245,np.uint8)
                methods=['reference','visual','image_blind'];titles=['Recorded view + reference tracks','Learned visual reconstruction','Image-blind control']
                colors=[(255,255,255),(30,255,100),(255,150,40)]
                for col,(method,title,color)in enumerate(zip(methods,titles,colors)):
                    x0=col*480+(480-w)//2;y0=52;canvas[y0:y0+h,x0:x0+w]=raw
                    valid=ids[s.labels['valid'][t,ids]];pts=labels[t,valid] if method=='reference' else pred[method][t,valid]
                    uv=project(pts,s.C[2],s.K[2])*scale
                    for p in uv:
                        if np.isfinite(p).all() and 0<=p[0]<w and 0<=p[1]<h:
                            loc=(x0+int(p[0]),y0+int(p[1]));cv2.circle(canvas,loc,3,(0,0,0),-1,cv2.LINE_AA);cv2.circle(canvas,loc,2,color,-1,cv2.LINE_AA)
                    cv2.putText(canvas,title,(col*480+10,23),cv2.FONT_HERSHEY_SIMPLEX,.58,(25,40,55),1,cv2.LINE_AA)
                    cv2.putText(canvas,f'{case} | frame {t}',(col*480+10,44),cv2.FONT_HERSHEY_SIMPLEX,.42,(35,35,35),1,cv2.LINE_AA)
                cv2.putText(canvas,'Dots are projected 3D predictions; backgrounds are recorded footage, not generated images. Target-time RGB from cameras 0/1 is input.',(10,348),cv2.FONT_HERSHEY_SIMPLEX,.48,(35,35,35),1,cv2.LINE_AA)
                proc.stdin.write(canvas.tobytes())
                if t==len(labels)-1:cv2.imwrite(str(directory/'comparison.png'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
            proc.stdin.close();assert proc.wait()==0
        inventory.append(dict(case=case,movie=str(movie.relative_to(ROOT)),frames=len(labels),scope='Observed-video reconstruction overlay, not generated RGB or forecasting. Reference tracks are author estimates.'))
    write(REPORT/'videos.json',dict(videos=inventory));print(inventory)

if __name__=='__main__':main()
