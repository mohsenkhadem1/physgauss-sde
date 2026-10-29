"""Camera-orbit comparisons of actual and predicted settled geometry."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
import cv2
import imageio_ffmpeg
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT
from scripts.phystwin.sequential_fixture import points
from scripts.position_ood.pilot import ACTIONS


def quads():
    shape=(9,3,3);faces=[]
    for axis in range(3):
        others=[j for j in range(3) if j!=axis]
        for side in [0,shape[axis]-1]:
            for i in range(shape[others[0]]-1):
                for j in range(shape[others[1]]-1):
                    row=[]
                    for a,b in [(0,0),(1,0),(1,1),(0,1)]:
                        ix=[0,0,0];ix[axis]=side;ix[others[0]]=i+a;ix[others[1]]=j+b;row.append(np.ravel_multi_index(ix,shape))
                    faces.append(row)
    return np.array(faces)


def draw(image,positions,rect,angle,color,label,error=None):
    x0,y0,w,h=rect;center=np.array([.12,0.,-.005]);az=angle;el=.35
    view=np.array([np.cos(el)*np.sin(az),-np.cos(el)*np.cos(az),np.sin(el)]);right=np.cross(view,[0,0,1.]);right/=np.linalg.norm(right);up=np.cross(right,view)
    Q=np.stack([right,up,view],axis=1);xyz=(positions-center)@Q;scale=min(w/.36,h/.24);xy=np.c_[x0+w/2+scale*xyz[:,0],y0+h*.53-scale*xyz[:,1]].astype(np.int32);faces=quads();depth=xyz[faces,2].mean(1)
    for face in faces[np.argsort(depth)]:
        normal=np.cross(positions[face[1]]-positions[face[0]],positions[face[3]]-positions[face[0]]);normal/=max(np.linalg.norm(normal),1e-12);shade=.55+.45*abs(float(normal@np.array([.3,-.4,.866])));fill=tuple(int(c*shade) for c in color);cv2.fillConvexPoly(image,xy[face],fill,cv2.LINE_AA);cv2.polylines(image,[xy[face]],True,(80,85,90),1,cv2.LINE_AA)
    for p in xy:cv2.circle(image,tuple(p),2,(30,35,40),-1,cv2.LINE_AA)
    cv2.putText(image,label,(x0+18,y0+30),cv2.FONT_HERSHEY_SIMPLEX,.63,(30,35,40),1,cv2.LINE_AA)
    if error is not None:cv2.putText(image,f'Coordinate RMSE: {error:.3f} mm',(x0+18,y0+h-15),0,.52,(40,45,50),1,cv2.LINE_AA)


def main(a):
    ref=np.load(BASE/f'fields/{a.split}-{a.object}.npz')['field'];ours=np.load(BASE/f'inverse/{a.split}-{a.object}/joint-n2-p3.npz')['predicted'];pt=np.load(BASE/f'phystwin/{a.split}-{a.object}/y{a.initial:g}-dt1/predictions.npz')['predicted'];mpm=np.load(BASE/f'mpm/{a.split}-{a.object}/predictions.npz')['predicted'];rows=[('Reference',ref,(105,185,120)),('Position-conditioned inverse',ours,(220,160,85)),('PhysTwin core adaptation',pt,(175,140,220)),('PhysGaussian MPM adaptation',mpm,(190,185,95))]
    dest=OUT/f'{a.split}-{a.object}-actual-vs-predicted.mp4';OUT.mkdir(exist_ok=True);W,H=1200,760;fps=24
    process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-vcodec','libx264','-crf','20','-pix_fmt','yuv420p',str(dest)],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    try:
        for action in range(len(ref)):
            for frame in range(fps*2):
                image=np.full((H,W,3),248,np.uint8);angle=-.7+1.4*frame/(fps*2-1)
                for k,(name,fields,color) in enumerate(rows):
                    error=None if k==0 else 1000*np.sqrt(np.mean((fields[action]-ref[action])**2));draw(image,points()+fields[action],((k%2)*600,65+(k//2)*310,600,300),angle,color,name,error)
                phase='OBSERVED PREFIX' if action<3 else 'HELD-OUT POSITIONAL ACTION';name=ACTIONS[action][0].replace('_',' ');cv2.putText(image,f'{a.split.upper()} OBJECT {a.object} | {phase}: {name}',(20,32),0,.66,(25,30,35),1,cv2.LINE_AA)
                cv2.putText(image,'Actual deformation scale. Camera orbit around settled shapes; this is not a dynamic rollout.',(20,725),0,.57,(35,40,45),1,cv2.LINE_AA);process.stdin.write(image.tobytes())
                if action==6 and frame==fps:cv2.imwrite(str(dest.with_suffix('.png')),image)
    finally:process.stdin.close();code=process.wait()
    assert code==0;print(dest)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--object',type=int,default=1);p.add_argument('--initial',type=float,default=3.);main(p.parse_args())
