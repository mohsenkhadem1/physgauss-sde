"""Matched 360-degree novel-camera visualization, with no novel-view RGB truth."""
from scripts.real_transfer_v20.render_audit import load_gaussians,skinning_setup,skin,ROOT,ASSETS
from scripts.real_transfer_v22.campaign import LABEL
from scripts.external_benchmark.data import phystwin
from scripts.external_benchmark.evaluate import expand_prediction
import argparse,json,pickle,hashlib,subprocess
from pathlib import Path
import numpy as np,torch,gsplat,cv2
from scipy.spatial.transform import Rotation



def look_at(eye,target,up=np.array([0.,0.,-1.])):
    forward=target-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,up);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    C=np.eye(4);C[:3,:3]=np.stack([right,down,forward],axis=1);C[:3,3]=eye
    return C


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--label',default=LABEL)
    p.add_argument('--source',default=None)
    a=p.parse_args();case=a.case;torch.set_num_threads(2)
    key=f'{a.source}--{case}' if a.source else case
    out=ROOT/'runs/real-transfer-v22/novel-views'/key;out.mkdir(parents=True,exist_ok=True)
    data=phystwin(a.source or case,240);flip=data['evaluation']['transform'];initial=data['support_geometry']@flip
    if a.source:
        fit=ROOT/'runs/real-transfer-v20/frozen-transfer/v22-corrected-frozen'/key/'neural/prediction.npz'
        with np.load(fit) as f:ours=f['predicted']@flip
        nativepath=ROOT/'runs/real-transfer-v20/native-transfer'/key/'prediction.npz'
        with np.load(nativepath) as f:native=f['predicted']@flip
    else:
        fit=ROOT/'runs/real-transfer-v19/fits'/a.label/case/'neural/prediction.npz'
        with np.load(fit) as f:ours=expand_prediction(data,f['predicted'])@flip
        nativepath=ROOT/'runs/third_party/PhysTwin/reference/experiments'/case/'inference.pkl'
        with nativepath.open('rb') as f:native=np.asarray(pickle.load(f))[:,:len(initial)]
    frame=min(len(ours),len(native))-1
    xyz,scales,quat,opacity,sh=load_gaussians(a.source or case)
    if a.source:
        reg=next(r for r in json.loads((ROOT/'reports/real-transfer-v20/initial-registration.json').read_text())['pairs'] if r['source']==a.source and r['target']==case)
        R=flip@np.asarray(reg['rotation'])@flip;b=np.asarray(reg['translation_m'])@flip
        initial=initial@R+b;xyz=xyz@R+b
    binding=skinning_setup(initial,xyz)
    # Initial-only binding: no future reference visibility or motion enters.
    warped=[skin(initial,pred[frame],xyz,binding) for pred in (ours,native)]
    tensor=lambda v:torch.as_tensor(np.asarray(v),dtype=torch.float32,device='cuda')
    constant=dict(scales=tensor(scales),quats=tensor(quat),opacities=tensor(opacity),colors=tensor(sh),sh_degree=3)
    means=[tensor(w) for w in warped]
    # A common camera envelope covers both forecasts; no per-method recentering.
    points=np.concatenate([ours[frame],native[frame]])
    lo,hi=points.min(0),points.max(0);target=(lo+hi)/2
    radius=max(float(np.linalg.norm(hi-lo))*.95,.15)
    original=ROOT/'runs/third_party/PhysTwin/data/different_types'/case
    with (original/'calibrate.pkl').open('rb') as f:real_cameras=np.asarray(pickle.load(f))
    base=real_cameras[0,:3,3]-target;base[2]=0;base/=np.linalg.norm(base)
    W=512;H=512;K=np.array([[560,0,W/2],[0,560,H/2],[0,0,1.]])
    images=[];cameras=[];errors=[];tiles=[]
    movie=out/'novel-camera-orbit.mp4'
    ffmpeg=str(ROOT/'runs/media-tooling/ffmpeg-7.0.2-amd64-static/ffmpeg')
    proc=subprocess.Popen([ffmpeg,'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x604','-r','20','-i','-',
        '-an','-vcodec','libx264','-crf','20','-pix_fmt','yuv420p',str(movie)],stdin=subprocess.PIPE,stderr=open(out/'encode.log','w'))
    for j in range(120):
        angle=2*np.pi*j/120
        direction=Rotation.from_rotvec(np.array([0.,0.,angle])).apply(base)
        eye=target+radius*(direction+np.array([0.,0.,-.25]))
        C=look_at(eye,target);cameras.append(C.tolist());panels=[]
        for mean in means:
            with torch.no_grad():rgb,alpha,_=gsplat.rasterization(means=mean,**constant,Ks=tensor(K)[None],viewmats=tensor(np.linalg.inv(C))[None],width=W,height=H,render_mode='RGB')
            image=np.clip(rgb[0].cpu().numpy(),0,1);panels.append((image*255).astype(np.uint8))
        canvas=np.full((604,1024,3),245,np.uint8);canvas[62:574]=np.concatenate(panels,axis=1)
        cv2.putText(canvas,'Ours: corrected forecast',(22,27),cv2.FONT_HERSHEY_SIMPLEX,.64,(20,65,80),2)
        cv2.putText(canvas,'PhysTwin: '+('adapted frozen twin' if a.source else 'released forecast'),(534,27),cv2.FONT_HERSHEY_SIMPLEX,.64,(110,65,20),2)
        cv2.putText(canvas,f'{case} | frozen frame {frame} | camera orbit {j*3:03d} deg',(22,51),cv2.FONT_HERSHEY_SIMPLEX,.49,(45,45,45),1)
        cv2.putText(canvas,'Unrecorded cameras: qualitative comparison, no ground-truth images',(22,594),cv2.FONT_HERSHEY_SIMPLEX,.52,(50,50,50),1)
        proc.stdin.write(canvas.tobytes())
        if j in (0,30,60,90):
            cv2.imwrite(str(out/f'angle-{j*3:03d}.png'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR));tiles.append(canvas)
    proc.stdin.close();code=proc.wait()
    if code:raise RuntimeError('Video encoding failed')
    cv2.imwrite(str(out/'contact-sheet.png'),cv2.cvtColor(np.concatenate(tiles,axis=0),cv2.COLOR_RGB2BGR))
    manifest=dict(case=case,source=a.source,frame=frame,label=a.label,frames=120,fps=20,camera_intrinsics=K.tolist(),camera_to_world=cameras,
        appearance='Shared initial released Gaussian appearance, no refitting; degree-three SH evaluated at each novel camera.',
        binding='Same full initial support geometry, local rigid transport, no future visibility selection.',
        scope='Qualitative frozen-time 360-degree orbit, not temporal animation or novel-view accuracy evaluation. Unobserved surfaces can have appearance holes.',
        movie=str(movie.relative_to(ROOT)),input_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (fit,nativepath,Path(__file__))})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(case,'novel views complete',flush=True)

if __name__=='__main__':main()
