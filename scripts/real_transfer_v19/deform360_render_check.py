"""Development-only held-camera reconstruction QA, not motion prediction."""
from . import deform360_reconstruct  # configure the local CUDA compiler first
import argparse,json
from pathlib import Path
import cv2,h5py,numpy as np,torch,gsplat
from plyfile import PlyData
from deform360.processing.depth_stage import load_splat_arrays
from deform360.processing.episode import load_episode_calibration

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frame',type=int,default=0)
    parser.add_argument('--object',default='046-sponge');parser.add_argument('--episode',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(2)
    episode=ROOT/'runs/real-transfer-v19/Deform360/processed'/args.object/f'episode_{args.episode:04d}'
    plan=json.loads((episode/'splatfacto/v19-plan.json').read_text())
    splat=load_splat_arrays(episode/f'splatfacto/splat_{args.frame}.ply')
    vertex=PlyData.read(str(episode/f'splatfacto/splat_{args.frame}.ply'))['vertex']
    rest_names=sorted((p.name for p in vertex.properties if p.name.startswith('f_rest_')),
                      key=lambda name:int(name.split('_')[-1]))
    rest=np.stack([vertex[name] for name in rest_names],axis=1).reshape(len(vertex),3,-1).transpose(0,2,1)
    sh=np.concatenate([splat['colors'][:,None,:],rest],axis=1)
    degree=int(round(np.sqrt(sh.shape[1])-1))
    K,C=load_episode_calibration(episode);records=[];tiles=[]
    def tensor(x):return torch.as_tensor(np.asarray(x),dtype=torch.float32,device='cuda')
    for camera in plan['held_views']:
        cap=cv2.VideoCapture(str(episode/camera/'undistorted.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES,args.frame);ok,frame=cap.read();cap.release()
        if not ok:raise ValueError('Missing image')
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(float)/255
        with h5py.File(episode/camera/'mask_refined.h5') as f:mask=f['data'][args.frame]>0
        height,width=mask.shape
        with torch.no_grad():
            image,alpha,_=gsplat.rasterization(means=tensor(splat['means']),quats=tensor(splat['quats']),
                scales=tensor(splat['scales']),opacities=tensor(splat['opacities']),colors=tensor(sh),sh_degree=degree,
                Ks=tensor(K[camera])[None],viewmats=tensor(np.linalg.inv(C[camera]))[None],
                width=width,height=height,render_mode='RGB')
        rendered=image[0].cpu().numpy();opacity=alpha[0,:,:,0].cpu().numpy()
        yy,xx=np.where(mask)
        if not len(xx):continue
        y0,y1=max(0,yy.min()-15),min(height,yy.max()+16)
        x0,x1=max(0,xx.min()-15),min(width,xx.max()+16)
        actual=rgb*mask[:,:,None];error=np.abs(actual-rendered)
        mse=float(np.mean(error[y0:y1,x0:x1]**2));pred=opacity>.5
        records.append(dict(camera=camera,crop_psnr_db=float(-10*np.log10(max(mse,1e-12))),
                            silhouette_iou=float((pred&mask).sum()/max(1,(pred|mask).sum()))))
        row=[]
        for panel in (actual,rendered,np.clip(error*3,0,1)):
            crop=panel[y0:y1,x0:x1];row.append(cv2.resize(crop,(240,180)))
        tiles.append(np.concatenate(row,axis=1))
    output=ROOT/'reports/real-transfer-v19';stem=f'deform360-{args.object}-frame{args.frame}-held-view-qa'
    report=dict(object=args.object,episode=args.episode,frame=args.frame,views=records,
                scope='Development reconstruction only, not physical forecast. Initial held-camera masks used for view qualification.',
                panel_order='Masked actual | reconstructed | absolute RGB difference x3')
    (output/(stem+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    cv2.imwrite(str(output/(stem+'.png')),cv2.cvtColor((np.clip(np.concatenate(tiles),0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
