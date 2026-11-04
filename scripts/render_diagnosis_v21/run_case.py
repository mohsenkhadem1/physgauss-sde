"""Recorded-track oracle proxy with matched motion anchors and appearance.

Recorded points are reconstructed tracks, not dense metrology ground truth.
Reference visibility is used only in this diagnostic, identically for every
method; none of these interventions is a new deployable forecast result.
"""
from scripts.real_transfer_v20.render_audit import load_gaussians,ASSETS
import argparse,hashlib,json,pickle
from pathlib import Path
import cv2,numpy as np,torch,gsplat
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from scripts.real_transfer_v20.metrics import foreground_render_metrics
from scripts.external_benchmark.data import phystwin
from scripts.external_benchmark.evaluate import expand_prediction
from .transport import setup,valid_anchors,warp

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/render-diagnosis-v21'
METHODS=['initial_static','neural_rigid','native_rigid','oracle_rigid','oracle_affine_centers','oracle_affine_cov','neural_affine_cov']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True);args=parser.parse_args();case=args.case
    torch.set_num_threads(2);out=BASE/case;out.mkdir(parents=True,exist_ok=False)
    original=ROOT/'runs/third_party/PhysTwin/data/different_types'/case;local=ASSETS/'data/different_types'/case
    with (original/'final_data.pkl').open('rb') as f:raw=pickle.load(f)
    observed=np.asarray(raw['object_points']);N=observed.shape[1]
    data=phystwin(case,240)
    fit=ROOT/'runs/real-transfer-v19/fits/v20-selected-mesh240'/case/'neural/prediction.npz'
    with np.load(fit) as f:neural=expand_prediction(data,f['predicted'])[:,:N]@data['evaluation']['transform']
    nativepath=ROOT/'runs/third_party/PhysTwin/reference/experiments'/case/'inference.pkl'
    with nativepath.open('rb') as f:native=np.asarray(pickle.load(f))[:,:N]
    metadata=json.loads((original/'metadata.json').read_text());K=np.array(metadata['intrinsics'])
    with (original/'calibrate.pkl').open('rb') as f:C=np.asarray(pickle.load(f))
    frames=json.loads((ASSETS/f'{case}-frames.json').read_text())['frames']
    xyz,scales,quats,opacity,sh=load_gaussians(case)
    if not np.allclose(scales[:,0],scales[:,1]) or not np.allclose(scales[:,0],scales[:,2]):raise ValueError('Expected isotropic initial appearance')
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=False).cuda().eval()
    tensor=lambda v:torch.as_tensor(np.asarray(v),dtype=torch.float32,device='cuda')
    appearance=dict(colors=tensor(sh),sh_degree=3,opacities=tensor(opacity));rows=[];supports=[];deformations=[]
    for frame in frames:
        valid=valid_anchors(raw,frame);initial=observed[0,valid];binding=setup(initial,xyz)
        supports.append(dict(frame=frame,anchors=int(valid.sum()),total_tracks=N,
            gaussian_anchor_distance_median_mm=float(np.median(binding[3])*1000),
            gaussian_fraction_within_10mm=float((binding[3]<.01).mean())))
        for method in METHODS:
            current=observed[frame,valid] if method.startswith('oracle') else neural[frame,valid] if method.startswith('neural') else native[frame,valid]
            if method=='initial_static':current=initial
            means,new_scales,new_quats,diagnostic=warp(initial,current,xyz,scales,quats,binding,
                affine='affine' in method,covariance=method.endswith('_cov'))
            deformations.append(dict(frame=frame,method=method,**diagnostic))
            for camera in range(3):
                labels=json.loads((local/f'mask/mask_info_{camera}.json').read_text())
                object_id=next(k for k,v in labels.items() if v!='hand')
                actual=cv2.cvtColor(cv2.imread(str(local/f'color/{camera}/{frame}.png')),cv2.COLOR_BGR2RGB).astype(float)/255
                mask=cv2.imread(str(local/f'mask/{camera}/{object_id}/{frame}.png'),0)>0
                hand=np.zeros(mask.shape,bool)
                for k,v in labels.items():
                    if v=='hand':hand|=cv2.imread(str(local/f'mask/{camera}/{k}/{frame}.png'),0)>0
                actual*=mask[:,:,None];h,w=mask.shape
                with torch.no_grad():
                    image,alpha,_=gsplat.rasterization(means=tensor(means),scales=tensor(new_scales),quats=tensor(new_quats),
                        **appearance,Ks=tensor(K[camera])[None],viewmats=tensor(np.linalg.inv(C[camera]))[None],width=w,height=h,render_mode='RGB')
                image=np.clip(image[0].cpu().numpy(),0,1);alpha=alpha[0,:,:,0].cpu().numpy();image[hand]=0
                scores=foreground_render_metrics(actual,image,mask,lpips);lpips.reset()
                predicted_mask=(alpha>.5)&~hand;scores['silhouette_iou']=float((predicted_mask&mask).sum()/max(1,(predicted_mask|mask).sum()))
                x0,y0,x1,y1=scores['crop_xyxy'];error=((image-actual)**2).mean(2)[y0:y1,x0:x1]
                mismatch=(predicted_mask^mask)[y0:y1,x0:x1]
                scores.update(rgb_mse=float(error.mean()),silhouette_mismatch_sse_fraction=float(error[mismatch].sum()/max(float(error.sum()),1e-15)))
                rows.append(dict(frame=frame,camera=camera,method=method,**scores))
                cv2.imwrite(str(out/f'{method}-camera{camera}-frame{frame}.png'),cv2.cvtColor((image*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
            print(case,frame,method,flush=True)
        (out/'progress.json').write_text(json.dumps(dict(last_frame=frame,rows=len(rows)))+'\n')
    result=dict(case=case,frames=frames,rows=rows,support=supports,transport=deformations,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (original/'final_data.pkl',fit,nativepath,Path(__file__),Path(__file__).with_name('transport.py'))},
        scope='Evaluation-only reconstructed-track oracle proxy. All methods share frame-specific valid observed anchors, including reference visibility information. This is a paired diagnostic, not new forecast performance. No appearance fitting, no new camera view; unknown/interior motion remains interpolated.')
    (out/'scores.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
