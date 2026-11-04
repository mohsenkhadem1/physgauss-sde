"""Common frozen Gaussian appearance, fixed-frame future-image comparison.

Uses initial-frame local rigid skinning identically for all physical rollouts.
This is a controlled renderer, not an exact reproduction of upstream's
incremental skinning or a novel-camera benchmark. No appearance is refitted.
"""
from scripts.real_transfer_v19 import deform360_reconstruct  # CUDA toolchain only
import argparse,json,pickle,os
from pathlib import Path
import cv2,numpy as np,torch,gsplat
from scipy.spatial import cKDTree
from plyfile import PlyData
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from .metrics import foreground_render_metrics
from scripts.external_benchmark.data import phystwin
from scripts.external_benchmark.evaluate import expand_prediction

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/real-transfer-v20';ASSETS=BASE/'render-assets'

def load_gaussians(case):
    path=next((ASSETS/'gaussian_output'/case).glob('*/point_cloud/iteration_10000/point_cloud.ply'))
    v=PlyData.read(str(path))['vertex'];stack=lambda names:np.stack([v[n] for n in names],axis=1)
    opacity=1/(1+np.exp(-np.asarray(v['opacity'])));keep=opacity>.1
    xyz=stack(('x','y','z'))[keep]
    scales=np.exp(stack(('scale_0',)))[keep].repeat(3,axis=1)
    quat=stack(('rot_0','rot_1','rot_2','rot_3'))[keep]
    dc=stack(('f_dc_0','f_dc_1','f_dc_2'))[keep,None,:]
    rest=stack([f'f_rest_{i}' for i in range(45)])[keep].reshape(-1,3,15).transpose(0,2,1)
    return xyz,scales,quat,opacity[keep],np.concatenate([dc,rest],axis=1)

def skinning_setup(initial,xyz):
    tree=cKDTree(initial);neighbors=tree.query(initial,k=17)[1][:,1:]
    distances,ids=tree.query(xyz,k=16)
    weights=1/np.maximum(distances,1e-4)**2;weights/=weights.sum(1,keepdims=True)
    return neighbors,ids,weights

def skin(initial,current,xyz,setup):
    neighbors,ids,weights=setup
    a=initial[neighbors]-initial[:,None];b=current[neighbors]-current[:,None]
    covariance=np.einsum('nki,nkj->nij',a,b)
    u,_,vt=np.linalg.svd(covariance);sign=np.ones((len(initial),3));sign[:,-1]=np.linalg.det(u@vt)
    rotation=(u*sign[:,None,:])@vt
    output=np.empty_like(xyz)
    for start in range(0,len(xyz),20000):
        end=min(start+20000,len(xyz));index=ids[start:end]
        transformed=np.einsum('gki,gkij->gkj',xyz[start:end,None]-initial[index],rotation[index])+current[index]
        output[start:end]=np.einsum('gk,gki->gi',weights[start:end],transformed)
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',default='single_lift_sloth')
    parser.add_argument('--label',default='v20-selected-mesh240');parser.add_argument('--nodes',type=int,default=240)
    parser.add_argument('--transfer-source',default=None)
    parser.add_argument('--initial-only',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    case=args.case;data=phystwin(args.transfer_source or case,args.nodes);flip=data['evaluation']['transform']
    initial=data['support_geometry']@flip
    original=ROOT/'runs/third_party/PhysTwin/data/different_types'/case
    local=ASSETS/'data/different_types'/case
    metadata=json.loads((original/'metadata.json').read_text());K=np.array(metadata['intrinsics'])
    with (original/'calibrate.pkl').open('rb') as f:C=np.asarray(pickle.load(f))
    frame_plan=json.loads((ASSETS/f'{case}-frames.json').read_text());frames=[0] if args.initial_only else frame_plan['frames']
    xyz,scales,quat,opacity,sh=load_gaussians(args.transfer_source or case)
    if args.transfer_source:
        registrations=json.loads((ROOT/'reports/real-transfer-v20/initial-registration.json').read_text())['pairs']
        reg=next(r for r in registrations if r['source']==args.transfer_source and r['target']==case)
        R=flip@np.array(reg['rotation'])@flip;translation=np.array(reg['translation_m'])@flip
        initial=initial@R+translation;xyz=xyz@R+translation
    setup=skinning_setup(initial,xyz)
    predictions={}
    if args.initial_only:predictions['shared_initial_appearance']=np.broadcast_to(initial,(1,*initial.shape))
    elif args.transfer_source:
        for method in ('neural','analytic','native_frozen'):
            folder=(BASE/'native-transfer'/f'{args.transfer_source}--{case}') if method=='native_frozen' else (BASE/'frozen-transfer/refined-frozen'/f'{args.transfer_source}--{case}'/method)
            with np.load(folder/'prediction.npz') as f:predictions[method]=f['predicted']@flip
    else:
        for method in ('neural','analytic'):
            path=ROOT/'runs/real-transfer-v19/fits'/args.label/case/method/'prediction.npz'
            with np.load(path) as f:predictions[method]=expand_prediction(data,f['predicted'])@flip
        with (ROOT/'runs/third_party/PhysTwin/reference/experiments'/case/'inference.pkl').open('rb') as f:
            predictions['official_phystwin']=np.asarray(pickle.load(f))[:,:len(initial)]
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=False).cuda().eval()
    output_label=('transfer-'+args.transfer_source) if args.transfer_source else args.label
    out=BASE/'render-audit'/output_label/case;out.mkdir(parents=True,exist_ok=True)
    tensor=lambda v:torch.as_tensor(np.asarray(v),dtype=torch.float32,device='cuda')
    constant=dict(scales=tensor(scales),quats=tensor(quat),opacities=tensor(opacity),colors=tensor(sh),sh_degree=3)
    rows=[];qa=[]
    for frame in frames:
        warped={method:skin(initial,pred[frame],xyz,setup) for method,pred in predictions.items()}
        for camera in range(3):
            labels=json.loads((local/f'mask/mask_info_{camera}.json').read_text())
            object_id=next(k for k,v in labels.items() if v!='hand');hand_ids=[k for k,v in labels.items() if v=='hand']
            actual=cv2.cvtColor(cv2.imread(str(local/f'color/{camera}/{frame}.png')),cv2.COLOR_BGR2RGB).astype(float)/255
            mask=cv2.imread(str(local/f'mask/{camera}/{object_id}/{frame}.png'),0)>0
            hand=np.zeros(mask.shape,bool)
            for i in hand_ids:hand|=cv2.imread(str(local/f'mask/{camera}/{i}/{frame}.png'),0)>0
            actual*=mask[:,:,None];height,width=mask.shape;panels=[actual]
            for method,means in warped.items():
                with torch.no_grad():
                    rgb,alpha,_=gsplat.rasterization(means=tensor(means),**constant,Ks=tensor(K[camera])[None],
                        viewmats=tensor(np.linalg.inv(C[camera]))[None],width=width,height=height,render_mode='RGB')
                image=np.clip(rgb[0].cpu().numpy(),0,1);alpha=alpha[0,:,:,0].cpu().numpy()
                image[hand]=0
                metrics=foreground_render_metrics(actual,image,mask,lpips);lpips.reset()
                visible=(alpha>.5)&~hand
                metrics['silhouette_iou']=float((visible&mask).sum()/max(1,(visible|mask).sum()))
                rows.append(dict(frame=frame,camera=camera,method=method,**metrics))
                if frame==frames[0]:panels.append(image)
                cv2.imwrite(str(out/f'{method}-camera{camera}-frame{frame}.png'),cv2.cvtColor((image*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
            if frame==frames[0]:qa.append(np.concatenate([cv2.resize(p,(424,240)) for p in panels],axis=1))
    status=dict(case=case,source=args.transfer_source,rows=rows,gaussians=len(xyz),frames=frames,
        scope='Fixed-frame controlled rendering audit. Shared released appearance, fixed opacity >0.1 pruning, initial local rigid skinning, isotropic scales and fixed SH coefficients. Hand occlusions removed equally. No target appearance fitting; initial three cameras were used to train appearance, so this is not novel-view evaluation.',
        transfer_note='If source is set: source appearance rigidly placed using frame-zero registration; shared fixed SH coefficients, no target appearance fitting. Adapted frozen native baseline, not author benchmark rendering.' if args.transfer_source else None,
        columns=['actual']+list(predictions))
    name='initial-qa' if args.initial_only else 'scores'
    (out/f'{name}.json').write_text(json.dumps(status,indent=2)+'\n')
    cv2.imwrite(str(out/f'{name}.png'),cv2.cvtColor((np.clip(np.concatenate(qa),0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
    print(json.dumps(dict(case=case,rows=len(rows),output=str(out)),indent=2))

if __name__=='__main__':main()
