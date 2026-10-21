"""Camera-correct Gaussian observation model for the Isaac orbit benchmark.

All geometry and appearance come from the known reference asset. Gaussian
covariance describes spatial support, never posterior uncertainty. RGB-only
input loading deliberately excludes the optional observed-depth channel.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image
import torch
from torch import nn


def project_reference_crops(reference, poses, k, size=(192,160), margin=30, native_size=(640,480)):
    """Compute query crops exclusively from known rest geometry and cameras.

    Boxes are integer native image coordinates[x0,y0,x1,y1], clipped to sensor.
    No dynamic mask, hidden image, depth or deformed state participates.
    """
    points=np.asarray(reference['rest_points_m']);poses=np.asarray(poses)
    calibrations=np.broadcast_to(np.asarray(k),(*poses.shape[:-2],3,3)).reshape(-1,3,3)
    flatposes=poses.reshape(-1,4,4);boxes=[];adjusted=[]
    for pose,cal in zip(flatposes,calibrations):
        xyz=(points-pose[3,:3])@pose[:3,:3].T;depth=-xyz[:,2]
        if np.any(depth<=0):raise ValueError('Rest geometry behind query camera')
        uv=np.stack((cal[0,0]*xyz[:,0]/depth+cal[0,2],cal[1,2]-cal[1,1]*xyz[:,1]/depth),-1)
        lower=np.maximum(np.floor(uv.min(0)-margin).astype(int),0)
        upper=np.minimum(np.ceil(uv.max(0)+margin).astype(int),native_size)
        if np.any(upper<=lower):raise ValueError('Reference crop outside sensor')
        box=np.concatenate((lower,upper));new=cal.copy()
        new[0,2]-=lower[0];new[1,2]-=lower[1]
        new[0]*=size[0]/(upper[0]-lower[0]);new[1]*=size[1]/(upper[1]-lower[1])
        boxes.append(box);adjusted.append(new)
    return {'K':np.asarray(adjusted).reshape(*poses.shape[:-2],3,3),
            'crop_boxes_xyxy':np.asarray(boxes).reshape(*poses.shape[:-2],4),
            'calibration_size':tuple(size),'native_size':tuple(native_size),
            'crop_rule':'Known rest geometry projection plus fixed native pixel margin', 'margin_native_pixels':margin}


def load_rgb_inputs(episode, size=(96, 72), roi=False, margin=30):
    """Return public RGB, masks, cameras and reference; never open truth files.

    ROI=True uses projection of known rest geometry only, and returns per-frame K
    already calibrated to output size. The renderer then needs calibration_size=size.
    """
    episode = Path(episode)
    with np.load(episode / 'reference.npz') as f:
        reference = {k: f[k] for k in f.files}
    with np.load(episode / 'observations/cameras.npz') as f:
        cameras = {k: f[k] for k in f.files}
    ids = cameras['frame_ids'];crop=None
    if roi:
        crop=project_reference_crops(reference,cameras['camera_to_world_usd_row'],cameras['K'],size,margin)
        cameras={**cameras,'native_K':cameras['K'],'K':crop['K'],'crop_boxes_xyxy':crop['crop_boxes_xyxy']}
    images=[];masks=[]
    for index,frame in enumerate(ids):
        image=Image.open(episode/'observations/rgb'/f'{frame:05d}.png').convert('RGB')
        mask=Image.open(episode/'observations/masks'/f'{frame:05d}.png').convert('L')
        if roi:
            box=tuple(map(int,crop['crop_boxes_xyxy'][index]));image=image.crop(box);mask=mask.crop(box)
        images.append(np.asarray(image.resize(size,Image.Resampling.BILINEAR))/255.)
        masks.append(np.asarray(mask.resize(size,Image.Resampling.BILINEAR))/255.)
    return {'reference':reference,'cameras':cameras,'rgb':np.asarray(images,dtype=np.float32),
            'mask':np.asarray(masks,dtype=np.float32),'crop':crop,
            'calibration_size':tuple(size) if roi else (640,480),
            'input_channels':'RGB and supplied masks; no depth or hidden motion'}


def usd_projection(points, poses, k):
    """Project [T,N,3] world points using USD row poses; return pixels/depth/J."""
    xyz = torch.einsum('tnj,tij->tni', points-poses[:, None, 3, :3], poses[:, :3, :3])
    depth = -xyz[..., 2]
    safe = depth.clamp_min(1e-5)
    uv = torch.stack((k[0, 0]*xyz[..., 0]/safe+k[0, 2], k[1, 2]-k[1, 1]*xyz[..., 1]/safe), -1)
    jac = points.new_zeros((*points.shape[:-1], 2, 3))
    jac[..., 0, 0] = k[0, 0]/safe
    jac[..., 1, 1] = -k[1, 1]/safe
    jac[..., 0, 2] = k[0, 0]*xyz[..., 0]/safe.square()
    jac[..., 1, 2] = -k[1, 1]*xyz[..., 1]/safe.square()
    return uv, depth, jac


class IsaacGaussianRenderer(nn.Module):
    """Dense reference splatter. Render small frame batches to bound memory.

    Asset reference RGB is converted from linear to sRGB. A fixed background
    (estimated from observed frame 0 outside its supplied mask) may be passed.
    No appearance fitting, per-frame color correction or truth visibility is used.
    """
    def __init__(self, reference, size=(128, 96), device='cpu', dtype=torch.float32,
                 antialias_variance=.16, opacity_scale=1., chunk_size=256,
                 calibration_size=(640,480), color_matrix=None, color_bias=None):
        super().__init__()
        self.size = tuple(size)
        self.calibration_size=tuple(calibration_size)
        self.antialias_variance = float(antialias_variance)
        self.opacity_scale = float(opacity_scale)
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1: raise ValueError("Positive Gaussian chunk size required")
        def buf(name, value, integer=False):
            self.register_buffer(name, torch.as_tensor(value, device=device, dtype=torch.long if integer else dtype))
        rest = reference['rest_points_m']; tets = reference['tetrahedra']
        # Every surface rendering triangle belongs to exactly one volume element.
        by_face = {}
        for index, tet in enumerate(tets):
            for local in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
                key = tuple(sorted(int(tet[i]) for i in local))
                by_face.setdefault(key, []).append(index)
        incident=[]
        for face in reference['face_vertex_ids']:
            found = by_face[tuple(sorted(map(int,face)))]
            if len(found)!=1: raise ValueError('Rendering point is not on a boundary triangle')
            incident.append(found[0])
        buf('rest',rest); buf('tets',tets,True)
        buf('face_ids',reference['face_vertex_ids'],True)
        buf('weights',reference['barycentric_weights']);buf('incident',incident,True)
        dm = np.stack((rest[tets[:,1]]-rest[tets[:,0]], rest[tets[:,2]]-rest[tets[:,0]], rest[tets[:,3]]-rest[tets[:,0]]),-1)
        buf('dm_inv',np.linalg.inv(dm))
        buf('cov0',np.array([np.diag(s*s) for s in reference['scales_m']]))
        linear=np.asarray(reference['colors'])
        srgb=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
        if color_matrix is not None:srgb=srgb@np.asarray(color_matrix)+np.asarray(color_bias)
        buf('colors',np.clip(srgb,0,1));buf('opacity',reference['opacities'])
        yy,xx=torch.meshgrid(torch.arange(size[1],device=device,dtype=dtype)+.5,torch.arange(size[0],device=device,dtype=dtype)+.5,indexing='ij')
        self.register_buffer('pixels',torch.stack((xx,yy),-1))

    def forward(self, positions, poses, k, background=None):
        """Render [...,frames,nodes,3]; shared [frames,4,4] poses broadcast.

        Supports leading Monte Carlo sample/batch axes. Frames render separately;
        depth-sorted Gaussian chunks bound forward temporary memory. Autograd checkpoints
        Gaussian chunks to bound fitting memory; use small frame/sample batches.
        """
        positions=torch.as_tensor(positions,device=self.rest.device,dtype=self.rest.dtype)
        poses=torch.as_tensor(poses,device=self.rest.device,dtype=self.rest.dtype)
        k=torch.as_tensor(k,device=self.rest.device,dtype=self.rest.dtype)
        if positions.ndim<3 or positions.shape[-2:]!=self.rest.shape:
            raise ValueError('Expected [...,frames,reference_nodes,3]')
        leading=positions.shape[:-2]
        poses=torch.broadcast_to(poses,(*leading,4,4)).reshape(-1,4,4)
        k=torch.broadcast_to(k,(*leading,3,3)).reshape(-1,3,3)
        positions=positions.reshape(-1,*self.rest.shape)
        bg=self.rest.new_tensor([.89,.89,.89]) if background is None else torch.as_tensor(background,device=self.rest.device,dtype=self.rest.dtype)
        bg=torch.broadcast_to(bg,(*leading,3)).reshape(-1,3)
        images=[];masks=[]
        for index in range(len(positions)):
            result=self._frame(positions[index:index+1],poses[index:index+1],k[index],bg[index])
            images.append(result['rgb'][0]);masks.append(result['alpha'][0])
        return {'rgb':torch.stack(images).reshape(*leading,self.size[1],self.size[0],3),
                'alpha':torch.stack(masks).reshape(*leading,self.size[1],self.size[0])}

    def _frame(self,positions,poses,k,background):
        k=k.clone();k[0]*=self.size[0]/self.calibration_size[0];k[1]*=self.size[1]/self.calibration_size[1]
        centers=torch.einsum('gi,tgij->tgj',self.weights,positions[:,self.face_ids])
        x=positions[:,self.tets]
        ds=torch.stack((x[:,:,1]-x[:,:,0],x[:,:,2]-x[:,:,0],x[:,:,3]-x[:,:,0]),-1)
        f=(ds@self.dm_inv)[:,self.incident]
        cov=f@self.cov0@f.transpose(-1,-2)
        mean,depth,jac=usd_projection(centers,poses,k)
        r=poses[:,:3,:3]
        cov_cam=r[:,None]@cov@r[:,None].transpose(-1,-2)
        cov2=jac@cov_cam@jac.transpose(-1,-2)+torch.eye(2,device=k.device,dtype=k.dtype)*self.antialias_variance
        order=depth.argsort(-1)[0]
        mean=mean[:,order];cov2=cov2[:,order];depth=depth[:,order]
        colors=self.colors[order];opacity=self.opacity[order]
        transmission=positions.new_ones((1,self.size[1],self.size[0]))
        rgb=positions.new_zeros((1,self.size[1],self.size[0],3))
        for start in range(0,len(order),self.chunk_size):
            stop=min(start+self.chunk_size,len(order))
            args=(mean[:,start:stop],cov2[:,start:stop],opacity[start:stop],colors[start:stop],depth[:,start:stop])
            if torch.is_grad_enabled() and positions.requires_grad:
                from torch.utils.checkpoint import checkpoint
                contribution,remaining=checkpoint(self._composite_chunk,*args,use_reentrant=False)
            else:contribution,remaining=self._composite_chunk(*args)
            rgb=rgb+transmission[...,None]*contribution
            transmission=transmission*remaining
        return {'rgb':rgb+transmission[...,None]*background,'alpha':1-transmission}


    def _composite_chunk(self,mu,cv,opacity,color,depth):
        det=torch.linalg.det(cv).clamp_min(1e-8)
        delta=self.pixels[None,None]-mu[:,:,None,None]
        dx,dy=delta.unbind(-1)
        a,b,d=cv[...,0,0],cv[...,0,1],cv[...,1,1]
        mahal=(d[...,None,None]*dx.square()-2*b[...,None,None]*dx*dy+a[...,None,None]*dy.square())/det[...,None,None]
        alpha=(self.opacity_scale*opacity[None,:,None,None]*torch.exp(-.5*mahal)).clamp(0,.995)
        alpha=alpha*(depth>0)[:,:,None,None]
        local=torch.cumprod(torch.cat((torch.ones_like(alpha[:,:1]),1-alpha),1),1)
        contribution=torch.einsum('tghw,gc->thwc',alpha*local[:,:-1],color)
        return contribution,local[:,-1]


def fit_train_appearance(root,output,device='cuda:3',size=(192,160),margin=30,ridge=1e-3):
    """Freeze one global affine Gaussian color correction on TRAIN reference frame 0.

    No test/validation image or deformed state is opened. Ridge anchors the
    underdetermined three-color mapping toward identity; no per-object correction.
    """
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    xtx=np.zeros((4,4));xty=np.zeros((4,3));pixels=0;sources=[];input_hashes={}
    for case in plan['episodes']:
        if case['split']!='train':continue
        episode=root/'episodes'/case['episode_id'];inputs=load_rgb_inputs(episode,size,roi=True,margin=margin)
        ref=inputs['reference'];c=inputs['cameras'];bg=np.median(inputs['rgb'][0][inputs['mask'][0]<.01],axis=0)
        renderer=IsaacGaussianRenderer(ref,size,device,calibration_size=size)
        with torch.no_grad():pred=renderer(ref['rest_points_m'][None],c['camera_to_world_usd_row'][:1],c['K'][:1],bg)
        rgb=pred['rgb'][0].cpu().numpy();alpha=pred['alpha'][0].cpu().numpy()
        valid=(inputs['mask'][0]>.99)&(alpha>.99)
        x=np.concatenate((rgb[valid],np.ones((valid.sum(),1))),-1);y=inputs['rgb'][0][valid]
        xtx+=x.T@x;xty+=x.T@y;pixels+=len(x);sources.append(case['episode_id'])
        names=['reference.npz','observations/cameras.npz','observations/rgb/00000.png','observations/masks/00000.png']
        input_hashes[case['episode_id']]={name:hashlib.sha256((episode/name).read_bytes()).hexdigest() for name in names}
    if pixels<100:raise ValueError('Too few training reference pixels')
    prior=np.concatenate((np.eye(3),np.zeros((1,3))),0);strength=ridge*pixels
    coefficients=np.linalg.solve(xtx+strength*np.eye(4),xty+strength*prior)
    result={'color_matrix':coefficients[:3].tolist(),'color_bias':coefficients[3].tolist(),
            'training_episodes':sources,'training_pixels':pixels,'training_input_sha256':input_hashes,
            'plan_sha256':plan.get('plan_sha256'),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'resolution':list(size),'margin_native_pixels':margin,
            'ridge_per_pixel':ridge,'rule':'One global affine correction, TRAIN frame 0 only; fixed for all validation/test times',
            'qualification':'Sensor/shader appearance calibration; not material or force inference'}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


def image_metrics(prediction, target_rgb, target_mask):
    rgb=np.asarray(prediction['rgb']);alpha=np.asarray(prediction['alpha'])
    target_rgb=np.asarray(target_rgb);target_mask=np.asarray(target_mask)>.5
    mse=float(np.mean((rgb-target_rgb)**2));pred_mask=alpha>.5
    union=(pred_mask|target_mask).sum()
    return {'psnr_db':float(-10*np.log10(max(mse,1e-12))),
            'mask_iou':float((pred_mask&target_mask).sum()/max(union,1)),
            'foreground_rgb_rmse':float(np.sqrt(np.mean((rgb[target_mask]-target_rgb[target_mask])**2))) if target_mask.any() else 0.}


def run_static(root, output, device='cuda:3', size=(128,96), episodes=None, allow_test=False):
    """Evaluation truth is opened only after predictions, never by input loader."""
    rows=[]
    paths=sorted((Path(root)/'episodes').glob('e*'))
    plan=json.loads((Path(root)/'plan.json').read_text())
    splits={c['episode_id']:c['split'] for c in plan['episodes']}
    if episodes:paths=[p for p in paths if p.name in episodes]
    if not allow_test:
        if episodes and any(splits[p.name]=='test' for p in paths):raise ValueError('Test groups are locked; freeze protocol before explicit allow_test')
        paths=[p for p in paths if splits[p.name]!='test']
    for path in paths:
        observed=load_rgb_inputs(path,size);ref=observed['reference']
        observed0mask=observed['mask'][0]<.01
        background=np.median(observed['rgb'][0][observed0mask],axis=0)
        renderer=IsaacGaussianRenderer(ref,size,device)
        # Evaluation camera calibration is an allowed query; images remain hidden.
        with np.load(path/'truth/cameras.npz') as raw:cameras={k:raw[k] for k in raw.files}
        heldout=np.setdiff1d(cameras['frame_ids'],observed['cameras']['frame_ids'])
        scores=[]
        for frame in heldout:
            with torch.no_grad():
                pred=renderer(ref['rest_points_m'][None],cameras['camera_to_world_usd_row'][[frame]],cameras['K'],background)
            pred={k:v[0].cpu().numpy() for k,v in pred.items()}
            target=np.asarray(Image.open(path/'truth/rgb'/f'{frame:05d}.png').resize(size,Image.Resampling.BILINEAR))/255.
            mask=np.asarray(Image.open(path/'truth/masks'/f'{frame:05d}.png').resize(size,Image.Resampling.BILINEAR))/255.
            scores.append(image_metrics(pred,target,mask))
        row={'episode_id':path.name,'heldout_frames':len(heldout),**{k:float(np.mean([s[k] for s in scores])) for k in scores[0]}}
        rows.append(row);print(json.dumps(row),flush=True)
    result={'baseline':'static known reference Gaussian asset; no deformation',
            'inputs':'RGB/masks only to estimate constant background; known reference and cameras',
            'renderer':'4,096 material-bound Gaussians; full tet F transport; fixed sRGB appearance; alpha composite',
            'resolution':list(size),'rows':rows,
            'qualification':'Withheld orbit frames jointly change time and viewpoint; Gaussian shader differs from Isaac RGB',
            'aggregate':{k:float(np.mean([r[k] for r in rows])) for k in ['psnr_db','mask_iou','foreground_rgb_rmse']}}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='runs/isaac-boundary-orbit-v01')
    parser.add_argument('--output',default='runs/isaac-static-gaussian.json')
    parser.add_argument('--device',default='cuda:3');parser.add_argument('--episodes',nargs='*')
    parser.add_argument('--allow-test',action='store_true',help='Only after evaluation protocol is frozen')
    args=parser.parse_args();run_static(args.root,args.output,args.device,episodes=args.episodes,allow_test=args.allow_test)
