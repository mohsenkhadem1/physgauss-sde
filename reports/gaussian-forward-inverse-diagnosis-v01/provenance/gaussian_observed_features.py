"""Measured geometry features from public masks/cameras, with optional depth.

RGB mode uses supplied image segmentation and color summaries. RGBD is a
separate, explicitly depth-assisted track. Static visibility comes exclusively
from the known rest cuboid; no deformed mesh, simulator state, truth visibility
or material correspondences enter feature extraction.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
try:
    from gaussian_observation import load_rgb_inputs
except ImportError:
    from experiments.gaussian_observation import load_rgb_inputs


def slab_depth(origins,rays,lower,upper):
    """Intersect world rays o+t*d with known reference AABB; t is axis depth.

    Camera rays must have local z=-1 rather than unit length. Infinite depth
    denotes misses. Parallel rays outside a slab are rejected explicitly.
    """
    origins=np.asarray(origins);rays=np.asarray(rays);lower=np.asarray(lower);upper=np.asarray(upper)
    parallel=abs(rays)<1e-12
    inverse=np.divide(1.,rays,out=np.zeros_like(rays,dtype=float),where=~parallel)
    a=(lower-origins)*inverse;b=(upper-origins)*inverse
    near=np.where(parallel,-np.inf,np.minimum(a,b)).max(-1)
    far=np.where(parallel,np.inf,np.maximum(a,b)).min(-1)
    outside=(parallel&((origins<lower)|(origins>upper))).any(-1)
    return np.where((far>=np.maximum(near,0))&~outside,np.maximum(near,0),np.inf)


def reference_cuboid_sensor(reference,pose,k,size,supersample=2):
    """Rest depth at pixel centers and supersampled mask from public geometry."""
    lower=reference['rest_points_m'].min(0);upper=reference['rest_points_m'].max(0)
    # The benchmark asset is a cuboid. Reject unsupported geometry rather than
    # silently interpreting an arbitrary COLMAP shape as its bounding box.
    points=reference['rest_points_m'];faces=reference.get('surface_faces')
    if faces is not None:
        surface=points[np.unique(faces)]
        if not np.all((np.isclose(surface,lower,atol=1e-7)|np.isclose(surface,upper,atol=1e-7)).any(-1)):
            raise ValueError('Exact slab reference sensor currently requires a cuboid asset')
    yy,xx=np.meshgrid(np.arange(size[1]),np.arange(size[0]),indexing='ij')
    def depth_at(dx,dy):
        camera=np.stack(((xx+dx-k[0,2])/k[0,0],-(yy+dy-k[1,2])/k[1,1],-np.ones_like(xx)),-1)
        rays=camera@pose[:3,:3]
        return slab_depth(pose[3,:3],rays,lower,upper)
    depth=depth_at(.5,.5);mask=np.zeros_like(depth)
    for dy in (np.arange(supersample)+.5)/supersample:
        for dx in (np.arange(supersample)+.5)/supersample:mask+=np.isfinite(depth_at(dx,dy))/(supersample**2)
    return {'depth_m':np.where(np.isfinite(depth),depth,0).astype(np.float32),'mask':mask.astype(np.float32)}


def load_observed_arrays(episode,observation_mode='rgb',size=(192,160)):
    """Public-only loader. RGB never opens depth; neither mode opens truth."""
    if observation_mode not in ('rgb','rgbd'):raise ValueError('Expected rgb or rgbd observation mode')
    inputs=load_rgb_inputs(episode,size,roi=True);inputs['observation_mode']=observation_mode
    inputs['native_K']=inputs['cameras']['native_K']
    if observation_mode=='rgbd':
        depth=[];masks=[]
        for frame in inputs['cameras']['frame_ids']:
            with np.load(Path(episode)/'observations/depth'/f'{frame:05d}.npz') as raw:depth.append(raw['depth_m'])
            masks.append(np.asarray(Image.open(Path(episode)/'observations/masks'/f'{frame:05d}.png'))>0)
        inputs['depth_native_m']=np.asarray(depth,dtype=np.float32);inputs['native_mask']=np.asarray(masks)
        if not np.isfinite(inputs['depth_native_m']).all():raise ValueError('Nonfinite observed depth')
    return inputs


def sample_depth_bilinear(depth,u,v):
    """Interpolate inverse camera depth, exact on planes; reject mixed edges.

    Coordinates use pixel edges (center i+.5). Background and large range
    discontinuities are excluded; linear-z interpolation biases grazing planes.
    """
    depth=np.asarray(depth);x=np.asarray(u)-.5;y=np.asarray(v)-.5
    ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
    valid=(ix>=0)&(iy>=0)&(ix+1<depth.shape[1])&(iy+1<depth.shape[0])
    ix=np.clip(ix,0,depth.shape[1]-2);iy=np.clip(iy,0,depth.shape[0]-2)
    values=np.stack((depth[iy,ix],depth[iy,ix+1],depth[iy+1,ix],depth[iy+1,ix+1]),-1)
    valid&=(np.isfinite(values)&(values>0)).all(-1)
    dx=x-ix;dy=y-iy
    valid&=(values.max(-1)-values.min(-1))<np.maximum(.02,.03*np.median(values,axis=-1))
    inverse=np.divide(1.,values,out=np.zeros_like(values),where=values>0)
    interpolated=inverse[...,0]*(1-dx)*(1-dy)+inverse[...,1]*dx*(1-dy)+inverse[...,2]*(1-dx)*dy+inverse[...,3]*dx*dy
    sampled=np.divide(1.,interpolated,out=np.zeros_like(interpolated),where=interpolated>0)
    return np.where(valid,sampled,0),valid


def mask_geometry(mask,k,reference_depth):
    """Area, centroid, covariance and32 silhouette support functions in metres."""
    yy,xx=np.meshgrid(np.arange(mask.shape[0])+.5,np.arange(mask.shape[1])+.5,indexing='ij')
    uv=np.stack(((xx-k[0,2])/k[0,0]*reference_depth,-(yy-k[1,2])/k[1,1]*reference_depth),-1)
    weight=np.asarray(mask,dtype=float);total=weight.sum()
    if total<1:raise ValueError('Observed object mask is empty')
    center=(weight[...,None]*uv).sum((0,1))/total
    delta=uv-center;cov=np.einsum('hw,hwi,hwj->ij',weight,delta,delta)/total
    occupied=uv[weight>.5]
    if not len(occupied):raise ValueError('No confident object pixels')
    angles=np.arange(32)*2*np.pi/32;directions=np.stack((np.cos(angles),np.sin(angles)),-1)
    support=(occupied@directions.T).max(0)
    return np.r_[total,center,cov[0,0],cov[1,1],cov[0,1],support]


def projective_depth_features(reference,pose,k,depth):
    """Eulerian range residuals at static material kernels, not true tracks."""
    means=np.asarray(reference['means_m'],dtype=float);lower=reference['rest_points_m'].min(0);upper=reference['rest_points_m'].max(0)
    center=(lower+upper)/2;size=upper-lower
    xyz=(means-pose[3,:3])@pose[:3,:3].T;z=-xyz[:,2]
    u=k[0,0]*xyz[:,0]/z+k[0,2];v=k[1,2]-k[1,1]*xyz[:,1]/z
    camera_rays=np.stack(((u-k[0,2])/k[0,0],-(v-k[1,2])/k[1,1],-np.ones_like(u)),-1)
    known_depth=slab_depth(pose[3,:3],camera_rays@pose[:3,:3],lower,upper)
    static_visible=(abs(known_depth-z)<1e-5)&(z>0)
    measured,valid=sample_depth_bilinear(depth,u,v)
    valid&=static_visible&(abs(measured-z)<.05)
    residual=(measured-z)/.01
    distances=np.concatenate((abs(means-lower)/size,abs(means-upper)/size),-1)
    face=distances.argmin(-1)
    results=[]
    for face_id in range(6):
        eligible=(face==face_id)&static_visible;selected=eligible&valid
        ratio=selected.sum()/max(eligible.sum(),1)
        if selected.sum()<3:
            results.extend([ratio]+[0.]*8);continue
        r=residual[selected];coords=(means[selected]-center)/size
        results.extend([ratio,float(r.mean()),float(r.std()),*np.quantile(r,[.1,.5,.9]).tolist(),*(r[:,None]*coords).mean(0).tolist()])
    return np.asarray(results,dtype=np.float32),{'static_visible_kernels':int(static_visible.sum()),'valid_depth_kernels':int(valid.sum())}


def extract_features(inputs):
    """Return[observations,D] features from loader arrays; depth is opt-in."""
    mode=inputs['observation_mode'];ref=inputs['reference'];c=inputs['cameras']
    center=(ref['rest_points_m'].min(0)+ref['rest_points_m'].max(0))/2
    extent=ref['rest_points_m'].max(0)-ref['rest_points_m'].min(0)
    rows=[];audits=[];size=inputs['calibration_size']
    names=['relative_area','centroid_x_10mm','centroid_y_10mm','cov_xx_100mm2','cov_yy_100mm2','cov_xy_100mm2']+[f'support_{i}_10mm' for i in range(32)]
    names += [f'rgb_mean_{i}' for i in range(3)]+[f'rgb_std_{i}' for i in range(3)]
    names += [f'camera_{i}' for i in range(18)]+[f'reference_extent_{i}' for i in range(3)]
    if mode=='rgbd':names += [f'depth_face_{face}_{field}' for face in range(6) for field in ['valid_fraction','mean_10mm','std_10mm','q10_10mm','q50_10mm','q90_10mm','moment_x','moment_y','moment_z']]
    for index,pose in enumerate(c['camera_to_world_usd_row']):
        k=c['K'][index];sensor=reference_cuboid_sensor(ref,pose,k,size)
        zcenter=-((center-pose[3,:3])@pose[:3,:3].T)[2]
        measured=mask_geometry(inputs['mask'][index],k,zcenter);expected=mask_geometry(sensor['mask'],k,zcenter)
        geometric=measured-expected;geometric[0]/=expected[0];geometric[1:3]/=.01;geometric[3:6]/=.0001;geometric[6:]/=.01
        rgb=inputs['rgb'][index];weight=inputs['mask'][index];total=weight.sum()
        color=(rgb*weight[...,None]).sum((0,1))/total
        color_std=np.sqrt(((rgb-color)**2*weight[...,None]).sum((0,1))/total)
        camera=np.r_[(pose[3,:3]-center)/np.linalg.norm(extent),pose[:3,:3].ravel(),
                     k[0,0]/size[0],k[1,1]/size[1],k[0,2]/size[0],k[1,2]/size[1],
                     c['times_s'][index]/3.,np.sin(np.pi*c['times_s'][index]/3.)]
        row=np.r_[geometric,color,color_std,camera,extent/.1];audit={}
        if mode=='rgbd':
            depth,audit=projective_depth_features(ref,pose,inputs['native_K'],inputs['depth_native_m'][index]);row=np.r_[row,depth]
        rows.append(row);audits.append(audit)
    result=np.asarray(rows,dtype=np.float32)
    if result.shape[-1]!=len(names) or not np.isfinite(result).all():raise ValueError('Malformed observation features')
    return {'features':result,'feature_names':names,'audits':audits,
            'observation_mode':mode,'qualification':'Known cuboid rest prior; segmentation supplied; RGBD uses exact simulated depth only when declared'}


class ObservedFeatureEncoder(nn.Module):
    def __init__(self,feature_dim,context_dim=128,hidden=128,mode='features'):
        super().__init__();self.mode=mode;self.context_dim=context_dim
        if mode not in ('features','prior'):raise ValueError('Expected features or prior mode')
        self.input=nn.Sequential(nn.Linear(feature_dim,hidden),nn.LayerNorm(hidden),nn.SiLU(),nn.Linear(hidden,128))
        layer=nn.TransformerEncoderLayer(128,4,256,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.attention=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.output=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,context_dim))

    def forward(self,features,observation_mask=None):
        if self.mode=='prior':return features.new_zeros((len(features),self.context_dim))
        if observation_mask is None:observation_mask=torch.ones(features.shape[:2],dtype=torch.bool,device=features.device)
        if not observation_mask.any(-1).all():raise ValueError('Each clip needs an observation')
        encoded=self.attention(self.input(features),src_key_padding_mask=~observation_mask)
        pooled=(encoded*observation_mask[...,None]).sum(1)/observation_mask.sum(-1,keepdim=True)
        return self.output(pooled)
