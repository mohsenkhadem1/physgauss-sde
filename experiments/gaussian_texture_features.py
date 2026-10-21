"""Local RGB texture measurements tied to the known reference Gaussian asset.

A global color curve is fitted on TRAIN reference-frame pixels only, then frozen.
Neither observed depth nor deformed geometry/visibility/correspondences is used.
The existing vertex checker has rank-one gradients per face: fitted shifts are
apparent, regularized tangent measurements, not unique material trajectories.
Known geometry and texture support explicit subpixel measurements here; a
foundation image encoder is unnecessary before this observable signal is tested.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
try:
    from gaussian_observed_features import slab_depth
except ImportError:
    from experiments.gaussian_observed_features import slab_depth

PALETTE=np.array([[.12,.30,.65],[.72,.43,.08]])


def texture_geometry(reference):
    points=np.asarray(reference['rest_points_m']);ids=reference['face_vertex_ids']
    colors=np.asarray(reference['vertex_colors']);direction=PALETTE[1]-PALETTE[0]
    vertex_w=(colors-PALETTE[0])@direction/(direction@direction)
    if not np.allclose(colors,PALETTE[0]+vertex_w[:,None]*direction,atol=1e-6):
        raise ValueError('This texture model requires the declared two-color reference palette')
    triangle=points[ids];edges=np.stack((triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),1)
    delta=np.stack((vertex_w[ids[:,1]]-vertex_w[ids[:,0]],vertex_w[ids[:,2]]-vertex_w[ids[:,0]]),-1)
    gradient=np.einsum('gij,gj->gi',np.linalg.pinv(edges),delta)
    weights=reference['barycentric_weights'];w=np.einsum('gi,gi->g',weights,vertex_w[ids])
    means=np.einsum('gi,gij->gj',weights,triangle)
    lower=points.min(0);upper=points.max(0);size=upper-lower
    distance=np.concatenate((abs(means-lower)/size,abs(means-upper)/size),-1)
    faces=distance.argmin(-1);axis=faces%3;normal=np.eye(3)[axis]*np.where(faces[:,None]<3,-1,1)
    return {'means':means,'w_reference':w,'gradient_world':gradient,'face':faces,'normal':normal,
            'interior':weights.min(-1)>.12,'lower':lower,'upper':upper}


def bilinear_rgb(rgb,mask,u,v):
    """Pixel-edge coordinates; all four neighbors must be foreground."""
    x=np.asarray(u)-.5;y=np.asarray(v)-.5;ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
    valid=(ix>=0)&(iy>=0)&(ix+1<rgb.shape[1])&(iy+1<rgb.shape[0])
    ix=np.clip(ix,0,rgb.shape[1]-2);iy=np.clip(iy,0,rgb.shape[0]-2)
    valid&=mask[iy,ix]&mask[iy,ix+1]&mask[iy+1,ix]&mask[iy+1,ix+1]
    dx=x-ix;dy=y-iy
    value=(rgb[iy,ix]*(1-dx)[:,None]*(1-dy)[:,None]+rgb[iy,ix+1]*dx[:,None]*(1-dy)[:,None]
           +rgb[iy+1,ix]*(1-dx)[:,None]*dy[:,None]+rgb[iy+1,ix+1]*dx[:,None]*dy[:,None])
    return value,valid


def static_kernel_measurements(reference,pose,k,rgb,mask):
    geometry=texture_geometry(reference);means=geometry['means']
    xyz=(means-pose[3,:3])@pose[:3,:3].T;depth=-xyz[:,2]
    u=k[0,0]*xyz[:,0]/depth+k[0,2];v=k[1,2]-k[1,1]*xyz[:,1]/depth
    rays=np.stack(((u-k[0,2])/k[0,0],-(v-k[1,2])/k[1,1],-np.ones_like(u)),-1)@pose[:3,:3]
    first=slab_depth(pose[3,:3],rays,geometry['lower'],geometry['upper'])
    static_visible=(abs(first-depth)<1e-5)&(depth>0)&geometry['interior']
    color,valid=bilinear_rgb(rgb,mask,u,v);valid&=static_visible
    normal=geometry['normal'];denominator=np.sum(normal*rays,axis=-1)
    ratio=np.divide(np.sum(geometry['gradient_world']*rays,axis=-1),denominator,
                    out=np.zeros_like(depth),where=abs(denominator)>1e-8)
    # delta-w = -J_projective * world translation for a translated plane.
    projective_gradient=geometry['gradient_world']-ratio[:,None]*normal
    return {**geometry,'observed_rgb':color,'valid':valid,'static_visible':static_visible,
            'projective_gradient_world':projective_gradient}


def fit_color_curve(w,rgb,degree=3):
    w=np.asarray(w);rgb=np.asarray(rgb);design=np.polynomial.polynomial.polyvander(w,degree)
    coefficients=np.linalg.lstsq(design,rgb,rcond=1e-10)[0]
    prediction=design@coefficients
    return {'coefficients':coefficients.tolist(),'degree':degree,'samples':len(w),
            'training_rgb_rmse':float(np.sqrt(np.mean((prediction-rgb)**2)))}


def evaluate_color_curve(curve,w):
    return np.polynomial.polynomial.polyvander(np.asarray(w),curve['degree'])@np.asarray(curve['coefficients'])


def invert_color_curve(curve,rgb):
    grid=np.linspace(0,1,2049);colors=evaluate_color_curve(curve,grid)
    origin=colors[0];direction=colors[-1]-origin;scale=float(direction@direction)
    if scale<1e-5:raise ValueError('Reference palette is not photometrically distinguishable')
    scalar=(colors-origin)@direction/scale
    monotone=np.maximum.accumulate(scalar);unique,ids=np.unique(monotone,return_index=True)
    observed=(np.asarray(rgb)-origin)@direction/scale
    w=np.interp(observed,unique,grid[ids])
    error=np.linalg.norm(np.asarray(rgb)-evaluate_color_curve(curve,w),axis=-1)
    return w,error


def fit_texture_calibration(roots,output,catalog=None,degree=3,minimum_face_samples=500):
    from train_gaussian_inverse import collect_cases,apply_catalog
    records=collect_cases(roots)
    if catalog:records=apply_catalog(records,catalog)
    records=[r for r in records if r['recipe']['split']=='train']
    values=[];colors=[];faces=[];sources=[]
    for record in records:
        episode=record['path']
        with np.load(episode/'reference.npz') as raw:reference={k:raw[k] for k in raw.files}
        with np.load(episode/'observations/cameras.npz') as raw:c={k:raw[k] for k in raw.files}
        if c['frame_ids'][0]!=0 or abs(c['times_s'][0])>1e-8:raise ValueError('Calibration requires known reference frame0')
        rgb=np.asarray(Image.open(episode/'observations/rgb/00000.png').convert('RGB'))/255.
        mask=np.asarray(Image.open(episode/'observations/masks/00000.png').convert('L'))>0
        measured=static_kernel_measurements(reference,c['camera_to_world_usd_row'][0],c['K'],rgb,mask)
        valid=measured['valid'];values.append(measured['w_reference'][valid]);colors.append(measured['observed_rgb'][valid]);faces.append(measured['face'][valid])
        sources.append({'key':record['key'],'frame':0,'samples':int(valid.sum()),
                        'rgb_sha256':hashlib.sha256((episode/'observations/rgb/00000.png').read_bytes()).hexdigest()})
    w=np.concatenate(values);rgb=np.concatenate(colors);face=np.concatenate(faces)
    global_curve=fit_color_curve(w,rgb,degree);per_face={}
    for identity in range(6):
        selected=face==identity
        if selected.sum()>=minimum_face_samples:per_face[str(identity)]=fit_color_curve(w[selected],rgb[selected],degree)
    inferred,error=invert_color_curve(global_curve,rgb)
    result={'global_curve':global_curve,'face_curves':per_face,'sources':sources,
            'global_inverse_training_w_rmse':float(np.sqrt(np.mean((inferred-w)**2))),
            'minimum_face_samples':minimum_face_samples,'calibration_rule':'TRAIN native reference-frame RGB only; unseen faces use frozen global curve',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'qualification':'Known cuboid/two-color reference asset; no VAL fitting, depth, hidden motion or true visibility'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Color calibrations are immutable; choose a new output path')
    output.write_text(json.dumps(result,indent=2)+'\n');return result


def apparent_tangent_fit(gradient_world,delta_w,face,ridge_relative=1e-4):
    axes=[i for i in range(3) if i!=face%3];g=np.asarray(gradient_world)[:,axes]
    target=-np.asarray(delta_w);normal=g.T@g
    eigen=np.linalg.eigvalsh(normal/max(len(g),1));threshold=max(float(eigen.max()),1.)*1e-6
    rank=int((eigen>threshold).sum());ridge=max(float(np.trace(normal))*.5*ridge_relative,1e-8)
    shift=np.linalg.solve(normal+ridge*np.eye(2),g.T@target)
    residual=target-g@shift
    return {'shift_tangent_m':shift,'tangent_axes':axes,'rank':rank,'eigenvalues_m_minus2':eigen,
            'residual_w':residual,'ridge_relative':ridge_relative}


def frame_texture_features(reference,pose,k,rgb,mask,calibration):
    measured=static_kernel_measurements(reference,pose,k,rgb,mask);feature=[];audits=[]
    center=(measured['lower']+measured['upper'])/2;size=measured['upper']-measured['lower']
    for face in range(6):
        selected=(measured['face']==face)&measured['valid'];eligible=(measured['face']==face)&measured['static_visible']
        curve=calibration['face_curves'].get(str(face),calibration['global_curve'])
        if selected.sum()<5:
            feature.extend([0.]*13);audits.append({'face':face,'samples':int(selected.sum()),'rank':0});continue
        observed_w,distance=invert_color_curve(curve,measured['observed_rgb'][selected])
        gate=distance<max(.06,4*curve['training_rgb_rmse']);ids=np.flatnonzero(selected)[gate]
        observed_w=observed_w[gate];distance=distance[gate]
        if len(ids)<5:
            feature.extend([0.]*13);audits.append({'face':face,'samples':len(ids),'rank':0});continue
        delta=observed_w-measured['w_reference'][ids]
        fit=apparent_tangent_fit(measured['gradient_world'][ids],delta,face)
        axes=fit['tangent_axes'];coordinate=(measured['means'][ids]-center)/size
        feature.extend([len(ids)/max(eligible.sum(),1),*list(fit['shift_tangent_m']/.01),
                        float(delta.mean()),float(delta.std()),float(np.std(fit['residual_w'])),
                        float(distance.mean()),fit['rank'],float(fit['eigenvalues_m_minus2'][0]/2500),
                        float(fit['eigenvalues_m_minus2'][1]/2500),float(abs(delta).mean()),
                        *list((delta[:,None]*coordinate[:,axes]).mean(0))])
        projective=measured['projective_gradient_world'][ids]
        audits.append({'face':face,'samples':len(ids),'rank':fit['rank'],'eigenvalues_m_minus2':fit['eigenvalues_m_minus2'].tolist(),
                       'projective_eigenvalues_m_minus2':np.linalg.eigvalsh(projective.T@projective/len(projective)).tolist(),
                       'uses_face_specific_train_calibration':str(face) in calibration['face_curves']})
    return np.asarray(feature,dtype=np.float32),audits


def load_texture_features(episode,calibration,max_time_s=None):
    """Public native RGB/masks only, optionally reading a strict time prefix."""
    episode=Path(episode)
    with np.load(episode/'reference.npz') as raw:reference={k:raw[k] for k in raw.files}
    with np.load(episode/'observations/cameras.npz') as raw:c={k:raw[k] for k in raw.files}
    selected=np.ones(len(c['times_s']),dtype=bool) if max_time_s is None else c['times_s']<=max_time_s+1e-8
    rows=[];audits=[]
    for index in np.flatnonzero(selected):
        frame=c['frame_ids'][index]
        rgb=np.asarray(Image.open(episode/'observations/rgb'/f'{frame:05d}.png').convert('RGB'))/255.
        mask=np.asarray(Image.open(episode/'observations/masks'/f'{frame:05d}.png').convert('L'))>0
        values,audit=frame_texture_features(reference,c['camera_to_world_usd_row'][index],c['K'],rgb,mask,calibration)
        rows.append(values);audits.append(audit)
    return {'features':np.asarray(rows,dtype=np.float32),'audits':audits,'frame_ids':c['frame_ids'][selected],
            'qualification':'Apparent local texture shifts; rank-one aperture ambiguity and normal-motion parallax remain'}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--roots',nargs='+',required=True)
    parser.add_argument('--catalog');parser.add_argument('--output',required=True);args=parser.parse_args()
    result=fit_texture_calibration(args.roots,args.output,args.catalog)
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','face_curves']},indent=2))
