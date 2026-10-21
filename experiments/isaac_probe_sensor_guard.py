"""Well-conditioned independent ray checks for synchronous Isaac captures.

Near-grazing intersections amplify float32 camera/triangle rounding by1/cosθ.
The freshness tolerance remains2µm; deterministic ray selection excludes cosθ
below.2 and still checks64 distinct current-surface rays. This is a sensor
numerics criterion, never a physics/J/fixture/trajectory acceptance change.
"""
import numpy as np


def ray_depth_conditioning(points,faces,pose,k,xs,ys):
    points=np.asarray(points,dtype=float);pose=np.asarray(pose,dtype=float);k=np.asarray(k,dtype=float)
    rays=np.stack([(xs+.5-k[0,2])/k[0,0],-(ys+.5-k[1,2])/k[1,1],-np.ones(len(xs))],-1)@pose[:3,:3]
    tri=points[faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    h=np.cross(rays[:,None],e2[None]);det=(e1[None]*h).sum(-1)
    inv=np.divide(1,det,out=np.zeros_like(det),where=abs(det)>1e-10)
    s=pose[3,:3]-tri[:,0];u=inv*(s[None]*h).sum(-1);q=np.cross(s,e1)
    v=inv*(rays[:,None]*q[None]).sum(-1);t=inv*(e2*q).sum(-1)[None]
    intersections=np.where((u>=0)&(v>=0)&(u+v<=1)&(t>0)&(abs(det)>1e-10),t,np.inf)
    closest=intersections.argmin(1);depth=intersections[np.arange(len(rays)),closest]
    normal=np.cross(e1[closest],e2[closest]);denominator=np.linalg.norm(normal,axis=-1)*np.linalg.norm(rays,axis=-1)
    cosine=np.divide(abs((normal*rays).sum(-1)),denominator,out=np.zeros_like(depth),where=denominator>0)
    return depth,np.where(np.isfinite(depth),cosine,0.),closest


def well_conditioned_depth_state_agreement(points,faces,pose,k,depth,mask,
        samples=64,tolerance_m=2e-6,minimum_incidence_cosine=.2):
    if depth.shape!=(480,640) or mask.shape!=depth.shape:return {'passed':False,'reason':'sensor shape'}
    interior=np.zeros_like(mask)
    interior[1:-1,1:-1]=(mask[1:-1,1:-1]&mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:])
    ys,xs=np.where(interior)
    if len(xs)<samples:return {'passed':False,'reason':'insufficient interior foreground'}
    count=min(len(xs),samples*4);candidate=np.linspace(0,len(xs)-1,count).astype(int);xs,ys=xs[candidate],ys[candidate]
    expected,cosine,_=ray_depth_conditioning(points,faces,pose,k,xs,ys)
    finite=np.isfinite(expected);conditioned=finite&(cosine>=minimum_incidence_cosine)
    eligible=np.flatnonzero(conditioned)
    raw_error=float(abs(expected[finite]-depth[ys[finite],xs[finite]]).max()) if finite.any() else None
    metadata={'candidate_rays':count,'finite_candidate_rays':int(finite.sum()),
        'excluded_grazing_rays':int((finite&~conditioned).sum()),
        'incidence_cosine_threshold':minimum_incidence_cosine,
        'maximum_candidate_error_m':raw_error,'tolerance_m':tolerance_m,
        'selection':'Current-surface interior rays with incidence cosine>=.2; no tolerance relaxation'}
    if len(eligible)<samples:return {**metadata,'passed':False,'reason':'insufficient well-conditioned current-surface rays'}
    selected=eligible[np.linspace(0,len(eligible)-1,samples).astype(int)]
    error=float(abs(expected[selected]-depth[ys[selected],xs[selected]]).max())
    return {**metadata,'passed':error<=tolerance_m,'ray_count':samples,'finite_rays':samples,
        'maximum_error_m':error,'minimum_selected_incidence_cosine':float(cosine[selected].min()),
        'maximum_selected_amplification':float((1/cosine[selected]).max())}
