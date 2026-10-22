"""C38 study controls: no datasets/simulations loaded and no model trained.

Opaque P1/P2 geometry is supported. This module enforces declared metadata,
not the physics: independent numerical admission evidence remains mandatory.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROLES=('TRAIN','DEVELOPMENT','CALIBRATION','RESERVED_EVALUATION')
FACTORS=('material_sha256','support_layout_sha256','force_history_sha256')

def array_digest(value,dtype):
    a=np.ascontiguousarray(value,dtype=np.dtype(dtype).newbyteorder('<'))
    if a.dtype.kind=='f' and not np.isfinite(a).all():raise ValueError('Finite canonical physical factor required')
    h=hashlib.sha256();h.update(json.dumps({'shape':list(a.shape),'dtype':a.dtype.str},sort_keys=True).encode());h.update(a.tobytes())
    return h.hexdigest()

def factors(material,support_mask,force_times,force_coefficients):
    """Hash entire literal known history, not an ID, split name or private Hz."""
    m=np.asarray(material,dtype=float);s=np.asarray(support_mask);t=np.asarray(force_times,dtype=float);f=np.asarray(force_coefficients,dtype=float)
    if m.shape!=(4,) or s.ndim!=1 or s.dtype!=np.bool_ or t.ndim!=1 or len(t)<2 or f.ndim!=2 or f.shape[0]!=len(t)-1 or np.any(np.diff(t)<=0):raise ValueError('E/rho/nu/drag, exact mask, interval-boundaries/coefficients required')
    if not np.isfinite(m).all() or m[0]<=0 or m[1]<=0 or not 0<m[2]<.5 or m[3]<0:raise ValueError('Physical homogeneous material tuple required')
    force_hash=hashlib.sha256((array_digest(t,'f8')+array_digest(f,'f8')).encode()).hexdigest()
    return dict(material_sha256=array_digest(m,'f8'),support_layout_sha256=array_digest(s,'u1'),force_history_sha256=force_hash)

def validate_catalog(rows,*,expected_counts=None):
    """Keep every object/counterfactual in one role; whole factor values disjoint."""
    if not rows:raise ValueError('Nonempty prospectively assigned catalog required')
    ids=set();objects={};factor_roles={key:{} for key in FACTORS};counts={r:0 for r in ROLES}
    for row in rows:
        role=row.get('role');identifier=row.get('episode_id');obj=row.get('object_id')
        if role not in ROLES or not isinstance(identifier,str) or not identifier or identifier in ids or not isinstance(obj,str) or not obj:raise ValueError('Unique episode/object IDs and declared role required')
        ids.add(identifier);counts[role]+=1
        if obj in objects and objects[obj]!=role:raise ValueError('Object or counterfactual leaked across roles')
        objects[obj]=role
        for key in FACTORS:
            value=row.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Canonical SHA256 factor fingerprints required')
            previous=factor_roles[key].get(value)
            if previous is not None and previous!=role:raise ValueError('Entire '+key+' reused across roles')
            factor_roles[key][value]=role
    if expected_counts is not None and counts!=expected_counts:raise ValueError('Exact prospective role counts differ; no case filtering')
    return dict(role_counts=counts,objects=len(objects),episodes=len(rows),joint_factor_OOD=True)

def verify_admission(manifest,*,root):
    """Root-generated admission binds independent solver/mesh reports, no fake truth.

    This is a metadata barrier. It does not replace numerical review or infer
    mechanics validity from a learned score or a successful process exit.
    """
    root=Path(root).resolve();r=manifest['forward_admission'];path=(root/r['path']).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Changed forward admission evidence')
    gate=json.loads(path.read_text())
    for name in ('qualified_for_prediction','spatial_qualified','temporal_qualified','nonlinear_solve_qualified','constitutive_matched'):
        if gate.get(name) is not True:raise ValueError('Forward truth remains unqualified: '+name)
    if gate.get('domain_sha256')!=manifest.get('domain_sha256') or not gate.get('evidence'):raise ValueError('Exact qualified domain and primary evidence required')
    for record in gate['evidence']:
        p=(root/record['path']).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Changed primary forward evidence')
    for name,digest in gate.get('source_sha256',{}).items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Changed admitted physical source')
    if not gate.get('source_sha256'):raise ValueError('Physical source pins required')
    return gate

def causal_prefix(times,positions,cutoff):
    """Select acquired past BEFORE checking values; future NaNs cannot influence it.

    Real inference files contain only this extracted prefix. This helper also
    makes future-value invariance testable without exposing targets to models.
    """
    t=np.asarray(times,dtype=float);q=np.asarray(positions)
    if t.ndim!=1 or q.shape[0]!=len(t) or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or not np.isfinite(cutoff):raise ValueError('Ordered known observation clock required')
    ids=np.flatnonzero(t<=cutoff+1e-12)
    if not len(ids):raise ValueError('At least one acquired past observation required')
    past=np.array(q[ids],dtype=float,copy=True)
    if past.ndim!=3 or past.shape[-1]!=3 or not np.isfinite(past).all():raise ValueError('Finite past positions[K,N,3] required')
    return t[ids].copy(),past

def rgbd_public_prefix(*,rest_nodes,elements,surface_triangles,observation_times,rgb,depth_m,visible,camera_poses,intrinsics,cutoff,force_times,force_coefficients,load_basis):
    """Actual sparse-view B inputs; no material, support or full-state arguments.

    Slice acquired images BEFORE inspecting their values. This is a payload
    constructor, not an image encoder, tracker or sparse-view reconstruction.
    Full planned controls are public; future RGBD never enters its output.
    """
    t=np.asarray(observation_times,dtype=float);d=np.asarray(depth_m);v=np.asarray(visible);color=np.asarray(rgb)
    poses=np.asarray(camera_poses);k=np.asarray(intrinsics)
    if t.ndim!=1 or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or d.ndim!=4 or d.shape[0]!=len(t) or v.shape!=d.shape or v.dtype!=np.bool_ or color.shape!=(*d.shape,3) or poses.shape!=(*d.shape[:2],4,4) or k.shape!=(*d.shape[:2],3,3) or not np.isfinite(cutoff):raise ValueError('Clocked RGBD, visibility and known calibrated cameras required')
    ids=np.flatnonzero(t<=cutoff+1e-12)
    if not len(ids):raise ValueError('Acquired RGBD prefix required')
    d=d[ids].astype(float,copy=True);v=v[ids].copy();color=color[ids].copy();poses=poses[ids].copy();k=k[ids].copy()
    if not all(np.isfinite(x).all() for x in (d,color,poses,k)) or (d[v]<=0).any():raise ValueError('Finite acquired images/cameras and positive visible depth required')
    X=np.asarray(rest_nodes,dtype=float);e=np.asarray(elements);f=np.asarray(surface_triangles);ft=np.asarray(force_times,dtype=float);fc=np.asarray(force_coefficients,dtype=float);basis=np.asarray(load_basis,dtype=float)
    if X.ndim!=2 or X.shape[1]!=3 or e.ndim!=2 or e.shape[1]!=10 or f.ndim!=2 or f.shape[1]!=3 or e.dtype.kind not in 'iu' or f.dtype.kind not in 'iu' or np.any(e<0) or np.any(e>=len(X)) or np.any(f<0) or np.any(f>=len(X)):raise ValueError('Public reference P2 geometry required')
    if ft.ndim!=1 or len(ft)<2 or np.any(np.diff(ft)<=0) or fc.ndim!=2 or len(fc)!=len(ft)-1 or basis.shape!=(fc.shape[1],len(X),3) or not all(np.isfinite(x).all() for x in (X,ft,fc,basis)):raise ValueError('Literal known force schedule and physical reference basis required')
    return dict(information_role='public_sparse_RGBD_prefix_B',rest_nodes=X.copy(),elements=e.copy(),surface_triangles=f.copy(),observation_times=t[ids].copy(),rgb=color,depth_m=d,visible=v,camera_poses=poses,intrinsics=k,force_times=ft.copy(),force_coefficients=fc.copy(),load_basis=basis.copy(),cutoff_s=float(cutoff))

def future_times(times,last_observation):
    t=np.asarray(times,dtype=float)
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or np.any(t<=last_observation):raise ValueError('Strictly future forecast targets required; no gap filling')
    return t

def persistence(past_times,past_positions,targets):
    t,q=causal_prefix(past_times,past_positions,np.max(past_times));future_times(targets,t[-1])
    return np.repeat(q[-1,None],len(targets),axis=0)

def constant_velocity(past_times,past_positions,targets):
    t,q=causal_prefix(past_times,past_positions,np.max(past_times));s=future_times(targets,t[-1])
    if len(t)<2:raise ValueError('Two acquired frames required for velocity extrapolation')
    v=(q[-1]-q[-2])/(t[-1]-t[-2])
    return q[-1]+(s-t[-1])[:,None,None]*v

def rigid_extrapolation(rest,past_times,past_positions,targets,weights,*,fitting_rest=None,fitting_past_positions=None):
    """Constant angular/COM velocity of last observed SHAPE, retaining deformation.

    P2 callers supply positive material-quadrature samples for fitting; signed
    P2 nodal integration coefficients are NOT a positive mass sampling measure.
    The fitted motion then transports every last-observed output node.
    """
    t,q=causal_prefix(past_times,past_positions,np.max(past_times));s=future_times(targets,t[-1]);rest=np.asarray(rest,dtype=float)
    X=rest if fitting_rest is None else np.asarray(fitting_rest,dtype=float)
    fit=q if fitting_past_positions is None else np.asarray(fitting_past_positions,dtype=float);w=np.asarray(weights,dtype=float)
    if len(t)<2 or rest.shape!=q.shape[1:] or X.ndim!=2 or X.shape[1]!=3 or fit.shape!=(len(t),*X.shape) or not np.isfinite(X).all() or not np.isfinite(fit).all() or w.shape!=(len(X),) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Past pair and explicit positive material-mass measure required')
    w=w/w.sum();xc=w@X;poses=[];centers=[]
    for p in fit[-2:]:
        c=w@p;u,_,vh=np.linalg.svd(((X-xc)*w[:,None]).T@(p-c));R=vh.T@u.T
        if np.linalg.det(R)<0:vh[-1]*=-1;R=vh.T@u.T
        poses.append(R);centers.append(c)
    dt=t[-1]-t[-2];omega=Rotation.from_matrix(poses[1]@poses[0].T).as_rotvec()/dt
    rot=Rotation.from_rotvec((s-t[-1])[:,None]*omega).as_matrix();v=(centers[1]-centers[0])/dt
    return np.einsum('tij,nj->tni',rot,q[-1]-centers[1])+centers[1]+(s-t[-1])[:,None,None]*v

def forecast_metrics(prediction,truth,times,cutoff):
    t=future_times(times,cutoff);p=np.asarray(prediction,dtype=float);q=np.asarray(truth,dtype=float)
    if p.shape!=q.shape or p.ndim!=3 or p.shape[0]!=len(t) or p.shape[-1]!=3 or not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('Complete finite matched future trajectories required')
    delta=p-q
    return dict(coordinate_MSE_m2=float(np.mean(delta**2)),coordinate_RMSE_m=float(np.sqrt(np.mean(delta**2))),maximum_material_point_error_m=float(np.linalg.norm(delta,axis=-1).max()),forecast_frames=len(t),cutoff_s=float(cutoff),observations_after_cutoff_used=False)

def paired_object_interval(first,second,*,seed=2026122200,replicates=4096):
    """Descriptive paired OBJECT bootstrap, never treat frames as iid objects."""
    a,b=np.asarray(first,dtype=float),np.asarray(second,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Matched complete object-level scores required')
    d=a-b;ids=np.random.default_rng(seed).integers(len(d),size=(replicates,len(d)));means=d[ids].mean(1)
    return dict(mean_difference=float(d.mean()),percentile_95percent_interval=np.quantile(means,[.025,.975]).tolist(),objects=len(d),seed=seed,replicates=replicates,unit='same as supplied object scores',not_a_numerical_or_physical_calibration_guarantee=True)
