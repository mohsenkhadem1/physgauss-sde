"""Material-coordinate Gaussian boundary features and explicit soft-support law.

The optional compliant/distributed models are DIFFERENT boundary hypotheses.
Exact reference node constraints and rigid-handle forces are never replaced.
No simulator, observations, recipe, contact or learned architecture is loaded.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

S=2
H=2
ROLES=('known_hardware','candidate_boundary_hypothesis','candidate_force_hypothesis')


def finite(value,shape,name):
    a=np.array(value,dtype=float,copy=True)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite '+name+' shaped'+str(shape)+' required')
    return a


def psd(value,name):
    a=finite(value,(3,3),name);scale=max(float(np.abs(a).max()),np.finfo(float).tiny)
    if np.abs(a-a.T).max()>64*np.finfo(float).eps*scale:raise ValueError(name+' must be symmetric PSD')
    # A rank-deficient Gram matrix can have tiny negative eigensolver roundoff.
    # Accept only that scale-relative window; do NOT project/jitter/change K/D.
    if np.linalg.eigvalsh(a)[0]<-64*np.finfo(float).eps*scale:raise ValueError(name+' must be PSD, no stiffness repair')
    return a


@dataclass(frozen=True)
class SurfaceQuadrature:
    points_material_m:np.ndarray
    node_area_m2:np.ndarray

    @classmethod
    def from_mesh(cls,points_material_m,surface_triangles):
        points=np.asarray(points_material_m,dtype=float);faces=np.asarray(surface_triangles)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Finite rest-material points[N,3] required')
        if faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu' or not len(faces):raise ValueError('Integer surface triangles[F,3] required')
        if faces.min()<0 or faces.max()>=len(points) or any(len(set(f))!=3 for f in faces):raise ValueError('Valid distinct surface indices required')
        if len(np.unique(np.sort(faces,axis=1),axis=0))!=len(faces):raise ValueError('Duplicate surface triangle')
        tri=points[faces];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        if not np.isfinite(areas).all() or np.any(areas<=0):raise ValueError('Nondegenerate finite surface triangles required')
        weights=np.zeros(len(points))
        for column in range(3):np.add.at(weights,faces[:,column],areas/3)
        return cls(points.copy(),weights)

    def __post_init__(self):
        p=np.asarray(self.points_material_m,dtype=float)
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('Finite material points required')
        w=finite(self.node_area_m2,(len(p),),'surface areas')
        if np.any(w<0) or not w.sum()>0:raise ValueError('Positive total surface area required')
        object.__setattr__(self,'points_material_m',p.copy());object.__setattr__(self,'node_area_m2',w)


@dataclass(frozen=True)
class GaussianPatch:
    center_material_m:np.ndarray
    covariance_material_m2:np.ndarray
    information_role:str
    active:bool=True

    def __post_init__(self):
        c=finite(self.center_material_m,(3,),'material Gaussian center');cov=psd(self.covariance_material_m2,'material covariance')
        if np.linalg.eigvalsh(cov)[0]<=0:raise ValueError('Strictly positive covariance bandwidth required')
        if self.information_role not in ROLES or type(self.active) is not bool:raise ValueError('Explicit Gaussian information role and boolean activity required')
        object.__setattr__(self,'center_material_m',c);object.__setattr__(self,'covariance_material_m2',cov)


def _normalized_patch_weights(quadrature,patch):
    """Area-weighted Gaussian, normalized to1; interior/nonactive nodes are0.

    Material weights attach to X, never recomputed from deforming world x.
    Vertex-lumped quadrature integrates affine fields exactly on each triangle.
    Gaussian integrals are discrete quadrature approximations, not analytic
    continuous-surface integrals or a mesh-convergence certificate.
    """
    if not isinstance(quadrature,SurfaceQuadrature) or not isinstance(patch,GaussianPatch):raise TypeError('Explicit quadrature and GaussianPatch required')
    out=np.zeros(len(quadrature.points_material_m))
    if not patch.active:return out
    ids=np.flatnonzero(quadrature.node_area_m2>0);r=quadrature.points_material_m[ids]-patch.center_material_m
    whitened=np.linalg.solve(np.linalg.cholesky(patch.covariance_material_m2),r.T).T
    log=np.log(quadrature.node_area_m2[ids])-.5*np.einsum('ni,ni->n',whitened,whitened)
    if not np.isfinite(log).all():raise ValueError('Gaussian bandwidth/distance overflows; no silent nearest-node projection')
    values=np.exp(log-log.max());out[ids]=values/values.sum()
    return out


def normalized_surface_weights(reference_nodes,surfacefaces,centers,covariance,active):
    """Graph API: return (weights[J,N], lumped_surface_area_m2[N]).

    All coordinates/covariances are in the same supplied material frame.
    This pure quadrature helper supplies no information-role authorization.
    """
    quadrature=SurfaceQuadrature.from_mesh(reference_nodes,surfacefaces)
    centers=np.asarray(centers,dtype=float);covariance=np.asarray(covariance,dtype=float);active=np.asarray(active)
    if centers.ndim!=2 or centers.shape[1]!=3 or covariance.shape!=(len(centers),3,3) or active.shape!=(len(centers),) or active.dtype!=bool:raise ValueError('Explicit centers[J,3], covariance[J,3,3], boolean active[J] required')
    weights=np.stack([_normalized_patch_weights(quadrature,GaussianPatch(c,cov,'known_hardware',bool(on)))
        for c,cov,on in zip(centers,covariance,active)]) if len(centers) else np.zeros((0,len(reference_nodes)))
    return weights,quadrature.node_area_m2.copy()


def patch_bank(quadrature,patches,capacity,allowed_roles):
    if len(patches)!=capacity or any(p.information_role not in allowed_roles for p in patches):raise ValueError('Fixed-capacity patches with correct information roles required')
    return {'centers_material_m':np.array([p.center_material_m for p in patches]),
        'covariances_material_m2':np.array([p.covariance_material_m2 for p in patches]),
        'active_mask':np.array([p.active for p in patches]),
        'normalized_node_weights':np.stack([_normalized_patch_weights(quadrature,p) for p in patches])}


def feature_groups(quadrature,*,support_patches,load_patches):
    """Complete conditioned features. Only known-hardware load groups are public.

    Support activity/covariance/centers always remain latent. Unknown force
    kernels are a distinct latent hypothesis, never public known commands.
    Known actuator force values belong to the existing explicit query context.
    """
    supports=patch_bank(quadrature,support_patches,S,{'candidate_boundary_hypothesis'})
    loads=patch_bank(quadrature,load_patches,H,{'known_hardware','candidate_force_hypothesis'})
    roles={p.information_role for p in load_patches}
    if len(roles)!=1:raise ValueError('One load-bank role required; mixed roles need separate fixed banks')
    load_key='public_known_load_gaussians' if roles=={'known_hardware'} else 'latent_force_gaussians'
    return {'feature_groups':{'latent_support_gaussians':supports,load_key:loads},
        'schema':{'version':'material-boundary-gaussians-v0.1','support_capacity':S,'handle_capacity':H,
            'material_coordinates':'X in m; covariance in m^2; weights dimensionless normalized rest-surface area',
            'role':'features or explicit compliant/distributed hypothesis; NOT exact reference boundary replacement'}}


def public_features(conditioned):
    return {'feature_groups':{key:{name:array.copy() for name,array in group.items()}
        for key,group in conditioned['feature_groups'].items() if key.startswith('public_')},'schema':dict(conditioned['schema'])}


def compliant_support_response(quadrature,patches,*,positions_world_m,velocities_world_m_s,
        target_positions_world_m,target_velocities_world_m_s,stiffness_world_N_m,damping_world_Ns_m):
    """Explicit SOFT support: total K[N/m], D[N s/m], normalized patch weights.

    K/D are world-frame symmetric PSD3x3 tensors, not penalties for hard masks.
    Targets are explicitly supplied[S,N,3]; no motion/interpolation is invented.
    """
    bank=patch_bank(quadrature,patches,S,{'candidate_boundary_hypothesis'});w=bank['normalized_node_weights'];N=w.shape[1]
    x=finite(positions_world_m,(N,3),'world positions');v=finite(velocities_world_m_s,(N,3),'world velocities')
    y=finite(target_positions_world_m,(S,N,3),'support targets');vy=finite(target_velocities_world_m_s,(S,N,3),'support target velocities')
    K=np.stack([psd(a,'world stiffness') for a in np.asarray(stiffness_world_N_m)])
    D=np.stack([psd(a,'world damping') for a in np.asarray(damping_world_Ns_m)])
    if K.shape!=(S,3,3) or D.shape!=(S,3,3):raise ValueError('Exactly two K/D tensors required')
    e=x[None]-y;relative=v[None]-vy;Ke=np.einsum('sij,snj->sni',K,e);Dv=np.einsum('sij,snj->sni',D,relative)
    elastic=-w[:,:,None]*Ke;viscous=-w[:,:,None]*Dv;force=elastic+viscous
    return {'kind':'explicit_compliant_fixture_hypothesis_NOT_hard_constraint',
        'elastic_energy_J':float(.5*np.einsum('sn,snc,snc->',w,e,Ke)),
        'elastic_forces_each_world_N':elastic,'damping_forces_each_world_N':viscous,'forces_each_world_N':force,
        'forces_world_N':force.sum(0),'dissipation_rate_W':float(np.einsum('sn,snc,snc->',w,relative,Dv)),
        'target_actuator_power_W':float(np.sum(force*vy)),'normalized_surface_weights':w}


def distributed_load_response(quadrature,patches,*,total_forces_world_N,positions_world_m,handle_origins_world_m):
    """Distributed-load hypothesis/feature with exact mathematical resultant.

    A float64 roundoff correction is reported. Torque is generally DIFFERENT
    from force at the actual rigid handle origin: do not substitute this field
    for an exact-hardware reference. Surface area is already in weights.
    """
    bank=patch_bank(quadrature,patches,H,{'known_hardware','candidate_force_hypothesis'});w=bank['normalized_node_weights'];N=w.shape[1]
    F=finite(total_forces_world_N,(H,3),'total forces');x=finite(positions_world_m,(N,3),'load positions');origins=finite(handle_origins_world_m,(H,3),'actual handle origins')
    for j,p in enumerate(patches):
        if not p.active and np.any(F[j]):raise ValueError('Inactive load patch cannot carry force')
    nodal=w[:,:,None]*F[:,None];correction=np.zeros((H,3))
    for j,p in enumerate(patches):
        if p.active:
            correction[j]=F[j]-nodal[j].sum(0);nodal[j,np.argmax(w[j])]+=correction[j]
    actual=nodal.sum(1);torque=np.cross(x[None],nodal).sum(1);reference=np.cross(origins,F)
    return {'kind':'distributed_load_hypothesis_NOT_rigid_handle_reference','forces_each_world_N':nodal,'forces_world_N':nodal.sum(0),
        'requested_total_forces_world_N':F,'integrated_total_forces_world_N':actual,'resultant_roundoff_error_N':actual-F,
        'roundoff_correction_N':correction,'moments_about_world_origin_Nm':torque,
        'rigid_handle_origin_moments_Nm':reference,'moment_difference_from_rigid_origin_Nm':torque-reference,
        'application_centroids_world_m':w@x,'normalized_surface_weights':w}


def exact_reference_boundary(*,hard_node_axis_masks,hard_targets_world_m,ordered_handle_patch_nodes,handle_origin_node_ids):
    """Transparent exact reference decode: no Gaussian/K/D approximation."""
    masks=np.asarray(hard_node_axis_masks)
    if masks.ndim!=3 or masks.shape[0]!=S or masks.shape[2]!=3 or masks.dtype!=bool:raise ValueError('Exact boolean hard masks[2,N,3] required')
    if np.any(masks.sum(0)>1):raise ValueError('Conflicting exact fixture masks')
    targets=finite(hard_targets_world_m,masks.shape,'exact hard targets');N=masks.shape[1]
    origins=np.asarray(handle_origin_node_ids);patches=[np.asarray(p) for p in ordered_handle_patch_nodes]
    if origins.shape!=(H,) or origins.dtype.kind not in 'iu' or len(patches)!=H:raise ValueError('Exactly two ordered rigid patches/origins required')
    used=set()
    for origin,patch in zip(origins,patches):
        if patch.ndim!=1 or patch.dtype.kind not in 'iu' or not len(patch) or len(set(patch))!=len(patch) or patch.min()<0 or patch.max()>=N or origin not in patch:raise ValueError('Full exact ordered patch/origin required')
        if used.intersection(patch.tolist()) or np.any(masks[:,patch]):raise ValueError('Reference hard fixture/rigid patches must not overlap')
        used.update(patch.tolist())
    return {'fixture_kind':'exact_hard_node_axis_constraints','hard_node_axis_masks':masks.copy(),'hard_targets_world_m':targets,
        'load_kind':'exact_rigid_handle_origin_force','ordered_handle_patch_nodes':[p.copy() for p in patches],
        'handle_origin_node_ids':origins.copy(),'Gaussian_substitution_performed':False}


def world_contact_response(*args,**kwargs):
    raise NotImplementedError('World contact is unsupported: an explicit gap/normal/friction/penetration law and state are required; no material Gaussian pretends to be contact.')
