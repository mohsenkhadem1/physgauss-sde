"""Public oracle-input feature contract for the custom direct mechanics baseline.

Fingerprints evaluate the SAME frozen learned laws used by the mechanical model
on fixed public coupon probes. They are numeric inputs, not checkpoint IDs or
teacher/scene responses. This is an oracle-descriptor comparison; inferred-latent
experiments must replace descriptors consistently for both model families.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.scenes import material_field
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.interfaces import SharedInterfaceLaw

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def default_probes():
    """Declared coupon deformation gradients and bilateral separations, SI units."""
    bulk=[np.eye(3),np.diag([1.10,.95,1.]),np.diag([1.30,.90,.90]),np.eye(3)*1.05]
    for shear in (.1,.3,.6):
        F=np.eye(3);F[0,1]=shear;bulk.append(F)
    interface=np.array([[0,0,0],[.001,0,0],[.003,0,0],[.006,0,0],[0,.003,0],[0,0,.003]])
    return dict(bulk_F=np.asarray(bulk).tolist(),interface_delta_m=interface.tolist(),version='fixed-coupons-v1')

def validate_probes(probes):
    F=np.asarray(probes['bulk_F'],float);d=np.asarray(probes['interface_delta_m'],float)
    if F.ndim!=3 or F.shape[1:]!=(3,3)or not len(F)or not np.isfinite(F).all()or (np.linalg.det(F)<=0).any():raise ValueError('Finite positive-J coupon gradients required')
    if d.ndim!=2 or d.shape[1]!=3 or not len(d)or not np.isfinite(d).all():raise ValueError('Finite interface coupon displacements required')
    return F,d

def load_frozen_laws(bulk_path,interface_path,expected_sha256,device='cpu'):
    paths={'bulk':Path(bulk_path),'interface':Path(interface_path)}
    if any(sha(path)!=expected_sha256[key]for key,path in paths.items()):raise ValueError('Frozen law checkpoint changed')
    b=torch.load(paths['bulk'],map_location=device,weights_only=False);i=torch.load(paths['interface'],map_location=device,weights_only=False)
    bulk=ScaleSeparatedMaterialLaw().double().to(device);bulk.load_state_dict(b['state_dict'])
    interface=SharedInterfaceLaw(reference_length=i['reference_length'],bases=i['bases'],width=i['width']).double().to(device);interface.load_state_dict(i['state_dict'])
    for law in (bulk,interface):
        law.eval()
        for parameter in law.parameters():parameter.requires_grad_(False)
    return bulk,interface

def tensor_on_law(value,law):
    if isinstance(value,np.ndarray):value=np.ascontiguousarray(value)
    parameter=next(law.parameters());return torch.as_tensor(value,dtype=parameter.dtype,device=parameter.device)

def bulk_fingerprint(law,latents,F):
    z=tensor_on_law(latents,law);probe=tensor_on_law(F,law)
    x=probe[None].expand(len(z),-1,-1,-1).clone().requires_grad_(True)
    with torch.enable_grad():
        energy=law.energy_density(x,z[:,None]);stress=torch.autograd.grad(energy.sum(),x)[0]
    return torch.cat([energy[...,None],stress.reshape(len(z),len(probe),9)],-1).reshape(len(z),-1).detach().cpu().numpy()

def interface_fingerprint(law,latents,delta):
    z=tensor_on_law(latents,law);probe=tensor_on_law(delta,law)
    x=probe[None].expand(len(z),-1,-1).clone().requires_grad_(True)
    with torch.enable_grad():
        energy=law.energy_density(x,z[:,None]);traction=torch.autograd.grad(energy.sum(),x)[0]
    return torch.cat([energy[...,None],traction],-1).reshape(len(z),-1).detach().cpu().numpy()

def coordinates(value):
    q=np.asarray(value,float)
    if q.ndim!=2 or q.shape[1]!=3 or not len(q)or not np.isfinite(q).all()or (q<0).any()or (q>1).any():raise ValueError('Nonempty material coordinates in unit cube required')
    return q

def feature_contract(probes,checkpoint_sha256):
    F,d=validate_probes(probes)
    return dict(version='oracle-geometry-law-features-v1',physical_descriptor_fields=['bulk_latent','material_gradient','heterogeneity_pattern','interface_latents','centers','widths','geometry','density','gravity','prescribed_pose'],bulk_law_fingerprint=dict(checkpoint_sha256=checkpoint_sha256['bulk'],probes_F=F.tolist(),components='energy_density_Pa then row-major first_Piola_Pa for each probe'),interface_law_fingerprint=dict(checkpoint_sha256=checkpoint_sha256['interface'],probes_delta_m=d.tolist(),components='energy_J then traction_N xyz for each probe; normalized reference-area interface law'),node_columns='material_q[3],reference_xyz_m[3],local_bulk_descriptor[3],left/right_endpoint_flags[2],bulk_coupon_fingerprint[10*K]',global_columns='geometry(L,H,D_m,taper,arch_m)[5],prescribed_pose_m_rad[12],gravity_m_s2[3],density_kg_m3[1],bulk_descriptor[3],gradient[1],heterogeneity_pattern[1],interface_descriptor[6],centers[4],widths[4],interface_coupon_fingerprints[2*4*M]',query_columns='material_q[3],reference_xyz_m[3],local_bulk_descriptor[3]',scope='Same oracle descriptors and frozen law access as mechanical solver. Coupon outputs are deterministic input features, not additional scene-response labels. No scene/action IDs in network features; no normalization fitted here.')

def build_features(scene,prescribed_pose,node_material,query_material,bulk_law,interface_law,*,probes=None,checkpoint_sha256=None):
    probes=default_probes()if probes is None else probes;F,d=validate_probes(probes)
    nodes=coordinates(node_material);query=coordinates(query_material);pose=np.asarray(prescribed_pose,float)
    if pose.shape!=(2,6)or not np.isfinite(pose).all():raise ValueError('Explicit finite two-tool prescribed pose required')
    geometry=GeometrySpec(**scene['geometry']);field=material_field(scene)
    nz=field(tensor_on_law(nodes,bulk_law)).detach().cpu().numpy();qz=field(tensor_on_law(query,bulk_law)).detach().cpu().numpy()
    endpoints=np.stack([np.isclose(nodes[:,0],0,rtol=0,atol=1e-12),np.isclose(nodes[:,0],1,rtol=0,atol=1e-12)],-1).astype(float)
    nf=np.concatenate([nodes,geometry.map(nodes),nz,endpoints,bulk_fingerprint(bulk_law,nz,F)],-1)
    interface_z=np.asarray(scene['interface_latents'],float)
    gf=np.concatenate([np.r_[geometry.dimensions,geometry.taper,geometry.arch],pose.ravel(),np.asarray(scene['gravity']),[scene['density_kg_m3']],scene['bulk_latent'],[scene['gradient'],scene['factors']['heterogeneity']],interface_z.ravel(),np.asarray(scene['centers']).ravel(),np.asarray(scene['widths']).ravel(),interface_fingerprint(interface_law,interface_z,d).ravel()])
    qf=np.concatenate([query,geometry.map(query),qz],-1)
    if not all(np.isfinite(x).all()for x in (nf,gf,qf)):raise ValueError('Nonfinite public features')
    return dict(nodefeatures=nf,globalfeatures=gf,queryfeatures=qf,reference_query_positions_m=geometry.map(query),material_query=query.copy(),prescribed_pose=pose.copy(),feature_contract=feature_contract(probes,checkpoint_sha256 or {'bulk':'uncommitted-development','interface':'uncommitted-development'}))

def load_labelled_record(scene,metadata,label_path,*,expected_sha256,node_material,bulk_law,interface_law,probes,checkpoint_sha256,allowed_scenes_by_split):
    """Read one committed training/validation record; never fit normalization.

metadata={scene_id,split,prescribed_pose}; NPZ keys exactly material_query,
displacement[Q,3],wrenches[2,6]. Split roles are declared externally. Actual
reserved/test labels intentionally unsupported by this training-loader API.
"""
    role=metadata['split'];scene_id=metadata['scene_id']
    if role not in ('train','validation')or scene_id not in allowed_scenes_by_split.get(role,[]):raise ValueError('Record outside declared train/validation scene split')
    if set(allowed_scenes_by_split.get('train',[]))&set(allowed_scenes_by_split.get('validation',[])):raise ValueError('Scene split overlap')
    if scene_id!=scene['name']:raise ValueError('Scene metadata/geometry identity mismatch')
    path=Path(label_path)
    if sha(path)!=expected_sha256:raise ValueError('Label artifact changed')
    with np.load(path,allow_pickle=False)as data:
        if set(data.files)!={'material_query','displacement','wrenches'}:raise ValueError('Unexpected supervised label fields')
        q=coordinates(data['material_query']);u=data['displacement'].copy();w=data['wrenches'].copy()
    if u.shape!=(len(q),3)or w.shape!=(2,6)or not np.isfinite(u).all()or not np.isfinite(w).all():raise ValueError('Finite matching supervised labels required')
    features=build_features(scene,metadata['prescribed_pose'],node_material,q,bulk_law,interface_law,probes=probes,checkpoint_sha256=checkpoint_sha256)
    return dict(**features,displacement=u,wrenches=w,scene_id=scene_id,split=role,label_sha256=expected_sha256)

def batch_records(records,dtype=torch.float32):
    """Collate equal-size input clouds/query sets; arbitrary Q supported perbatch."""
    if not records:raise ValueError('Empty records')
    role=records[0]['split'];contract=records[0]['feature_contract']
    if any(r['split']!=role or r['feature_contract']!=contract for r in records):raise ValueError('Mixed roles or feature contracts')
    pack=dict(split=role,scene_ids=[r['scene_id']for r in records])
    for key in ('nodefeatures','globalfeatures','queryfeatures','displacement','wrenches'):
        try:pack[key]=torch.as_tensor(np.stack([r[key]for r in records]),dtype=dtype)
        except ValueError as error:raise ValueError('Bucket records by node/query count before collation')from error
    return pack,contract
