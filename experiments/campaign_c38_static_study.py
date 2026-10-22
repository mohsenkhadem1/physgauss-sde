"""Static-only evidence/role boundary. No admission, observations or roles created.

Acquisition indices are ledger entries, never physical seconds. The existing
trajectory admission is not changed. Private labels are metadata/targets only;
public RGB-D inputs reuse the original exact allowlists. This is an API barrier,
not an OS sandbox or an independent proof of physical qualification.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np
from experiments.campaign_c38_data import _checked,_load,_geometry,load_public_geometry
from experiments.campaign_c38_static_data import load_public_static_controls,load_public_static_rgbd_prefix

SOURCE='experiments/campaign_c38_static_study.py'
ROLES=('TRAIN','DEVELOPMENT','CALIBRATION','RESERVED_EVALUATION')
SUPPORT_CATEGORIES=('both_ends','separated_bottom_patches')
TASK='static_equilibrium_load_prediction'
STATIC_FLAGS=('qualified_for_static_prediction','all_declared_qualification_cases_complete','spatial_qualified','nonlinear_solve_qualified','force_state_qualified','constitutive_matched','whole_element_J_qualified','root_reviewed_static_admission')
EVIDENCE_KINDS={'spatial_mesh','nonlinear_equilibrium','force_compiler','constitutive','whole_element_J'}


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _record(root,record):
    if not isinstance(record,dict) or not isinstance(record.get('path'),str) or not _digest(record.get('sha256')):raise ValueError('Canonical hash-bound record required')
    path=Path(record['path'])
    if path.is_absolute() or '..' in path.parts or path.as_posix()!=record['path']:raise ValueError('Canonical repository-relative path required')
    return _checked(root,record)

def _digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def _indices(values):
    a=np.asarray(values)
    if a.ndim!=1 or not len(a) or a.dtype.kind not in 'iu' or np.any(a<0) or len(np.unique(a))!=len(a):raise ValueError('Unique nonnegative acquisition ledger indices required')
    return a


def verify_static_admission(manifest,*,root):
    """Only an explicit reviewed static gate can admit exact static fields.

    Temporal/contact flags do not become true by using this interface. A finite
    14-point screen alone cannot manufacture reviewed continuous-domain status.
    """
    if manifest.get('schema')!='C38-static-study-v01' or manifest.get('task_kind')!=TASK or not _digest(manifest.get('domain_sha256')):raise ValueError('Explicit static manifest/domain required; no invented time')
    path=_record(root,manifest['static_admission']);gate=json.loads(path.read_text())
    if gate.get('schema')!='C38-reviewed-static-admission-v01' or gate.get('task_kind')!=TASK or gate.get('domain_sha256')!=manifest['domain_sha256']:raise ValueError('Exact reviewed static domain required')
    for name in STATIC_FLAGS:
        if gate.get(name) is not True:raise ValueError('Static forward evidence remains unqualified: '+name)
    if gate.get('finite_screen_is_uniform_continuous_domain_certificate') is not False:raise ValueError('Finite screen cannot claim a uniform continuous-domain certificate')
    evidence=gate.get('evidence')
    if not isinstance(evidence,list) or not EVIDENCE_KINDS.issubset({r.get('kind') for r in evidence if isinstance(r,dict)}):raise ValueError('Independent static mesh/nonlinear/force/law/J evidence required')
    for r in evidence:_record(root,r)
    if not isinstance(gate.get('review'),dict):raise ValueError('Explicit hash-bound static review required')
    review=json.loads(_record(root,gate['review']).read_text())
    if review.get('schema')!='C38-static-domain-review-v01' or review.get('approved_static_prediction') is not True or review.get('domain_sha256')!=gate['domain_sha256'] or review.get('reviewed_evidence')!=evidence:raise ValueError('Explicit hash-bound static review required; finite screen alone is insufficient')
    sources=gate.get('source_sha256')
    if not isinstance(sources,dict) or SOURCE not in sources:raise ValueError('Physical and static-admission source pins required')
    physical=gate.get('physical_source_sha256')
    if not isinstance(physical,dict) or not physical or SOURCE in physical:raise ValueError('Independent physical source pins required')
    if any(k in sources and sources[k]!=v for k,v in physical.items()):raise ValueError('Conflicting physical source pins')
    for name,digest in {**sources,**physical}.items():
        if not _digest(digest):raise ValueError('Canonical source SHA required')
        _record(root,{'path':name,'sha256':digest})
    if sources[SOURCE]!=_sha(__file__):raise ValueError('Running static admission source differs')
    fields=gate.get('admitted_static_fields')
    if not isinstance(fields,list) or not fields:raise ValueError('Explicit reviewed static field records required')
    seen=set()
    for r in fields:
        if not isinstance(r,dict):raise ValueError('Static field record required')
        field_path=r.get('path')
        if not isinstance(field_path,str) or Path(field_path).is_absolute() or '..' in Path(field_path).parts or Path(field_path).as_posix()!=field_path:raise ValueError('Canonical admitted static field path required')
        key=(field_path,r.get('sha256'))
        if not isinstance(r,dict) or not _digest(r.get('sha256')) or key in seen or r.get('domain_sha256')!=gate['domain_sha256'] or not isinstance(r.get('object_id'),str) or not r['object_id'] or type(r.get('load_state_index')) is not int or r['load_state_index']<0:raise ValueError('Unique exact object/load/domain static bindings required')
        seen.add(key)
    return gate


@dataclass(frozen=True)
class QualifiedStaticField:
    geometry:dict
    positions_m:np.ndarray
    constraint_mask:np.ndarray
    object_id:str
    load_state_index:int
    information_role:str='PRIVATE_STATIC_TARGET_OR_SUPPLIED_PHYSICS_ORACLE_A'


def load_admitted_static_targets(root,manifest,record):
    """Gate metadata and exact field binding BEFORE opening any private array."""
    gate=verify_static_admission(manifest,root=root)
    matches=[r for r in gate['admitted_static_fields'] if (r['path'],r['sha256'])==(record.get('path'),record.get('sha256'))]
    if len(matches)!=1:raise ValueError('Static target is not bound to the reviewed domain')
    bound=matches[0]
    for key in ('object_id','load_state_index','domain_sha256'):
        if key not in record or record[key]!=bound[key]:raise ValueError('Static target object/load/domain mismatch')
    if ('case_id' in record or 'case_id' in bound) and record.get('case_id')!=bound.get('case_id'):raise ValueError('Static target case mismatch')
    _record(root,record)
    d=_load(root,record,('rest_m','elements','tets','faces','positions_m','constraint_mask'))
    geometry=_geometry({k:d[k] for k in ('rest_m','elements','tets','faces')});q=d['positions_m'];mask=d['constraint_mask']
    if q.shape!=geometry['rest_m'].shape or not np.isfinite(q).all() or mask.shape!=q.shape or mask.dtype!=np.bool_:raise ValueError('One equilibrium field[N,3] and exact constrained DOFs required; no clock/velocity')
    q.flags.writeable=False;mask.flags.writeable=False
    return QualifiedStaticField(geometry,q,mask,record['object_id'],record['load_state_index'])


def public_static_rgbd_inputs(root,*,geometry_record,controls_record,observation_record,last_acquisition):
    """No private labels/full fields accepted. Planned force states stay public."""
    for r in (geometry_record,controls_record,observation_record):_record(root,r)
    geometry=load_public_geometry(root,geometry_record)
    controls=load_public_static_controls(root,controls_record,number_of_nodes=len(geometry['rest_m']))
    observations=load_public_static_rgbd_prefix(root,observation_record,last_acquisition=last_acquisition,known_load_state_indices=controls['load_state_indices'])
    return dict(information_role='PUBLIC_ACQUIRED_STATIC_RGBD_B',geometry=geometry,known_controls=controls,observations=observations,observation_kind='rgbd',is_temporal_forecast=False)


def load_public_ideal_static_points(root,record,*,last_acquisition,known_load_state_indices):
    """Declared ideal material correspondences; explicitly NOT RGB-D perception."""
    if type(last_acquisition) is not int or last_acquisition<0:raise ValueError('Nonnegative acquired ledger cutoff required')
    keys=('acquisition_indices','load_state_indices','material_points_m','observed_positions_m','sensor_covariance_m2')
    with np.load(_record(root,record),allow_pickle=False) as z:
        if set(z.files)!=set(keys):raise ValueError('Exact ideal-point public allowlist required')
        acquisition=_indices(z['acquisition_indices'])
        if np.any(np.diff(acquisition)<=0) or np.any(acquisition>last_acquisition):raise ValueError('Only acquired ideal-point observations allowed')
        states=np.asarray(z['load_state_indices']);known=_indices(known_load_state_indices)
        if states.shape!=acquisition.shape or states.dtype.kind not in 'iu' or not np.isin(states,known).all():raise ValueError('Known acquired static loads required')
        d={k:z[k].copy() for k in keys}
    X=d['material_points_m'];y=d['observed_positions_m'];C=d['sensor_covariance_m2']
    if X.ndim!=2 or X.shape[1]!=3 or not len(X) or y.shape!=(len(acquisition),len(X),3) or C.shape!=(3*len(X),3*len(X)) or not all(np.isfinite(a).all() for a in (X,y,C)) or not np.allclose(C,C.T,rtol=1e-12,atol=1e-18):raise ValueError('Finite ideal tracked points and declared sensor covariance required')
    try:np.linalg.cholesky(C)
    except np.linalg.LinAlgError as error:raise ValueError('Positive definite ideal-point sensor covariance required') from error
    return dict(information_role='PUBLIC_IDEAL_TRACKED_MATERIAL_POINTS_STATIC_B',observation_kind='ideal_tracked_points',is_RGBD_perception=False,is_temporal_forecast=False,**d)


def material_fingerprint(young_Pa):
    if isinstance(young_Pa,bool) or not isinstance(young_Pa,(int,float,np.integer,np.floating)) or not np.isfinite(young_Pa) or young_Pa<=0:raise ValueError('Positive continuous material metadata required')
    return hashlib.sha256(np.asarray([young_Pa],dtype='<f8').tobytes()).hexdigest()


def _ids(values):
    if not isinstance(values,(list,tuple)) or not values or any(not isinstance(s,str) or not s for s in values) or len(set(values))!=len(values):raise ValueError('Complete unique acquisition IDs required')
    return tuple(values)


def validate_static_catalog(rows,*,expected_acquisitions_by_object,expected_object_counts=None):
    """Private catalog only; same known support categories may repeat by role.

    Whole object/material/counterfactual/acquisition identities cannot cross
    roles. Explicit complete prospective ledgers prevent partial-case filtering.
    No Young value, support label or private catalog is a public model feature.
    """
    if not isinstance(rows,list) or not rows or not isinstance(expected_acquisitions_by_object,dict):raise ValueError('Explicit complete static object catalog/ledgers required')
    expected={o:_ids(a) for o,a in expected_acquisitions_by_object.items()};objects={};materials={};acquisitions={};episodes=set();modalities=set();seen_acquisitions={};counts={r:0 for r in ROLES}
    for row in rows:
        role=row.get('role');obj=row.get('object_id');episode=row.get('episode_id');kind=row.get('observation_kind')
        if role not in ROLES or not isinstance(obj,str) or not obj or not isinstance(episode,str) or not episode or episode in episodes or kind not in ('ideal_tracked_points','rgbd') or row.get('support_category') not in SUPPORT_CATEGORIES:raise ValueError('Unique static episode/object IDs, role and known support category required')
        episodes.add(episode);modalities.add(kind)
        if obj not in expected or _ids(row.get('complete_acquisition_ids'))!=expected[obj]:raise ValueError('Complete prospective object acquisition ledger differs')
        if not set(_ids(row.get('acquisition_ids'))).issubset(expected[obj]):raise ValueError('Undeclared static acquisition/counterfactual')
        seen_acquisitions.setdefault(obj,set()).update(row['acquisition_ids'])
        fingerprint=material_fingerprint(row.get('young_Pa'))
        if obj in objects and objects[obj]!=(role,fingerprint,row['support_category']):raise ValueError('Object/counterfactual changed role or physical metadata')
        if fingerprint in materials and materials[fingerprint]!=role:raise ValueError('Continuous material draw reused across roles')
        if obj not in objects:counts[role]+=1
        objects[obj]=(role,fingerprint,row['support_category']);materials[fingerprint]=role
        for acquisition in expected[obj]:
            if acquisition in acquisitions and acquisitions[acquisition]!=(obj,role):raise ValueError('Acquisition ledger reused across objects/roles')
            acquisitions[acquisition]=(obj,role)
    if set(objects)!=set(expected):raise ValueError('All declared objects must remain present')
    if any(seen_acquisitions[o]!=set(a) for o,a in expected.items()):raise ValueError('All declared acquisitions/counterfactuals must remain present')
    if len(modalities)!=1:raise ValueError('Ideal tracked points and RGB-D need separate declared catalogs')
    if expected_object_counts is not None and counts!=expected_object_counts:raise ValueError('Complete whole-object role counts differ')
    return dict(object_counts=counts,objects=len(objects),episodes=len(rows),observation_kind=next(iter(modalities)),whole_object_material_and_acquisition_roles_disjoint=True,known_support_categories_may_repeat=True,unseen_support_layout_generalization=False,is_temporal_forecast=False,private_catalog_not_public_features=True)
