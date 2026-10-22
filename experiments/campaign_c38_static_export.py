"""Source-only admitted-static to ideal-point observation adapter.

No dataset admission, random draws, renders or physical solves occur here. An
explicit acquired ledger chooses fields BEFORE their private arrays are opened.
Provided measurement errors are additive *realizations*, not posterior or shape
covariances. Ideal P2 material correspondences are not RGB-D perception. The API
boundary is trusted application logic, not an operating-system sandbox.
"""
from __future__ import annotations
import hashlib
import numpy as np
from experiments.campaign_c38_data import load_public_geometry
from experiments.campaign_c38_static_data import load_public_static_controls
from experiments.campaign_c38_static_study import (
    _record, verify_static_admission, load_admitted_static_targets,
)

_EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
_GEOMETRY_KEYS=('rest_m','elements','tets','faces')


def force_row_sha256(row):
    """Canonical physical N-valued row digest; no material/support argument."""
    row=np.asarray(row)
    if row.ndim!=2 or row.shape[1]!=3 or not np.isfinite(row).all():
        raise ValueError('Finite full reference-nodal force row[N,3] required')
    return hashlib.sha256(np.ascontiguousarray(row,dtype='<f8').tobytes()).hexdigest()


def _points(geometry,element_indices,barycentric):
    ids=np.asarray(element_indices);b=np.asarray(barycentric)
    if ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(geometry['elements'])):
        raise ValueError('Public reference element indices required')
    if b.shape!=(len(ids),4) or not np.isfinite(b).all() or np.any(b<0) or np.any(b>1) or not np.allclose(b.sum(1),1,rtol=0,atol=1e-14):
        raise ValueError('Reference tetrahedral barycentric material points required')
    N=np.concatenate([b*(2*b-1),np.stack([4*b[:,i]*b[:,j] for i,j in _EDGES],-1)],-1)
    nodes=geometry['elements'][ids]
    return nodes,N,np.einsum('pi,pij->pj',N,geometry['rest_m'][nodes])


def export_ideal_static_prefix(root,manifest,*,object_id,geometry_record,
                               controls_record,acquisitions,last_acquisition,
                               element_indices,barycentric,
                               measurement_errors_m,sensor_covariance_m2):
    """Return (PUBLIC inputs, PRIVATE provenance) without a target/full-field leak.

    ``acquisitions`` is an authored integer ledger of dictionaries with exactly
    acquisition_index/load_state_index/field. Future entries' arrays are never
    read or hashed. A reviewed admitted field MUST additionally bind
    ``public_controls`` (exact path/SHA record) and ``load_vector_sha256`` to the
    physical nodal force row used to obtain that field. No binding is inferred
    from a requested load ID or successful solver exit.

    Caller supplies errors for ACQUIRED rows only, shape[A,P,3]; this function
    never draws or selects noise. Supplying zeros is an explicitly noiseless
    ideal diagnostic. The separate private receipt must never be a B feature.
    No view renderer is implemented or claimed.
    """
    if not isinstance(object_id,str) or not object_id or type(last_acquisition) is not int or last_acquisition<0:
        raise ValueError('Explicit object identity and acquired ledger cutoff required')
    gate=verify_static_admission(manifest,root=root)
    _record(root,geometry_record);_record(root,controls_record)
    geometry=load_public_geometry(root,geometry_record)
    controls=load_public_static_controls(root,controls_record,number_of_nodes=len(geometry['rest_m']))
    if not isinstance(acquisitions,list) or not acquisitions:
        raise ValueError('Explicit prospective acquisition ledger required')
    indices=[];selected=[]
    for entry in acquisitions:
        if not isinstance(entry,dict) or set(entry)!= {'acquisition_index','load_state_index','field'}:
            raise ValueError('Exact acquisition metadata allowlist required')
        a=entry['acquisition_index'];state=entry['load_state_index']
        if type(a) is not int or a<0 or type(state) is not int or state<0 or not isinstance(entry['field'],dict):
            raise ValueError('Integer acquisition/load indices and private field record required')
        indices.append(a)
        if a<=last_acquisition:selected.append(entry)
    if any(b<=a for a,b in zip(indices,indices[1:])) or not selected:
        raise ValueError('Ordered unique ledger with at least one acquired entry required')
    nodes,N,X=_points(geometry,element_indices,barycentric)
    errors=np.asarray(measurement_errors_m);C=np.asarray(sensor_covariance_m2)
    if errors.shape!=(len(selected),len(X),3) or not np.isfinite(errors).all():
        raise ValueError('Finite provided errors for acquired entries only required')
    if C.shape!=(3*len(X),3*len(X)) or not np.isfinite(C).all() or not np.allclose(C,C.T,rtol=1e-12,atol=1e-18):
        raise ValueError('Declared sensor covariance shape/symmetry required')
    try:np.linalg.cholesky(C)
    except np.linalg.LinAlgError as error:raise ValueError('Positive definite measurement covariance required') from error
    values=[];provenance=[]
    for entry in selected:
        record=entry['field'];state=entry['load_state_index']
        if record.get('object_id')!=object_id or record.get('load_state_index')!=state:
            raise ValueError('Acquired object/load does not bind private field')
        matches=[r for r in gate['admitted_static_fields'] if (r['path'],r['sha256'])==(record.get('path'),record.get('sha256'))]
        if len(matches)!=1:raise ValueError('Acquired field is not uniquely admitted')
        bound=matches[0]
        positions=np.flatnonzero(controls['load_state_indices']==state)
        if len(positions)!=1:raise ValueError('Acquired load state absent from public controls')
        if bound.get('public_controls')!=controls_record or bound.get('load_vector_sha256')!=force_row_sha256(controls['load_vectors_N'][positions[0]]):
            raise ValueError('Reviewed physical field/public control force binding differs')
        field=load_admitted_static_targets(root,manifest,record)
        if any(not np.array_equal(field.geometry[k],geometry[k]) or field.geometry[k].dtype!=geometry[k].dtype for k in _GEOMETRY_KEYS):
            raise ValueError('Acquired field/public rest geometry differs')
        values.append(np.einsum('pi,pij->pj',N,field.positions_m[nodes]))
        provenance.append(dict(acquisition_index=entry['acquisition_index'],load_state_index=state,field=dict(record)))
    observation=dict(information_role='PUBLIC_IDEAL_TRACKED_MATERIAL_POINTS_STATIC_B',
                     observation_kind='ideal_tracked_points',is_RGBD_perception=False,is_temporal_forecast=False,
                     acquisition_indices=np.asarray([e['acquisition_index'] for e in selected],dtype=np.int64),
                     load_state_indices=np.asarray([e['load_state_index'] for e in selected],dtype=np.int64),
                     material_points_m=X,observed_positions_m=np.stack(values)+errors,
                     sensor_covariance_m2=C.copy())
    public=dict(information_role='PUBLIC_ACQUIRED_IDEAL_STATIC_POINTS_B',
                geometry={k:v.copy() for k,v in geometry.items()},
                known_controls={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in controls.items()},
                observations=observation,observation_kind='ideal_tracked_points',
                is_RGBD_perception=False,is_temporal_forecast=False)
    private=dict(information_role='PRIVATE_EXPORT_PROVENANCE_NOT_MODEL_INPUT',object_id=object_id,
                 domain_sha256=manifest['domain_sha256'],static_admission=dict(manifest['static_admission']),
                 geometry_record=dict(geometry_record),controls_record=dict(controls_record),acquired_fields=provenance,
                 provided_measurement_error_sha256=hashlib.sha256(np.ascontiguousarray(errors,dtype='<f8').tobytes()).hexdigest(),
                 no_random_draws=True,no_future_field_arrays_opened=True)
    return public,private
