"""Finite-point spatial-scope guard around immutable v01 ideal exporter.

A complete nodal file does not qualify arbitrary material points, a surface or
pixels. Only explicitly reviewed reference points may be observed here. Dense
RGB-D requires separate spatial/render/visibility evidence. No real dataset or
admission is generated. The old source/inventory is intentionally unchanged.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from experiments import campaign_c38_static_export as original
from experiments.campaign_c38_static_study import _record,verify_static_admission
from experiments.campaign_c38_data import load_public_geometry

SOURCE='experiments/campaign_c38_static_export_v02.py'
ROUND_OFF_ULPS=64


def export_reviewed_ideal_static_prefix(root,manifest,*,qualified_point_indices,**arguments):
    """Reject an unqualified attachment BEFORE v01 can open private q.

    Admission gate ``material_point_scope`` must equal the independently
    hash-bound review's ``reviewed_material_point_scope``. Its ``points`` record
    has exactly material_points_m[P,3]; ``spatial_evidence`` references already
    reviewed gate spatial-mesh evidence. Scope uses reference coordinates only.
    The same scope explicitly lists the admitted field path/SHA pairs to which
    that evidence applies. Every attachment supplies the index of its qualified reference point.
    Equality permits only 64*float64eps*max(reference extent,|coordinate|,tiny)
    arithmetic error, not a spatial discretization/noise acceptance tolerance.
    """
    gate=verify_static_admission(manifest,root=root)
    expected=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if gate.get('source_sha256',{}).get(SOURCE)!=expected:
        raise ValueError('Current finite-point exporter source must be pinned in admission')
    scope=gate.get('material_point_scope')
    if not isinstance(scope,dict) or set(scope)!={'kind','points','spatial_evidence','admitted_fields'} or scope['kind']!='finite_reference_material_points_only':
        raise ValueError('Explicit finite material-point spatial qualification required')
    review=json.loads(_record(root,gate['review']).read_text())
    if review.get('approved_material_point_scope') is not True or review.get('reviewed_material_point_scope')!=scope:
        raise ValueError('Exact independently reviewed material-point scope required')
    evidence=scope['spatial_evidence'];spatial=[r for r in gate['evidence'] if r.get('kind')=='spatial_mesh']
    if not isinstance(evidence,list) or not evidence or any(r not in spatial for r in evidence):
        raise ValueError('Point scope requires hash-bound reviewed spatial-mesh evidence')
    for r in evidence:_record(root,r)
    bindings=scope['admitted_fields']
    if not isinstance(bindings,list) or not bindings or any(not isinstance(r,dict) or set(r)!={'path','sha256'} for r in bindings):
        raise ValueError('Explicit point-scope admitted-field bindings required')
    keys=[(r['path'],r['sha256']) for r in bindings]
    admitted={(r['path'],r['sha256']) for r in gate['admitted_static_fields']}
    if len(set(keys))!=len(keys) or not set(keys).issubset(admitted):
        raise ValueError('Point spatial evidence must bind exact admitted fields')
    ledger=arguments.get('acquisitions');cutoff=arguments.get('last_acquisition')
    if not isinstance(ledger,list) or type(cutoff) is not int or cutoff<0:
        raise ValueError('Explicit acquired ledger required before point-scope/private reads')
    for entry in ledger:
        if not isinstance(entry,dict) or type(entry.get('acquisition_index')) is not int:
            raise ValueError('Integer acquired ledger metadata required')
        if entry['acquisition_index']<=cutoff:
            field=entry.get('field',{})
            if not isinstance(field,dict) or (field.get('path'),field.get('sha256')) not in set(keys):
                raise ValueError('Acquired field lacks reviewed material-point spatial scope')
    with np.load(_record(root,scope['points']),allow_pickle=False) as z:
        if set(z.files)!={'material_points_m'}:raise ValueError('Exact qualified-reference-point allowlist required')
        points=z['material_points_m'].copy()
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all() or len(np.unique(points,axis=0))!=len(points):
        raise ValueError('Finite unique reviewed reference points required')
    ids=np.asarray(qualified_point_indices)
    if ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(points)) or len(np.unique(ids))!=len(ids):
        raise ValueError('Explicit unique qualified reference-point indices required')
    geometry=load_public_geometry(root,arguments['geometry_record'])
    _,_,requested=original._points(geometry,arguments['element_indices'],arguments['barycentric'])
    extent=float(np.max(np.ptp(geometry['rest_m'],axis=0)))
    scale=max(extent,float(np.max(abs(geometry['rest_m']))),np.finfo(np.float64).tiny)
    tolerance=ROUND_OFF_ULPS*np.finfo(np.float64).eps*scale
    if requested.shape!=points[ids].shape or not np.allclose(requested,points[ids],rtol=0,atol=tolerance):
        raise ValueError('Requested attachment is outside reviewed material-point scope')
    public,private=original.export_ideal_static_prefix(root,manifest,**arguments)
    private.update(material_point_scope=scope,qualified_point_indices=ids.tolist(),coordinate_roundoff_tolerance_m=tolerance,
                   spatial_scope='Only the explicitly reviewed finite reference points; no whole-field/surface/pixel error qualification')
    return public,private
