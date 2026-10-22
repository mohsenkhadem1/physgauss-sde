"""Check declared hard-support patch conformity on the canonical straight mesh.

Node membership alone cannot certify that a finite fixed patch has the intended
continuous edge. A triangle crossing x=.06 or .18 fails this check, even when
its mask contains some plausible fixed nodes. No masks or mesh points change.
"""
import numpy as np
from experiments.campaign_block_surface_controls import support_groups, SIZE
from experiments.campaign_static_p2_block import EDGES


def support_trace_certificate(model, kind, *, tolerance_m=1e-13):
    if not np.isfinite(tolerance_m) or not 0 < tolerance_m <= 1e-12:
        raise ValueError('Explicit small positive geometry tolerance required')
    X=np.asarray(model.rest);T=np.asarray(model.tets);faces=np.asarray(model.faces)
    if X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all():
        raise ValueError('Finite canonical reference geometry required')
    groups,_=support_groups(X,kind)
    if not np.array_equal(np.asarray(model.mask),np.repeat(groups.any(0)[:,None],3,1)):
        raise ValueError('Exact declared fixed-node mask required')
    if len(np.unique(np.sort(faces,axis=1),axis=0))!=len(faces):
        raise ValueError('Unique boundary triangles required')
    # Straight T10 edges are necessary for planar patch/triangle reasoning.
    for start in range(0,len(T),4096):
        corner=X[T[start:start+4096]]
        midpoint=np.stack([.5*(corner[:,i]+corner[:,j]) for i,j in EDGES],axis=1)
        if np.max(abs(midpoint-X[model.elements[start:start+4096,4:]]))>tolerance_m:
            raise ValueError('Straight-reference quadratic edge nodes required')
    P=X[faces];left=np.all(abs(P[:,:,0])<=tolerance_m,axis=1)
    right=np.all(abs(P[:,:,0]-SIZE[0])<=tolerance_m,axis=1)
    bottom=np.all(abs(P[:,:,2]+SIZE[2]/2)<=tolerance_m,axis=1)
    crossing=0
    if kind=='separated_bottom_patches':
        lo=P[:,:,0].min(1);hi=P[:,:,0].max(1)
        for cut in (.06,.18):
            crossing+=int(np.sum(bottom&(lo<cut-tolerance_m)&(hi>cut+tolerance_m)))
        selected=bottom&((hi<=.06+tolerance_m)|(lo>=.18-tolerance_m))
        expected_area=2*.06*SIZE[1]
    else:
        selected={'left_face':left,'right_face':right,'both_ends':left|right,'bottom_face':bottom}[kind]
        expected_area={'left_face':SIZE[1]*SIZE[2],'right_face':SIZE[1]*SIZE[2],
                       'both_ends':2*SIZE[1]*SIZE[2],'bottom_face':SIZE[0]*SIZE[1]}[kind]
    area=.5*np.linalg.norm(np.cross(P[:,1]-P[:,0],P[:,2]-P[:,0]),axis=1)
    actual=float(area[selected].sum())
    conforming=bool(crossing==0 and np.isclose(actual,expected_area,rtol=1e-12,atol=1e-15))
    return dict(conforming=conforming,support=kind,selected_fixed_triangles=int(selected.sum()),
                triangles_crossing_declared_patch_edges=crossing,
                fixed_reference_area_m2=actual,expected_reference_area_m2=expected_area,
                geometry_tolerance_m=tolerance_m,mask_or_geometry_changed=False,
                scope='Canonical straight T10 boundary mesh and declared complete planar faces/patches')
