"""Prospective orthogonal static probe loads; no mechanics or data execution.

The force compiler accepts geometry and a PUBLIC action only. Stiffness and
support labels are not arguments. C39 is a distinct experiment, not a change to
the frozen C38 down/up comparison. No C39 observations are authorised by import.
"""
from __future__ import annotations
import numpy as np
from experiments.campaign_c38_static_family import compile_known_forces,wrench,array_record,boundary_faces6

ACTIONS={'gravity':(0.,0.,0.),'middle_down_2N':(0.,0.,-2.),
         'middle_side_2N':(0.,2.,0.),'left_down_2N':(0.,0.,-2.),
         'canonical_up_1N':(0.,0.,1.)}


def compile_known_action(rest,tets,faces,elements,action_id):
    """Same conforming full-width top band; rotate world dead-force direction."""
    if action_id not in ACTIONS:raise ValueError('Declared public action required')
    unit=compile_known_forces(rest,tets,faces,elements,1.)
    vector=np.asarray(ACTIONS[action_id],dtype=float)
    # The old +1N compiler supplies the exact T6 reference shape integrals;
    # rotating the dead-force vector does not change material-patch membership.
    surface=unit['surface_N'][:,2,None]*vector[None,:]
    centroid=np.array([.12,0.,.04]);area=unit['metadata']['reference_patch_area_m2']
    patch='z=+.04, .06<=x<=.18, full y width'
    if action_id=='left_down_2N':
        X=np.asarray(rest);six=boundary_faces6(elements,faces);corners=X[six[:,:3]]
        top=np.all(abs(corners[:,:,2]-.04)<1e-13,axis=1)
        lo=corners[:,:,0].min(axis=1);hi=corners[:,:,0].max(axis=1)
        if np.any(top & (lo<.06-1e-13) & (hi>.06+1e-13)):
            raise ValueError('Top faces must conform to left patch edge x=.06')
        selected=top & (hi<=.06+1e-13)
        triangles=corners[selected]
        areas=.5*np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
        area=float(areas.sum())
        if not np.isclose(area,.06*.08,rtol=0,atol=1e-14):raise ValueError('Exact full left patch area')
        surface=np.zeros_like(X)
        # Exact constant-traction T6 integrals: vertex0, each midsideA/3.
        for ids,A in zip(six[selected],areas):np.add.at(surface,ids[3:],A/(3*area)*vector)
        centroid=np.array([.03,0.,.04]);patch='z=+.04, 0<=x<=.06, full y width'
    body=unit['body_N'].copy();total=body+surface
    force,moment=wrench(np.asarray(rest),surface)
    expected_moment=np.cross(centroid-np.array([.12,0.,0.]),vector)
    if np.max(abs(force-vector))>1e-12 or np.max(abs(moment-expected_moment))>1e-12:
        raise ValueError('Public action resultant or physical moment failed')
    return dict(total_N=total,body_N=body,surface_N=surface,
                body_integral_weights_m3=unit['body_integral_weights_m3'].copy(),
                metadata=dict(action_id=action_id,force_vector_N=vector.tolist(),
                              rest_surface_resultant_N=force.tolist(),rest_surface_moment_Nm=moment.tolist(),
                              moment_origin_m=[.12,0.,0.],reference_patch=patch,
                              reference_patch_area_m2=area,
                              law='Reference-material band; prescribed world dead-force vector',
                              no_material_or_support_input=True,includes_fixed_node_loads=True,
                              actual_total_force=array_record(total)))
