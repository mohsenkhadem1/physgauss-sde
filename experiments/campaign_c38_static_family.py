"""NEW fixed public forces and conforming support traces for qualification only.

No observations/object draws. Constant known density/gravity, polynomial
stable-NH material, reference-attached uniform top band/world dead traction.
The force compiler has no Young modulus, support or hypothesis argument.
"""
from __future__ import annotations
import hashlib
import numpy as np
from experiments.campaign_block_surface_controls import SIZE,EDGES,surface_control,support_groups,wrench

DENSITY=625.;GRAVITY=9.81;POISSON=.35
SUPPORTS=('both_ends','separated_bottom_patches')
E_BOUNDS=(75000.,150000.)

def array_record(a):
    x=np.ascontiguousarray(a);return dict(shape=list(x.shape),dtype=x.dtype.str,sha256=hashlib.sha256(x.tobytes()).hexdigest())

def compile_known_forces(rest,tets,faces,elements,Fz):
    """Full consistent body + PUBLIC signed surface load, in newtons.

    Integral T10 shape weights are −V/20 at four corners and V/5 at six edges.
    These are signed FE load coefficients, NOT positive particle masses. Loads
    at constrained nodes are retained; they contribute to fixture reactions.
    """
    X=np.asarray(rest,dtype=np.float64);T=np.asarray(tets);E=np.asarray(elements)
    if X.ndim!=2 or X.shape[1]!=3 or T.ndim!=2 or T.shape[1]!=4 or E.shape!=(len(T),10) or T.dtype.kind not in 'iu' or E.dtype.kind not in 'iu' or not np.isfinite(X).all() or not np.array_equal(E[:,:4],T) or not np.isfinite(Fz) or not -2<=Fz<=2:raise ValueError('Finite compatible affine-reference T10 geometry and public Fz in[-2,2]N required')
    corners=X[T];volume=np.linalg.det(np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2))/6
    if not np.isfinite(volume).all() or (volume<=0).any():raise ValueError('Positive reference tetrahedron volumes required')
    for k,(i,j) in enumerate(EDGES):
        if np.max(abs(X[E[:,4+k]]-(X[T[:,i]]+X[T[:,j]])/2))>1e-14:raise ValueError('True physical midpoint T10 reference nodes required')
    if not np.isclose(volume.sum(),np.prod(SIZE),rtol=1e-13,atol=1e-16):raise ValueError('Declared same physical block required')
    unit,meta=surface_control(X,T,faces,'top_middle_down',E);surface=unit*(float(Fz)/(-.12));integral=np.zeros(len(X));local=volume[:,None]*np.array([-.05]*4+[.2]*6)
    np.add.at(integral,E,local);body=np.zeros_like(X);body[:,2]=-DENSITY*GRAVITY*integral;total=body+surface
    SF,SM=wrench(X,surface);BF,BM=wrench(X,body);expected=np.array([0.,0.,float(Fz)]);expectedbody=np.array([0.,0.,-DENSITY*np.prod(SIZE)*GRAVITY])
    if np.max(abs(SF-expected))>1e-12 or np.max(abs(BF-expectedbody))>1e-12 or np.max(abs(SM))>1e-12 or np.max(abs(BM))>1e-12:raise ValueError('Known force/wrench normalization failed')
    return dict(total_N=total,surface_N=surface,body_N=body,body_integral_weights_m3=integral,metadata=dict(pattern='top_middle_down scaled public Fz',Fz_N=float(Fz),density_kg_m3=DENSITY,gravity_m_s2=GRAVITY,physical_mass_kg=DENSITY*float(volume.sum()),reference_patch_area_m2=meta['reference_patch_area_m2'],reference_patch='z=+0.04 and0.06<=x<=0.18; full y width',law='Reference material patch, world dead force direction; deformed wrench must be recomputed',surface_rest_resultant_N=SF,surface_rest_moment_Nm=SM,body_rest_resultant_N=BF,body_rest_moment_Nm=BM,origin_m=[.12,0.,0.],units=dict(force='N',moment='Nm',shape_integral='m3'),no_E_or_BC_input=True,consistent_signed_P2_loads_not_particle_masses=True,includes_fixed_node_loads=True,actual_total_force=array_record(total)))

def boundary_faces6(elements,faces):
    E=np.asarray(elements);F=np.asarray(faces);edges={}
    if E.ndim!=2 or E.shape[1]!=10 or F.ndim!=2 or F.shape[1]!=3:raise ValueError('T10 and original corner boundary faces required')
    for tet in E:
        for k,(i,j) in enumerate(EDGES):
            key=tuple(sorted(map(int,tet[[i,j]])));node=int(tet[4+k])
            if key in edges and edges[key]!=node:raise ValueError('Nonconforming P2 edge')
            edges[key]=node
    return np.array([np.r_[f,[edges[tuple(sorted(map(int,f[[i,j]])))] for i,j in ((0,1),(0,2),(1,2))]] for f in F],dtype=np.int64)

def mask_trace_certificate(rest,elements,faces,mask,support):
    X=np.asarray(rest);M=np.asarray(mask)
    if support not in SUPPORTS or M.shape!=X.shape or M.dtype!=np.bool_:raise ValueError('Exact supported xyz fixed-mask declaration required')
    groups,active=support_groups(X,support);expected=np.repeat(groups.any(0)[:,None],3,1)
    if not np.array_equal(M,expected):raise ValueError('Actual mask must equal physical support on ALL P2 nodes')
    six=boundary_faces6(elements,faces);tri=X[six[:,:3]];tol=1e-13
    if support=='both_ends':selected=np.all(abs(tri[:,:,0])<tol,1)|np.all(abs(tri[:,:,0]-.24)<tol,1)
    else:
        bottom=np.all(abs(tri[:,:,2]+.04)<tol,1);lo=tri[:,:,0].min(1);hi=tri[:,:,0].max(1)
        if np.any(bottom&(((lo<.06-tol)&(hi>.06+tol))|((lo<.18-tol)&(hi>.18+tol)))):raise ValueError('Reference face straddles a physical patch edge; support is not conforming')
        selected=bottom&((hi<=.06+tol)|(lo>=.18-tol))
    if not selected.any() or not M[six[selected]].all():raise ValueError('Every six-node supported trace must be exactly fixed')
    area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1);actual=float(area[selected].sum());expectedarea=2*.08*.08 if support=='both_ends' else.12*.08
    if abs(actual-expectedarea)>1e-13:raise ValueError('Support trace area does not match physical patches')
    return dict(passed=True,support=support,groups=active.tolist(),supported_triangles=int(selected.sum()),physical_trace_area_m2=actual,all6_P2_shape_coefficients_fixed=True,whole_polynomial_trace_zero_for_admissible_displacements=True,mask=array_record(M),reference_faces6=array_record(six),geometry_only_certificate_not_mechanical_convergence=True)

def build_geometry(cells,young,support,Fz,*,rule='keast24'):
    from experiments.campaign_p2_keast_quadrature import KeastBulkP2Geometry
    from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
    if len(cells)!=3 or any(type(n) is not int or n<2 or n%2 for n in cells) or cells[0]%4 or not E_BOUNDS[0]<=young<=E_BOUNDS[1] or support not in SUPPORTS or rule not in ('keast24','exact80'):raise ValueError('Declared uniform conforming mesh/material/support/rule required')
    cls=KeastBulkP2Geometry if rule=='keast24' else ExactBulkP2Geometry;m=cls(tuple(cells),young=young,poisson=POISSON,density=DENSITY,gravity=GRAVITY,pattern='top_middle_down',support=support,mapping='uniform')
    compiled=compile_known_forces(m.rest,m.tets,m.faces,m.elements,Fz);defect=float(np.max(abs(compiled['body_integral_weights_m3']-m.body_integral_weights)))
    if defect*DENSITY*GRAVITY>1e-12:raise ValueError('Analytic body integral disagrees with original exact80 quadrature')
    m.load=compiled['total_N'];m.force_metadata=compiled['metadata'];m.public_force_components=compiled;m.mask_trace=mask_trace_certificate(m.rest,m.elements,m.faces,m.mask,support);m.consistent_body_weight_max_difference_m3=defect
    return m
