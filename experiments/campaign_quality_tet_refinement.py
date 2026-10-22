"""Geometry-only red refinement with shortest central-octahedral diagonal.

No physics, quadrature, responses or source mesh mutation. Corner tetrahedra
and face subdivisions match ordinary midpoint red refinement; central choices
are purely geometric. Existing frozen refiner/results are untouched.
"""
from __future__ import annotations
import numpy as np


def boundary_faces(tets):
    owners={}
    for tet in np.asarray(tets):
        for j in range(4):owners.setdefault(tuple(sorted(np.delete(tet,j))),[]).append(int(tet[j]))
    if any(len(v)>2 for v in owners.values()):raise ValueError('Nonmanifold tetrahedral face multiplicity')
    return np.asarray([f for f,v in sorted(owners.items()) if len(v)==1],dtype=np.int64)


def refine_shortest(points,tets):
    X=np.asarray(points,dtype=np.float64);T=np.asarray(tets,dtype=np.int64)
    if X.ndim!=2 or X.shape[1]!=3 or T.ndim!=2 or T.shape[1]!=4 or not np.isfinite(X).all():raise ValueError('Explicit finite tetrahedral geometry required')
    vertices=list(X.copy());midpoints={};children=[];choices=[]
    def midpoint(a,b):
        edge=tuple(sorted((int(a),int(b))))
        if edge not in midpoints:midpoints[edge]=len(vertices);vertices.append((X[a]+X[b])/2)
        return midpoints[edge]
    for a,b,c,d in T:
        ab,ac,ad,bc,bd,cd=[midpoint(i,j) for i,j in [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]]
        children.extend([[a,ab,ac,ad],[ab,b,bc,bd],[ac,bc,c,cd],[ad,bd,cd,d]])
        options=[(ab,cd,[ac,ad,bd,bc]),(ac,bd,[ab,ad,cd,bc]),(ad,bc,[ab,ac,cd,bd])]
        lengths=[]
        for u,v,_ in options:
            delta=np.asarray(vertices[u])-np.asarray(vertices[v]);lengths.append(float(delta@delta))
        # Near-equal geometric diagonals differ only by float roundoff. Use a
        # scale-relative64eps window, then invariant global vertex identities;
        # lexicographic world coordinates would change under passive rotation.
        tolerance=64*np.finfo(float).eps*max(lengths)
        eligible=[j for j,value in enumerate(lengths) if value<=min(lengths)+tolerance]
        selected=min(eligible,key=lambda j:tuple(sorted(options[j][:2])))
        u,v,cycle=options[selected];choices.append(selected)
        for j in range(4):children.append([u,v,cycle[j],cycle[(j+1)%4]])
    Y=np.asarray(vertices);S=np.asarray(children,dtype=np.int64)
    det=np.linalg.det((Y[S[:,1:]]-Y[S[:,:1]]).transpose(0,2,1))
    if np.any(det==0):raise ValueError('Degenerate child tetrahedron')
    negative=det<0;tmp=S[negative,1].copy();S[negative,1]=S[negative,2];S[negative,2]=tmp
    return Y,S,boundary_faces(S),np.bincount(choices,minlength=3)


def shape_metrics(points,tets):
    X=np.asarray(points);T=np.asarray(tets);P=X[T];Dm=(P[:,1:]-P[:,:1]).transpose(0,2,1)
    det=np.linalg.det(Dm);V=det/6
    if np.any(V<=0):raise ValueError('Positive orientation required')
    sv=np.linalg.svd(Dm,compute_uv=False);condition=sv[:,0]/sv[:,-1]
    pairs=np.array([(a,b) for a in range(4) for b in range(a+1,4)]);lengths=np.linalg.norm(P[:,pairs[:,0]]-P[:,pairs[:,1]],axis=-1);h=lengths.max(1)
    mean_ratio=12*(3*V)**(2/3)/np.sum(lengths**2,axis=1)
    surface=np.zeros(len(T))
    for f in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
        tri=P[:,f];surface+=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    radius_in=3*V/surface
    center=np.linalg.solve(Dm.transpose(0,2,1),(.5*np.sum((P[:,1:]-P[:,:1])**2,axis=2))[...,None])[...,0]
    radius_circum=np.linalg.norm(center,axis=1);radius_ratio=3*radius_in/radius_circum
    def stats(a):return dict(minimum=float(a.min()),quantile05=float(np.quantile(a,.05)),median=float(np.median(a)),quantile95=float(np.quantile(a,.95)),maximum=float(a.max()))
    return dict(nodes=len(X),tetrahedra=len(T),volume_m3=float(V.sum()),minimum_cell_volume_m3=float(V.min()),
        maximum_edge_length_m=stats(h),rest_Dm_singular_condition=stats(condition),
        permutation_invariant_mean_ratio=stats(mean_ratio),radius_ratio_3rin_over_Rcircum=stats(radius_ratio))


def topology_certificate(points,tets,original_points,original_boundary_faces):
    X=np.asarray(points);T=np.asarray(tets);base=np.asarray(original_points);original=np.asarray(original_boundary_faces);owners={}
    for tet in T:
        for j in range(4):owners.setdefault(tuple(sorted(np.delete(tet,j))),[]).append(int(tet[j]))
    multiplicities=np.array([len(v) for v in owners.values()]);boundary=[f for f,v in owners.items() if len(v)==1]
    opposed=True
    for f,v in owners.items():
        if len(v)==2:
            a,b,c=X[list(f)];n=np.cross(b-a,c-a);opposed &= bool(((X[v[0]]-a)@n)*((X[v[1]]-a)@n)<0)
    F=np.asarray(boundary,dtype=int);tri=X[F];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    original_tri=base[original];normals=np.cross(original_tri[:,1]-original_tri[:,0],original_tri[:,2]-original_tri[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    plane_error=np.max(np.abs(np.einsum('fvc,bc->fbv',tri,normals)-np.einsum('bc,bc->b',original_tri[:,0],normals)[None,:,None]),axis=2)
    membership=plane_error<1e-12;areas_per_original=np.array([areas[membership[:,j]].sum() for j in range(len(original))])
    reference_area=np.linalg.norm(np.cross(original_tri[:,1]-original_tri[:,0],original_tri[:,2]-original_tri[:,0]),axis=1)/2
    edge_counts={}
    for face in F:
        for a,b in [(0,1),(0,2),(1,2)]:edge_counts[tuple(sorted(face[[a,b]]))]=edge_counts.get(tuple(sorted(face[[a,b]])),0)+1
    vertices=np.unique(F);edges=list(edge_counts);parent={int(v):int(v) for v in vertices}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for a,b in edges:parent[find(a)]=find(b)
    components=len({find(int(v)) for v in vertices});euler=len(vertices)-len(edges)+len(F)
    passed=bool(np.max(multiplicities)<=2 and opposed and np.all(membership.sum(1)==1) and np.allclose(areas_per_original,reference_area,rtol=0,atol=1e-13)
        and all(n==2 for n in edge_counts.values()) and components==1 and euler==2 and len(np.unique(X,axis=0))==len(X))
    return dict(passed=passed,face_multiplicity_counts={str(int(k)):int(np.sum(multiplicities==k)) for k in np.unique(multiplicities)},
        shared_face_opposite_side_check=opposed,boundary_triangles=len(F),boundary_edge_incidence_two=all(n==2 for n in edge_counts.values()),
        boundary_components=components,boundary_Euler_characteristic=euler,maximum_original_face_area_error_m2=float(np.max(abs(areas_per_original-reference_area))),
        every_boundary_triangle_on_exact_original_face=bool(np.all(membership.sum(1)==1)),duplicate_coordinate_vertices=len(X)-len(np.unique(X,axis=0)))


def geometry(level=0):
    """Current physical two-tet domain, original five nodes preserved as IDs0..4."""
    if type(level) is not int or not 0<=level<=8:raise ValueError('Explicit finite refinement level0..8 required')
    X=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1],[.1,.1,.1]])
    T=np.array([[0,1,2,3],[4,2,1,3]],dtype=np.int64);F=boundary_faces(T)
    for _ in range(level):X,T,F,_=refine_shortest(X,T)
    return X,T,F
