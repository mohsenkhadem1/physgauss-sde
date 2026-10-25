"""NEW arbitrary-declared-resolution geometry/query adapter, no old mutations.

Same affine Freudenthal T10 topology/arithmetic on overlap1/2/4/8, admitting
3/6/12. Same bulk law/80rule/fixture/load law; explicit finest12 diagnostics.
No model, field, factor or solver executes on import.
"""
from functools import lru_cache
import itertools
import numpy as np
from scipy import sparse
from experiments import campaign_active_fixture_gaussian_load_v02 as loads
from experiments import campaign_surface_p2_static_v02 as core
from experiments.campaign_static_p2_block import EDGES,enrich,shapes
from experiments.run_clamped_surface_p2_qualification_v01 import weighted_rigid_metrics
volume_rule=core.volume_rule

@lru_cache(maxsize=7)
def _topology(n):
    if type(n)is not int or n not in(1,2,3,4,6,8,12):raise ValueError('Declared overlap1/2/4/8 and new3/6/12 only')
    X=np.array(list(itertools.product(np.linspace(0,.36,4*n+1),np.linspace(0,.08,n+1),np.linspace(0,.08,n+1))))
    node=lambda i,j,k:i*(n+1)**2+j*(n+1)+k;tets=[]
    for i,j,k in itertools.product(range(4*n),range(n),range(n)):
        for perm in itertools.permutations(range(3)):
            vertex=np.array([i,j,k]);path=[vertex.copy()]
            for axis in perm:vertex=vertex.copy();vertex[axis]+=1;path.append(vertex)
            ids=np.array([node(*v)for v in path])
            if np.linalg.det((X[ids[1:]]-X[ids[0]]).T)<0:ids[[1,2]]=ids[[2,1]]
            tets.append(ids)
    tets=np.array(tets);counts={}
    for tet in tets:
        for face in itertools.combinations(tet,3):
            key=tuple(sorted(face));counts[key]=counts.get(key,0)+1
    faces=np.array([f for f,count in counts.items()if count==1]);rest,elements=enrich(X,tets)
    edges=np.unique(np.sort(tets[:,np.asarray(EDGES)].reshape(-1,2),axis=1),axis=0);codes=edges[:,0]*len(X)+edges[:,1]
    faceedges=np.stack([faces[:,[0,1]],faces[:,[0,2]],faces[:,[1,2]]],axis=1);facecodes=faceedges[:,:,0]*len(X)+faceedges[:,:,1]
    faces6=np.concatenate([faces,len(X)+np.searchsorted(codes,facecodes)],axis=1)
    fixed=np.flatnonzero((rest[:,0]==0)|(rest[:,0]==.36));free=np.flatnonzero(np.repeat((rest[:,0]!=0)&(rest[:,0]!=.36),3))
    return dict(corner_X=X,corner_tets=tets,corner_faces=faces,X=rest,elements=elements,faces=faces6,edge_codes=codes,fixed=fixed,free=free)


def geometry(n):
    t=_topology(n);return tuple(t[k].copy()for k in('X','elements','faces','fixed','free'))


def _query(n,points):
    p=np.asarray(points,float);t=_topology(n);limit=np.array([.36,.08,.08]);tol=64*np.finfo(float).eps
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()or np.any(p < -tol)or np.any(p>limit+tol):raise ValueError('Physicalpoints insideblock')
    scale=limit/[4*n,n,n];x=np.clip(p,0,limit)/scale;cell=np.minimum(np.floor(x).astype(int),[4*n-1,n-1,n-1]);f=x-cell;perm=np.argsort(-f,axis=1,kind='stable');ordered=np.take_along_axis(f,perm,axis=1);bary=np.column_stack([1-ordered[:,0],ordered[:,0]-ordered[:,1],ordered[:,1]-ordered[:,2],ordered[:,2]])
    vertices=[cell.copy()];current=cell.copy()
    for k in range(3):current=current.copy();current[np.arange(len(p)),perm[:,k]]+=1;vertices.append(current)
    v=np.stack(vertices,axis=1);corners=v[:,:,0]*(n+1)**2+v[:,:,1]*(n+1)+v[:,:,2]
    edge=np.sort(corners[:,np.asarray(EDGES)],axis=2);codes=edge[:,:,0]*len(t['corner_X'])+edge[:,:,1];mid=len(t['corner_X'])+np.searchsorted(t['edge_codes'],codes)
    return np.concatenate([corners,mid],axis=1),bary


def field_queries(n,u,points):
    ids,bary=_query(n,points);u=np.asarray(u,float)
    if u.shape[-2:]!=_topology(n)['X'].shape or not np.isfinite(u).all():raise ValueError('FiniteP2field')
    return np.einsum('qa,...qac->...qc',shapes(bary),u[...,ids,:])


def point_query_matrix(forward,points):
    ids,bary=_query(forward.n,points);N=shapes(bary);q=len(ids);globaldof=(ids[:,:,None]*3+np.arange(3)).reshape(q,-1)
    row=np.broadcast_to(np.arange(3*q).reshape(q,1,3),(q,10,3)).ravel();col=globaldof.ravel();value=np.broadcast_to(N[:,:,None],(q,10,3)).ravel();inverse=np.full(forward.X.size,-1,int);inverse[forward.free]=np.arange(len(forward.free));freecol=inverse[col];keep=freecol>=0
    return sparse.coo_matrix((value[keep],(row[keep],freecol[keep])),shape=(3*q,len(forward.free))).tocsr()


def bulk_query_blocks(n=8,*,order=24,tet_chunk=128):
    t=_topology(n);X=t['corner_X'];tet=t['corner_tets'];bary,w=volume_rule(order);vol=np.linalg.det((X[tet[:,1:]]-X[tet[:,:1]]).transpose(0,2,1))/6
    for first in range(0,len(tet),tet_chunk):yield np.einsum('qa,tai->tqi',bary,X[tet[first:first+tet_chunk]]).reshape(-1,3),(vol[first:first+tet_chunk,None]*w[None]).ravel()


def make_forward(n,identity):
    t=_topology(n);f=object.__new__(core.SparseStaticForward)
    f.n=n;f.X=t['X'].copy();f.rest=f.X;f.elements=t['elements'].copy();f.tets=f.elements;f.tets10=f.elements;f.faces=t['faces'].copy();f.fixed=t['fixed'].copy();f.free=t['free'].copy();f.corner_X=t['corner_X'].copy();f.corner_tets=t['corner_tets'].copy();f.quadrature_order=24
    f.Kunit,f.linear_gradients,f.volumes,f.rows,f.cols=core._bulk_assembly(f.X,f.elements,f.corner_X,f.corner_tets,24)
    f.W_scalar,f.support_quadrature=loads.consistent_fixture(f.X,f.faces,identity,order=16)
    f.W=sparse.kron(f.W_scalar,sparse.eye(3),format='csr');f.weights=np.asarray(f.W_scalar.sum(axis=1)).ravel();f.Kunit_free=f.Kunit[f.free][:,f.free].tocsc();f.W_free=f.W[f.free][:,f.free].tocsc()
    return f


def load_catalog(n,*,order=16):
    # Same Gaussian-v02 formula/arithmetic; geometry supplied explicitly rather
    # than replacing any frozen module global or bypassing its n validation.
    X,_,faces,_,_=geometry(n);answer={};diagnostics={}
    for spec in loads.load_specs():
        c=np.asarray(spec['center_m'],float);d=np.asarray(spec['direction'],float);F=float(spec['force_N']);G,centroid=loads.gaussian_integrals(c[:2]);b,w=core.triangle_rule(order);N=core.triangle_shapes(b)
        selected=faces[np.all(abs(X[faces[:,:3],2]-.08)<1e-14,axis=1)];vertices=X[selected[:,:3]]
        area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2;q=np.einsum('qa,fai->fqi',b,vertices);profile=np.exp(-.5*np.sum(((q[:,:,:2]-c[:2])/np.asarray(loads.LOAD_STD_M))**2,axis=2))
        weights=np.zeros(len(X));local=np.einsum('f,q,fq,qa->fa',area,w,profile,N)/G;np.add.at(weights,selected.ravel(),local.ravel())
        force=F*weights[:,None]*d;moment=np.cross(X-loads.ORIGIN_M,force).sum(0);expected=np.cross(centroid-loads.ORIGIN_M,F*d);first=X.T@weights
        answer[spec['id']]=force
        diagnostics[spec['id']]=dict(id=spec['id'],order=order,load_version='whole-top-Gaussian-v02',top_area_m2=.36*.08,authored_center_m=c.tolist(),std_m=list(loads.LOAD_STD_M),analytic_normalizer_m2=G,canonical_truncated_centroid_m=centroid.tolist(),integrated_first_moment_m=first.tolist(),first_moment_error_m=float(np.max(abs(first-centroid))),integrated_weight_mass=float(weights.sum()),normalization_error=abs(float(weights.sum())-1),net_force_N=force.sum(0).tolist(),net_force_error_N=float(np.max(abs(force.sum(0)-F*d))),moment_Nm=moment.tolist(),analytic_moment_Nm=expected.tolist(),moment_error_Nm=float(np.max(abs(moment-expected))),equal_candidate_continuum_traction_norm_N=F,signed_T6_coefficients=True,world_dead_load=True,renormalized=False)
    return answer,diagnostics


def common_rms(fields,*,finest=12):
    sums=np.zeros((len(fields),len(fields)));volume=0.
    for points,w in bulk_query_blocks(finest):
        values=[field_queries(n,u,points) for n,u in fields];volume+=w.sum()
        for i in range(len(fields)):
            for j in range(i):sums[i,j]+=np.sum(w[:,None]*(values[i]-values[j])**2)
    return np.sqrt(sums/(3*volume))


def deformation(forward,u,*,finest=12):
    points=[];weights=[];values=[]
    for q,w in bulk_query_blocks(finest):points.append(q);weights.append(w);values.append(field_queries(forward.n,u,q))
    result=weighted_rigid_metrics(np.concatenate(points),np.concatenate(values),np.concatenate(weights))
    samples=core.strain_metrics(forward,u)
    result.update({k:float(np.asarray(samples[k]).reshape(-1)[0]) for k in ('minimum_J','maximum_absolute_symmetric_strain_eigenvalue')})
    result.update(checked_nodes_and24and80=True,whole_element_J_certified=False,common_finest_bulk_n=finest)
    return result
