"""Matched external surface tractions and consistent Gaussian fixture reference.

Source-only API; no automatic experiments or inference. Sparse small-strain T4
statics are not a nonlinear or dynamic reference. All coordinates/forces use SI.
"""
import itertools
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import erf

LENGTH=.36
LOWER=np.log([2000.,5.]);UPPER=np.log([8000.,30.])
SUPPORT_CENTER=np.array([.24,.08,.04]);SUPPORT_WIDTH=.025
PRODUCTION_QUADRATURE=12
GATES=dict(minimum_nonrigid_RMS_m=100e-6,minimum_nonrigid_fraction=.3,maximum_linear_strain=.05,minimum_J=.8,
           free_residual_N=1e-10,spatial_order_floor=1.,maximum_conditional_spatial_indicator_m=20e-6,
           support_normalization_relative_error=1e-10,support_12_vs_16_relative_Frobenius=1e-10,
           support_8_vs_12_relative_Frobenius=1e-8,load_net_force_error_N=1e-12,load_moment_error_Nm=1e-13)


def design():
    return dict(schema='surface-traction-static-v01',n_levels=[2,4,8],node_grids=[[9,3,3],[17,5,5],[33,9,9]],
        dimensions_m=[LENGTH,.08,.08],both_end_faces_hard_clamped=True,density_kg_m3=625.,poisson=.35,
        prior_intrinsic=['logE','logsupportK'],prior_lower=LOWER.tolist(),prior_upper=UPPER.tolist(),
        foundation='W_scalar=integral(g*N_i*N_j)dA/G_exact over six external faces; W=kron(W_scalar,I3); no lumping/renormalization/jitter',
        support_center_m=SUPPORT_CENTER.tolist(),support_width_m=SUPPORT_WIDTH,
        normalization='Exact separable erf integral over rectangular external faces, common to all meshes',
        traction='External z=0/.08 patch x[.135,.225], y[0,.08], sin²(pi*(x-.135)/.09); exact per-face profile integral .0036m²',
        loads=dict(normal_top_Fz_N=-.10,axial_top_Fx_N=.24,axial_bottom_Fx_N=.24),
        load_moment_origin_m=[.18,.04,.04],net_qualification_moments_Nm=[0.,0.,0.],
        quadrature=dict(production=12,diagnostic_orders=[4,8,12,16],triangle='Duffy tensor Gauss-Legendre, positive weights'),
        gates=GATES,spatial_gate='Three levels; p=log2(d_coarse_medium/d_medium_fine)>=1 and d_medium_fine/(2^p-1)<=20um, conditional on continuing order. Medium adequacy requires2*d_medium_fine<=20um.',
        common_queries='Same finest-mesh positive Keast24 bulk quadrature for every field; exact T4 physical-point interpolation',
        physical_cases='All four original E/K prior corners plus intrinsic center, both qualification loads; no filtering or retuning',
        analytic_before_responses='Paired surface axial load cancels eccentric moment. AtE2k peak surface shear strain estimate66.667/(2*740.741)=.045; bulk axial .01875. Beam normal root strain estimate .02637. Support, locking, traction localization and corner singularities can still fail gates.',
        gravity=False,contact=False,scope='Static finite linear reference only; dynamics, local stress convergence, nonlinear validity, posterior calibration and active probing remain separate.',no_automatic_execution=True)


@lru_cache(maxsize=3)
def _geometry_arrays(n):
    if type(n)is not int or n not in(2,4,8):raise ValueError('Prospective n=2,4,8 only')
    X=np.array(list(itertools.product(np.linspace(0,LENGTH,4*n+1),np.linspace(0,.08,n+1),np.linspace(0,.08,n+1))))
    node=lambda i,j,k:i*(n+1)**2+j*(n+1)+k;tets=[]
    for i,j,k in itertools.product(range(4*n),range(n),range(n)):
        for permutation in itertools.permutations(range(3)):
            corner=np.array([i,j,k]);vertices=[corner.copy()]
            for axis in permutation:corner=corner.copy();corner[axis]+=1;vertices.append(corner)
            ids=np.array([node(*v)for v in vertices])
            if np.linalg.det((X[ids[1:]]-X[ids[0]]).T)<0:ids[[1,2]]=ids[[2,1]]
            tets.append(ids)
    tets=np.asarray(tets);counts={}
    for tet in tets:
        for f in itertools.combinations(tet,3):
            key=tuple(sorted(f));counts[key]=counts.get(key,0)+1
    faces=np.array([f for f,count in counts.items()if count==1])
    fixed=np.flatnonzero((X[:,0]==0)|(X[:,0]==LENGTH));free=np.flatnonzero(np.repeat((X[:,0]!=0)&(X[:,0]!=LENGTH),3))
    fixed.setflags(write=False);free.setflags(write=False)
    return X,tets,faces,fixed,free


def geometry(n):
    """Cached topology, with private copies so callers cannot mutate the cache."""
    values=tuple(a.copy() for a in _geometry_arrays(n))
    values[3].setflags(write=False);values[4].setflags(write=False)
    return values


def triangle_rule(order):
    if type(order)is not int or order not in(4,8,12,16):raise ValueError('Declared quadrature order4/8/12/16')
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2;r,s=np.meshgrid(x,x,indexing='ij');wr,ws=np.meshgrid(w,w,indexing='ij')
    bary=np.column_stack([(1-r).ravel(),(r*(1-s)).ravel(),(r*s).ravel()]);weights=(2*r*wr*ws).ravel()
    return bary,weights


def gaussian_surface_normalization():
    upper=np.array([LENGTH,.08,.08]);c=SUPPORT_CENTER;s=SUPPORT_WIDTH
    integrals=s*np.sqrt(np.pi/2)*(erf((upper-c)/(np.sqrt(2)*s))-erf(-c/(np.sqrt(2)*s)))
    return float(sum(sum(np.exp(-.5*((side-c[axis])/s)**2)*np.prod(integrals[np.arange(3)!=axis])for side in(0.,upper[axis]))for axis in range(3)))


def consistent_support(X,faces,*,order=PRODUCTION_QUADRATURE):
    bary,qw=triangle_rule(order);local=[];mass=0.;normalization=gaussian_surface_normalization()
    for first in range(0,len(faces),256):
        vertices=X[faces[first:first+256]];area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2
        points=np.einsum('qa,fai->fqi',bary,vertices);g=np.exp(-.5*np.sum((points-SUPPORT_CENTER)**2,axis=2)/SUPPORT_WIDTH**2)
        local.append(np.einsum('f,q,fq,qa,qb->fab',area,qw,g,bary,bary)/normalization);mass+=float(np.einsum('f,q,fq->',area,qw,g))/normalization
    values=np.concatenate(local);rows=np.repeat(faces,3,axis=1).reshape(-1);cols=np.tile(faces,(1,3)).reshape(-1)
    W=sparse.coo_matrix((values.reshape(-1),(rows,cols)),shape=(len(X),len(X))).tocsr();W.sum_duplicates()
    return W,dict(exact_normalization_m2=normalization,integrated_normalized_mass=mass,normalization_relative_error=abs(mass-1),order=order)


def surface_traction(n,direction,total_force_N,*,face='top',order=PRODUCTION_QUADRATURE):
    X,_,faces,_,_=geometry(n);direction=np.asarray(direction,float)
    if direction.shape!=(3,)or not np.isfinite(direction).all()or not np.isclose(np.linalg.norm(direction),1,rtol=0,atol=1e-13)or not np.isfinite(total_force_N)or total_force_N<0 or face not in('top','bottom'):raise ValueError('Unit direction, nonnegative forceN and top/bottomface')
    z=.08 if face=='top'else 0.;vertices=X[faces];tol=64*np.finfo(float).eps
    active=np.all(abs(vertices[:,:,2]-z)<tol,axis=1)&np.all(vertices[:,:,0]>=.135-tol,axis=1)&np.all(vertices[:,:,0]<=.225+tol,axis=1)
    selected=faces[active];vertices=X[selected];area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2
    bary,qw=triangle_rule(order);points=np.einsum('qa,fai->fqi',bary,vertices);profile=np.sin(np.pi*(points[:,:,0]-.135)/.09)**2
    scalar=np.einsum('f,q,fq,qa->fa',area,qw,profile,bary)/(.09*.08/2);weights=np.zeros(len(X));np.add.at(weights,selected.reshape(-1),scalar.reshape(-1))
    force=weights[:,None]*total_force_N*direction
    return force,dict(face=face,patch_area_m2=float(area.sum()),exact_profile_integral_m2=.0036,integrated_weight_mass=float(weights.sum()),weights=weights,net_force_N=force.sum(0),moment_about_patch_centroid_Nm=np.cross(X-[.18,.04,z],force).sum(0),order=order)


def qualification_loads(n,*,order=PRODUCTION_QUADRATURE):
    normal,_=surface_traction(n,[0,0,-1],.10,order=order);top,_=surface_traction(n,[1,0,0],.24,order=order);bottom,_=surface_traction(n,[1,0,0],.24,face='bottom',order=order)
    return dict(normal=normal,paired_axial=top+bottom)


def sparse_elastic_assembly(X,tets):
    dm=(X[tets[:,1:]]-X[tets[:,:1]]).transpose(0,2,1);det=np.linalg.det(dm)
    if np.any(det<=0):raise ValueError('Positive element winding required')
    volumes=det/6;g=np.linalg.inv(dm);g=np.concatenate([-g.sum(1,keepdims=True),g],axis=1);B=np.zeros((len(tets),6,12))
    for a in range(4):
        dx,dy,dz=g[:,a].T;k=3*a;B[:,0,k]=dx;B[:,1,k+1]=dy;B[:,2,k+2]=dz;B[:,3,k+1]=dz;B[:,3,k+2]=dy;B[:,4,k]=dz;B[:,4,k+2]=dx;B[:,5,k]=dy;B[:,5,k+1]=dx
    nu=.35;mu=1/(2*(1+nu));lam=nu/((1+nu)*(1-2*nu));D=np.zeros((6,6));D[:3,:3]=lam;D[:3,:3]+=2*mu*np.eye(3);D[3:,3:]=mu*np.eye(3)
    local=(B.transpose(0,2,1)@D@B)*volumes[:,None,None];ids=(tets[:,:,None]*3+np.arange(3)).reshape(-1,12)
    rows=np.broadcast_to(ids[:,:,None],local.shape).ravel();cols=np.broadcast_to(ids[:,None,:],local.shape).ravel()
    K=sparse.coo_matrix((local.ravel(),(rows,cols)),shape=(3*len(X),3*len(X))).tocsr();K.sum_duplicates()
    mass=np.zeros(len(X));np.add.at(mass,tets.ravel(),np.broadcast_to(volumes[:,None]*625/4,(len(tets),4)).ravel())
    return K,np.repeat(mass,3),g,volumes


class SparseStaticForward:
    def __init__(self,n=2,*,quadrature_order=PRODUCTION_QUADRATURE):
        self.n=n;self.X,self.tets,self.faces,self.fixed,self.free=geometry(n)
        self.Kunit,self.mass,self.gradients,self.volumes=sparse_elastic_assembly(self.X,self.tets)
        self.W_scalar,self.support_quadrature=consistent_support(self.X,self.faces,order=quadrature_order)
        self.W=sparse.kron(self.W_scalar,sparse.eye(3),format='csr');self.weights=np.asarray(self.W_scalar.sum(axis=1)).ravel()
        self.Kunit_free=self.Kunit[self.free][:,self.free].tocsc();self.W_free=self.W[self.free][:,self.free].tocsc()
    def staticforward(self,theta,load,*,return_diagnostics=False):
        theta=np.asarray(theta,float);load=np.asarray(load,float)
        if theta.ndim!=2 or theta.shape[1]!=2 or not len(theta)or not np.isfinite(theta).all()or np.any(theta<=0):raise ValueError('Positivephysical[E,K]array[B,2]')
        if load.ndim==2:load=load[None]
        if load.ndim!=3 or load.shape[1:]!=self.X.shape or not len(load)or not np.isfinite(load).all():raise ValueError('Force[R,N,3] innewtons')
        outputs=[];diagnostics=[];rhs=load.reshape(len(load),-1).T
        for E,K in theta:
            A=E*self.Kunit_free+K*self.W_free;u=np.zeros((3*len(self.X),len(load)));u[self.free]=splu(A).solve(rhs[self.free]);residual=(E*self.Kunit+K*self.W)@u-rhs
            error=float(np.max(abs(residual[self.free])))
            if not np.isfinite(u).all()or error>GATES['free_residual_N']:raise FloatingPointError('Static physical residual gate failed')
            outputs.append(u.T.reshape(len(load),len(self.X),3));diagnostics.append(dict(maximum_free_residual_N=error,exact_clamp_zeros=bool(np.all(u.reshape(len(self.X),3,-1)[self.fixed]==0)),support_quadrature=self.support_quadrature))
        result=np.asarray(outputs)
        return (result,diagnostics)if return_diagnostics else result


def staticforward(theta,load,*,n=2,return_diagnostics=False):return SparseStaticForward(n).staticforward(theta,load,return_diagnostics=return_diagnostics)


def physical(unit):
    unit=np.asarray(unit,float)
    if unit.ndim!=2 or unit.shape[1]!=2 or not len(unit)or not np.isfinite(unit).all()or np.any((unit<0)|(unit>1)):raise ValueError('Unit[B,2] insideprior')
    return np.exp(LOWER+unit*(UPPER-LOWER))


def query_map(n,points):
    if type(n)is not int or n not in(2,4,8):raise ValueError('Prospective n=2,4,8 only')
    points=np.asarray(points,float);scale=np.array([LENGTH/(4*n),.08/n,.08/n]);limit=np.array([LENGTH,.08,.08]);tol=64*np.finfo(float).eps
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all()or np.any(points < -tol)or np.any(points>limit+tol):raise ValueError('Physicalquerypoints insideblock')
    normalized=np.clip(points,0,limit)/scale;cell=np.minimum(np.floor(normalized).astype(int),[4*n-1,n-1,n-1]);fraction=normalized-cell;permutation=np.argsort(-fraction,axis=1,kind='stable');ordered=np.take_along_axis(fraction,permutation,axis=1)
    bary=np.column_stack([1-ordered[:,0],ordered[:,0]-ordered[:,1],ordered[:,1]-ordered[:,2],ordered[:,2]])
    vertices=[cell.copy()];current=cell.copy()
    for step in range(3):current=current.copy();current[np.arange(len(points)),permutation[:,step]]+=1;vertices.append(current)
    vertices=np.stack(vertices,axis=1);ids=vertices[:,:,0]*(n+1)**2+vertices[:,:,1]*(n+1)+vertices[:,:,2]
    return ids,bary


def field_queries(n,u,points):
    u=np.asarray(u,float)
    if type(n)is not int or n not in(2,4,8):raise ValueError('Prospective n=2,4,8 only')
    if u.shape[-2:]!=((4*n+1)*(n+1)**2,3) or not np.isfinite(u).all():raise ValueError('Finite nodalfield[...,N,3]')
    ids,bary=query_map(n,points);return np.einsum('qa,...qac->...qc',bary,u[...,ids,:])


def bulk_query_blocks(n=8,*,tet_chunk=256):
    from experiments.campaign_p2_keast_quadrature import positive_keast24
    X,tets,_,_,_=geometry(n);bary,qw=positive_keast24();vol=np.linalg.det((X[tets[:,1:]]-X[tets[:,:1]]).transpose(0,2,1))/6
    for first in range(0,len(tets),tet_chunk):
        ids=tets[first:first+tet_chunk];points=np.einsum('qa,tai->tqi',bary,X[ids]).reshape(-1,3);weights=(vol[first:first+tet_chunk,None]*qw[None]).ravel();yield points,weights


def displacement_gradient(forward,u):
    u=np.asarray(u,float)
    if u.shape[-2:]!=forward.X.shape or not np.isfinite(u).all():raise ValueError('Finitefield[...,N,3]')
    return np.einsum('btnc,tni->btci',u.reshape(-1,len(forward.X),3)[:,forward.tets],forward.gradients)
