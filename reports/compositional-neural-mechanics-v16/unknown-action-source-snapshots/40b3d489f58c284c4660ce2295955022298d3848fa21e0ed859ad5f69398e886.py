"""Separate P2 compatible Neo-Hookean block, affine rest geometry, sparse tangent.

Positive Duffy quadrature; consistent signed nodal body loads are integration
weights, never negative physical particle masses. No clipping or stiffness repair.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from numpy.polynomial.legendre import leggauss
from experiments.boundary_block import block_mesh
from experiments.campaign_static_uniaxial_bar import PrismaticBar,lame,StaticFailure
from experiments.campaign_static_supported_block import SIZE,support_nodes
from experiments.diagnose_nonlinear_physical_probes import interpolate
EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))


def shapes(bary):
    b=np.asarray(bary);return np.concatenate([b*(2*b-1),np.stack([4*b[...,i]*b[...,j] for i,j in EDGES],-1)],-1)


def shape_gradients(bary,linear_gradients):
    b=np.asarray(bary);g=np.asarray(linear_gradients)
    return np.concatenate([(4*b[...,None]-1)*g,np.stack([4*(b[...,i,None]*g[...,j,:]+b[...,j,None]*g[...,i,:]) for i,j in EDGES],-2)],-2)


def duffy(order):
    if order not in (3,4):raise ValueError('Declared positive quadrature orders3 or4 only')
    z,w=leggauss(order);z=(z+1)/2;w=w/2;b=[];weights=[]
    for i,a in enumerate(z):
        for j,c in enumerate(z):
            for k,d in enumerate(z):
                r=a;s=(1-a)*c;t=(1-a)*(1-c)*d
                b.append([1-r-s-t,r,s,t]);weights.append(6*w[i]*w[j]*w[k]*(1-a)**2*(1-c))
    return np.asarray(b),np.asarray(weights)


def enrich(X,tets):
    edge,inverse=np.unique(np.sort(tets[:,np.asarray(EDGES)].reshape(-1,2),axis=1),axis=0,return_inverse=True)
    T=np.concatenate([tets,len(X)+inverse.reshape(len(tets),len(EDGES))],1)
    return np.concatenate([X,X[edge].mean(1)]),T


class QuadraticBlock(PrismaticBar):
    def __init__(self,cells=(4,2,2),*,young=300000.,poisson=.35,density=625.,gravity=9.81,traction=0.,order=3,boundary='left_face',memory_limit_bytes=8*1024**3,linear_memory_bytes_per_dof=20000):
        lame(young,poisson)
        if density<=0 or not np.isfinite([density,gravity,traction]).all():raise ValueError('Finite known loads and positive density required')
        X,self.tets,self.faces=block_mesh(SIZE,cells,(SIZE[0]/2,0,0));self.reference_local,self.elements=enrich(X,self.tets);self.rest=self.reference_local
        corners=X[self.tets];Dm=np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2);det=np.linalg.det(Dm)
        if np.any(det<=0):raise ValueError('Positive rest geometry required')
        inv=np.linalg.inv(Dm);self.linear_gradients=np.concatenate([-inv.sum(1)[:,None],inv],1);self.volumes=det/6
        self.young,self.poisson,self.density=float(young),float(poisson),float(density);self.minimum_J=.1;self.order=order
        self.bary,self.weights=duffy(order);self.N=shapes(self.bary);self.check_bary=np.concatenate([np.eye(4),np.array([(np.eye(4)[i]+np.eye(4)[j])/2 for i,j in EDGES])])
        dofs=(self.elements[:,:,None]*3+np.arange(3)).reshape(-1,30)
        self.rows=np.repeat(dofs,30,1).ravel().astype(np.int32);self.cols=np.tile(dofs,(1,30)).ravel().astype(np.int32)
        # Signed P2 integration coefficients; physical density remains positive.
        self.body_integral_weights=np.zeros(len(self.rest));local=self.volumes[:,None]*np.einsum('q,qa->a',self.weights,self.N)
        np.add.at(self.body_integral_weights,self.elements,local)
        self.load=np.zeros_like(self.rest);self.load[:,2]=-gravity*density*self.body_integral_weights
        right_tets=np.sum(abs(X[self.tets,0]-SIZE[0])<1e-13,axis=1)>=3
        edge_index={tuple(sorted(map(int,t[[i,j]]))):int(e[4+k]) for t,e in zip(self.tets[right_tets],self.elements[right_tets]) for k,(i,j) in enumerate(EDGES)}
        self.right_surface_weights=np.zeros(len(self.rest))
        for f in self.faces[np.all(abs(X[self.faces,0]-SIZE[0])<1e-13,axis=1)]:
            a=.5*np.linalg.norm(np.cross(X[f[1]]-X[f[0]],X[f[2]]-X[f[0]]))
            # Exact integral of each quadratic triangle vertex function is0;
            # each edge function integrates area/3 (same as positive Duffy rule).
            for i,j in ((0,1),(0,2),(1,2)):self.right_surface_weights[edge_index[tuple(sorted(map(int,f[[i,j]])))]]+=a/3
        self.load[:,0]+=traction*self.right_surface_weights/(SIZE[1]*SIZE[2]);self.force=float(np.linalg.norm(self.load.sum(0)))
        self.mask=np.repeat(support_nodes(self.rest,boundary)[:,None],3,1);self.free=np.flatnonzero(~self.mask.ravel())
        self.rotation=np.eye(3);self.translation=np.zeros(3)
        # Conservative assembly/factor budget estimate, including sparse fill.
        self.estimated_peak_memory_bytes=int(len(self.elements)*900*32+len(self.rest)*3*linear_memory_bytes_per_dof)
        if self.estimated_peak_memory_bytes>memory_limit_bytes:raise MemoryError('Prospective P2 memory limit exceeded')

    def elastic(self,positions,tangent=False):
        x=np.asarray(positions);u=x-self.rest
        if x.shape!=self.rest.shape or not np.isfinite(x).all():raise ValueError('Finite P2 positions required')
        mu,lam=lame(self.young,self.poisson);energy=0.;gradient=np.zeros_like(x);Hvalues=np.zeros((len(self.elements),30,30)) if tangent else None;allJ=[]
        eye=np.eye(3)
        for start in range(0,len(self.elements),256):
            sl=slice(start,start+256);T=self.elements[sl];ue=u[T];g0=self.linear_gradients[sl];V=self.volumes[sl];ge=np.zeros_like(ue);He=np.zeros((len(T),10,3,10,3)) if tangent else None
            for b in self.check_bary:
                checkg=shape_gradients(b[None,:],g0);checkJ=np.linalg.det(eye+np.einsum('eai,eaJ->eiJ',ue,checkg))
                if not np.isfinite(checkJ).all() or np.any(checkJ<=self.minimum_J):raise ValueError('Nodal Jacobian guard failed; no repair')
                allJ.append(checkJ)
            for b,w in zip(self.bary,self.weights):
                g=shape_gradients(b[None,:],g0);F=eye+np.einsum('eai,eaJ->eiJ',ue,g);J=np.linalg.det(F)
                if not np.isfinite(J).all() or np.any(J<=self.minimum_J):raise ValueError('Nonpositive or subguard quadrature Jacobian; no repair')
                allJ.append(J);G=F-eye;trace=np.trace(G,axis1=-2,axis2=-1);second=.5*(trace**2-np.einsum('eij,eji->e',G,G));detG=np.linalg.det(G);jm=trace+second+detG;log=np.log1p(jm)
                invT=np.swapaxes(np.linalg.inv(F),-1,-2)
                small=sum((-1.)**k*jm**k/k for k in range(2,10));xm=np.where(abs(jm)<1e-3,small,jm-log)
                density=mu*(.5*np.sum(G*G,(1,2))-second-detG+xm)+.5*lam*log**2
                P=mu*G+(lam*log-mu)[:,None,None]*(invT-eye)+lam*log[:,None,None]*eye;v=V*w
                energy+=float(v@density);ge+=np.einsum('e,eiJ,eaJ->eai',v,P,g)
                if tangent:
                    A=mu*np.einsum('ik,JL->iJkL',eye,eye)[None]+(mu-lam*log)[:,None,None,None,None]*np.einsum('eiL,ekJ->eiJkL',invT,invT)+lam*np.einsum('eiJ,ekL->eiJkL',invT,invT)
                    He+=np.einsum('e,eaJ,eiJkL,ebL->eaibk',v,g,A,g,optimize=True)
            np.add.at(gradient,T,ge)
            if tangent:Hvalues[sl]=He.reshape(-1,30,30)
        H=None if not tangent else sparse.coo_matrix((Hvalues.ravel(),(self.rows,self.cols)),shape=(x.size,x.size)).tocsr()
        return energy,gradient,H,np.concatenate(allJ)

    def evaluate(self,u,tangent=False):
        U,g,H,J=self.elastic(self.rest+u,tangent);work=float(np.sum(self.load*u))
        return dict(potential_J=U-work,elastic_J=U,spring_J=0.,work_J=work,residual=g-self.load,tangent=H,J=J,positions=self.rest+u)

    def solve(self,initial=None,maximum_iterations=40,force_tolerance_N=1e-11):
        u=np.zeros_like(self.rest) if initial is None else np.asarray(initial,dtype=float).copy()
        if u.shape!=self.rest.shape or not np.isfinite(u).all() or np.any(u[self.mask]!=0):raise ValueError('Initial state must satisfy exact local symmetry constraints')
        if not np.isfinite(force_tolerance_N) or force_tolerance_N<=0:raise ValueError('Positive original-force tolerance required')
        tolerance=float(force_tolerance_N);history=[]
        for iteration in range(maximum_iterations+1):
            state=self.evaluate(u,True);r=state['residual'].ravel()[self.free];norm=float(np.linalg.norm(r))
            history.append(dict(iteration=iteration,residual_N=norm,potential_J=state['potential_J'],minimum_J=float(state['J'].min())))
            if norm<=tolerance:
                state.update(displacement_local_m=u,iterations=iteration,history=history,tolerance_N=tolerance)
                return state
            if iteration==maximum_iterations:raise StaticFailure('Static original-force residual exceeds cap',history)
            matrix=state['tangent'][self.free][:,self.free].tocsc()
            try:direction=splu(matrix).solve(-r)
            except RuntimeError as error:raise StaticFailure('Unmodified static tangent factorization failed',history) from error
            slope=float(r@direction)
            if not np.isfinite(direction).all() or not np.isfinite(slope) or slope>=0:raise StaticFailure('Unmodified Newton direction not descent',history)
            for backtrack in range(40):
                alpha=2.**(-backtrack);trial=u.copy();trial.ravel()[self.free]+=alpha*direction
                try:candidate=self.evaluate(trial)
                except ValueError:continue
                # Near the float64 energy floor, also require decreasing FORCE
                # residual; this is acceptance arithmetic, not a changed potential.
                roundoff=128*np.finfo(float).eps*max(abs(state['potential_J']),self.young*self.volumes.sum(),1e-30)
                if candidate['potential_J']<=state['potential_J']+1e-4*alpha*slope or (abs(candidate['potential_J']-state['potential_J'])<=roundoff and np.linalg.norm(candidate['residual'].ravel()[self.free])<norm):
                    u=trial;history[-1]['alpha']=alpha;break
            else:raise StaticFailure('PositiveJ static Armijo exhausted',history)
        raise AssertionError('Unreachable static solve')

    def query(self,positions,points):
        _,e,b=interpolate(self.rest[:self.tets.max()+1],self.tets,self.rest[None,:self.tets.max()+1],points)
        return np.einsum('pa,pai->pi',shapes(b),np.asarray(positions)[self.elements[e]])

    def diagnostics(self,answer):
        q=answer['positions'];u=q-self.rest;mu,lam=lame(self.young,self.poisson);strain=[];qweights=[];X=[];Y=[]
        for b,w in zip(self.bary,self.weights):
            g=shape_gradients(b[None,:],self.linear_gradients);F=np.eye(3)+np.einsum('eai,eaJ->eiJ',u[self.elements],g)
            strain.append(np.linalg.svd(F,compute_uv=False)-1);qweights.append(w*self.volumes);X.append(np.einsum('a,eai->ei',shapes(b),self.rest[self.elements]));Y.append(np.einsum('a,eai->ei',shapes(b),q[self.elements]))
        nodal_strain=[];vertex_Gnorm=[]
        for b in self.check_bary:
            g=shape_gradients(b[None,:],self.linear_gradients);F=np.eye(3)+np.einsum('eai,eaJ->eiJ',u[self.elements],g);nodal_strain.append(np.linalg.svd(F,compute_uv=False)-1)
            if len(vertex_Gnorm)<4:vertex_Gnorm.append(np.linalg.norm(F-np.eye(3),ord=2,axis=(1,2)))
        s=np.concatenate(strain);weights=np.concatenate(qweights);X=np.concatenate(X);Y=np.concatenate(Y);weights/=weights.sum();cx=weights@X;cy=weights@Y
        a=X-cx;b=Y-cy;U,_,V=np.linalg.svd((a*weights[:,None]).T@b);Q=U@np.diag([1,1,np.linalg.det(U@V)])@V
        e=Y-((X-cx)@Q+cy);bulk=np.sqrt(np.sum(weights[:,None]*e**2)/3);peak=np.max(abs(s),1);idx=np.argsort(peak,kind='stable');p99=peak[idx[np.searchsorted(np.cumsum(weights[idx]),.99)]]
        reaction=np.where(self.mask,answer['residual'],0.);force=(reaction+self.load).sum(0);moment=np.cross(q,reaction+self.load).sum(0)
        maxG=float(np.max(vertex_Gnorm));certified_lower=(1-maxG)**3 if maxG<1 else None
        return {'global_affine_F_positive_J_sufficient_lower_bound':certified_lower,'maximum_vertex_displacement_gradient_spectral_norm':maxG,'minimum_J':float(answer['J'].min()),'maximum_absolute_principal_stretch_strain':float(max(peak.max(),np.max(abs(np.concatenate(nodal_strain))))),'volume_weighted99th_percentile_max_abs_principal_strain':float(p99),'volume_weighted_principal_strain_RMS':float(np.sqrt(np.sum(weights[:,None]*s*s)/3)),'bulk_nonrigid_coordinate_RMS_m':float(bulk),'maximum_displacement_m':float(np.linalg.norm(u,axis=1).max()),'original_free_residual_N':float(np.linalg.norm(answer['residual'].ravel()[self.free])),'residual_tolerance_N':answer['tolerance_N'],'force_balance_defect_N':float(np.linalg.norm(force)),'actual_moment_balance_defect_Nm':float(np.linalg.norm(moment)),'maximum_constraint_error_m':float(abs(u[self.mask]).max()),'iterations':answer['iterations'],'estimated_peak_memory_bytes':self.estimated_peak_memory_bytes,'physical_solid_mass_kg':float(self.density*self.volumes.sum())}
