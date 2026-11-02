"""Consistent surface-Robin P1/P2 fixtures; unchanged log-J Neo-Hookean material."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from experiments.campaign_static_uniaxial_bar import StaticFailure
from numpy.polynomial.legendre import leggauss
from scripts.phystwin.active_support_reference import ActiveBlock
from experiments.campaign_static_p2_block import QuadraticBlock,EDGES,shapes,shape_gradients
from experiments.campaign_static_uniaxial_bar import lame


def surface(model,side,lower=False,degree=1):
    faces=model.faces[np.all(abs(model.rest[model.faces,0]-side)<1e-13,axis=1)]
    if lower:faces=faces[np.all(model.rest[faces,2]<=1e-13,axis=1)]
    points=model.rest[faces];area=.5*np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)
    if degree==2:
        edges=getattr(model,'_surface_edges',None)
        if edges is None:
            x=model.rest[model.tets,0];boundary=(np.sum(abs(x)<1e-13,axis=1)>=3)|(np.sum(abs(x-.24)<1e-13,axis=1)>=3)
            edges={tuple(sorted(map(int,t[[i,j]]))):int(e[4+k]) for t,e in zip(model.tets[boundary],model.elements[boundary]) for k,(i,j) in enumerate(EDGES)}
            model._surface_edges=edges
        nodes=np.c_[faces,[[edges[tuple(sorted(map(int,f[[i,j]])))] for i,j in [(0,1),(0,2),(1,2)]] for f in faces]]
    else:nodes=faces
    z,w=leggauss(3);z=(z+1)/2;w=w/2;N=[];weights=[]
    for i,a in enumerate(z):
        for j,b in enumerate(z):
            L=np.array([1-a-(1-a)*b,a,(1-a)*b])
            N.append(L if degree==1 else np.r_[L*(2*L-1),4*L[0]*L[1],4*L[0]*L[2],4*L[1]*L[2]])
            weights.append(2*w[i]*w[j]*(1-a))
    N=np.array(N);weights=np.array(weights)
    local=np.einsum('q,qa,qb->ab',weights,N,N);integral=weights@N
    rows=np.broadcast_to(nodes[:,:,None],(len(nodes),len(integral),len(integral)))
    cols=np.broadcast_to(nodes[:,None,:],rows.shape)
    M=sparse.coo_matrix(((area[:,None,None]*local).ravel(),(rows.ravel(),cols.ravel())),shape=(len(model.rest),len(model.rest))).tocsr()
    load=np.zeros(len(model.rest));np.add.at(load,nodes,area[:,None]*integral)
    return M,load,float(area.sum())


class ConsistentFixture:
    def clear_linear_cache(self):
        self._gpu_multigrid=None
        self._cpu_multigrid=None

    def solve(self,initial=None,maximum_iterations=40,force_tolerance_N=1e-10):
        # Same unmodified Newton/Armijo objective as QuadraticBlock.solve;
        # symmetric-graph ordering reduces sparse fill, without regularization.
        u=np.zeros_like(self.rest) if initial is None else np.asarray(initial).copy()
        if u.shape!=self.rest.shape or not np.isfinite(u).all():raise ValueError('Finite initial displacement required')
        history=[];state=self.evaluate(u)
        for iteration in range(maximum_iterations+1):
            r=state['residual'].ravel();norm=float(np.linalg.norm(r))
            if getattr(self,'progress',False):print(f'NEWTON iteration={iteration} residual_N={norm:.6e}',flush=True)
            history.append(dict(iteration=iteration,residual_N=norm,potential_J=state['potential_J'],minimum_J=float(state['J'].min())))
            if norm<=force_tolerance_N:
                state.update(displacement_local_m=u,iterations=iteration,history=history,tolerance_N=force_tolerance_N)
                return state
            if iteration==maximum_iterations:raise StaticFailure('Consistent-fixture Newton cap',history)
            if getattr(self,'_gpu_multigrid',None) is not None:self._gpu_multigrid.prepare_assembly()
            started=time.monotonic();H=self.evaluate(u,True)['tangent'].tocsr()
            if getattr(self,'progress',False):print(f'TANGENT seconds={time.monotonic()-started:.2f}',flush=True)
            started=time.monotonic()
            if getattr(self,'linear_solver','superlu')=='cpu_multigrid':
                import pyamg
                from scipy.sparse.linalg import cg
                if getattr(self,'_cpu_multigrid',None) is None:
                    x=self.rest+u;x=x-x.mean(0);B=np.zeros((len(x),3,6));B[:,:,:3]=np.eye(3)
                    for j in range(3):B[:,:,3+j]=np.cross(np.eye(3)[j],x)
                    B=B.reshape(-1,6);B/=np.linalg.norm(B,axis=0)
                    self._cpu_multigrid=pyamg.smoothed_aggregation_solver(H.tobsr(blocksize=(3,3)),B=B,symmetry='symmetric',max_coarse=100)
                direction,info=cg(H,-r,M=self._cpu_multigrid.aspreconditioner(),rtol=1e-9,atol=1e-13,maxiter=1000)
                if info:raise StaticFailure(f'CPU multigrid CG failed: {info}',history)
            elif getattr(self,'linear_solver','superlu')=='gpu_multigrid':
                from scripts.phystwin.gpu_multigrid import GPUMultigrid
                if getattr(self,'_gpu_multigrid',None) is None:self._gpu_multigrid=GPUMultigrid(H,self.rest+u,self.engine.device)
                direction,steps=self._gpu_multigrid.solve(H,-r)
                if getattr(self,'progress',False):print('GPU_PCG iterations',steps,flush=True)
            elif getattr(self,'linear_solver','superlu')=='pardiso':
                from pypardiso.scipy_aliases import spsolve,pypardiso_solver
                try:direction=spsolve(H,-r,factorize=False)
                finally:pypardiso_solver.free_memory(everything=True)
            else:direction=splu(H.tocsc(),permc_spec='MMD_AT_PLUS_A').solve(-r)
            linear=np.linalg.norm(H@direction+r)
            if getattr(self,'progress',False):print(f'LINEAR seconds={time.monotonic()-started:.2f} residual_N={linear:.6e}',flush=True)
            if linear>max(1e-12,norm*1e-7):raise StaticFailure('Original linear residual failed',history)
            slope=float(r@direction)
            if not np.isfinite(direction).all() or slope>=0:raise StaticFailure('Newton direction not descent',history)
            for backtrack in range(40):
                alpha=2.**(-backtrack);trial=u+alpha*direction.reshape(u.shape)
                try:candidate=self.evaluate(trial)
                except ValueError:continue
                floor=128*np.finfo(float).eps*max(abs(state['potential_J']),self.young*self.volumes.sum(),1e-30)
                if candidate['potential_J']<=state['potential_J']+1e-4*alpha*slope or (abs(candidate['potential_J']-state['potential_J'])<=floor and np.linalg.norm(candidate['residual'])<norm):
                    u=trial;state=candidate;history[-1]['alpha']=alpha;break
            else:raise StaticFailure('Consistent-fixture Armijo cap',history)
        raise AssertionError('Unreachable')
    set_action=ActiveBlock.set_action
    def attach(self,layout,K,degree):
        self.layout=layout
        self.degree=degree;self.K=K;self.mask[:]=False;self.free=np.arange(self.rest.size)
        M,_,self.attachment_area=surface(self,0. if layout.startswith('left') else .24,layout.endswith('lower'),degree)
        self.fixture_scalar=K*M/self.attachment_area
        self.spring_matrix=sparse.kron(self.fixture_scalar,sparse.eye(3),format='csr')
        _,self.left,A=surface(self,0.,False,degree);self.left/=A
        _,self.right,A=surface(self,.24,False,degree);self.right/=A
        self.load=np.zeros_like(self.rest);self.force=10.
    def evaluate(self,u,tangent=False):
        U,g,H,J=self.elastic(self.rest+u,tangent);spring_force=self.fixture_scalar@u
        spring=.5*np.sum(u*spring_force);work=np.sum(self.load*u)
        return dict(potential_J=float(U+spring-work),elastic_J=float(U),spring_J=float(spring),work_J=float(work),
            residual=g+spring_force-self.load,tangent=H+self.spring_matrix if tangent else None,J=J,positions=self.rest+u)


class P1Fixture(ConsistentFixture,ActiveBlock):
    def __init__(self,n,E,K,layout):
        super().__init__(n,E,K,layout);self.n=n;self.attach(layout,K,1)


class P2Fixture(ConsistentFixture,QuadraticBlock):
    def __init__(self,n,E,K,layout,order=3,device=None,grading=False,memory_limit_bytes=24*1024**3,z_refinement=1,iterative=False):
        if not isinstance(z_refinement,(int,np.integer)) or z_refinement<1:raise ValueError('Positive integer z refinement required')
        if layout.endswith('lower') and (n*z_refinement)%2:raise ValueError('Half-face patch must align with z=0')
        self.cells=np.array([4*n,n,z_refinement*n],dtype=int)
        super().__init__(cells=tuple(self.cells),young=E,poisson=.35,density=625.,gravity=0.,traction=0.,order=order,memory_limit_bytes=memory_limit_bytes,linear_memory_bytes_per_dof=2000 if iterative else 20000)
        self.n=n
        if grading:
            axes=[np.linspace(0,.24,4*n+1),np.linspace(-.04,.04,n+1),np.linspace(-.04,.04,z_refinement*n+1)]
            t=axes[0]/.24
            power=2. if grading in ('boundary_layer','edge_refined') else 1.5
            axes[0]=.24*(t**power if layout.startswith('left') else 1-(1-t)**power)
            z=np.linspace(-1,1,z_refinement*n+1);s=abs(z)
            shape=(126*s**5-420*s**6+540*s**7-315*s**8+70*s**9) if grading=='boundary_layer' else (1-np.cos(np.pi*s))/2
            axes[2]=np.sign(z)*.04*shape
            X=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
            self.rest=self.rest.copy();self.rest[:len(X)]=X
            for k,(i,j) in enumerate(EDGES):self.rest[self.elements[:,4+k]]=(X[self.tets[:,i]]+X[self.tets[:,j]])/2
            self.reference_local=self.rest;self.axes=axes
            corners=X[self.tets];Dm=np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2)
            self.volumes=np.linalg.det(Dm)/6;inverse=np.linalg.inv(Dm)
            self.linear_gradients=np.concatenate([-inverse.sum(1)[:,None],inverse],1)
            assert self.volumes.min()>0
        self.attach(layout,K,2);self.engine=GPUQuadratic(self,device) if device else None
    def elastic(self,positions,tangent=False):
        return self.engine.elastic(positions,tangent) if self.engine is not None else super().elastic(positions,tangent)


def locate(model,points):
    points=np.asarray(points);cells=np.asarray(getattr(model,'cells',[4*model.n,model.n,model.n]));lo=np.array([0.,-.04,-.04]);size=np.array([.24,.08,.08])
    coordinates=(points-lo)/size*cells
    if np.any(coordinates < -1e-10) or np.any(coordinates>cells+1e-10):raise ValueError('Query outside block')
    axes=getattr(model,'axes',[np.linspace(lo[d],lo[d]+size[d],cells[d]+1) for d in range(3)])
    cell=np.stack([np.searchsorted(axes[d],points[:,d],side='right')-1 for d in range(3)],axis=1)
    cell=np.minimum(np.maximum(cell,0),cells-1)
    flat=np.ravel_multi_index(cell.T,tuple(cells));ids=6*flat[:,None]+np.arange(6)
    X=model.rest[model.tets[ids]];inv=np.linalg.inv(np.swapaxes(X[:,:,1:]-X[:,:,:1],-1,-2))
    local=np.einsum('pkij,pkj->pki',inv,points[:,None]-X[:,:,0])
    bary=np.concatenate([1-local.sum(-1,keepdims=True),local],axis=-1)
    inside=np.all(bary>=-1e-10,axis=-1)&np.all(bary<=1+1e-10,axis=-1)
    assert inside.any(1).all()
    k=inside.argmax(1);return ids[np.arange(len(points)),k],bary[np.arange(len(points)),k]


def query(model,u,points,gradient=False):
    if hasattr(model,'query_field'):return model.query_field(u,points,gradient)
    ids,b=locate(model,points)
    if model.degree==1:
        if gradient:return np.einsum('pai,paJ->piJ',u[model.tets[ids]],model.gradients[ids])
        return np.einsum('pa,pai->pi',b,u[model.tets[ids]])
    if gradient:
        g=shape_gradients(b,model.linear_gradients[ids]);return np.einsum('pai,paJ->piJ',u[model.elements[ids]],g)
    return np.einsum('pa,pai->pi',shapes(b),u[model.elements[ids]])


class GPUQuadratic:
    """FP64 log-J material assembly on CUDA; original sparse Newton solve on CPU."""
    def __init__(self,model,device):
        import torch
        self.t=torch;self.model=model;self.device=device
        self.rest=self.tensor(model.rest);self.elements=self.tensor(model.elements,torch.int64)
        self.linear=self.tensor(model.linear_gradients)
        self.bary=self.tensor(model.bary);self.check_bary=self.tensor(model.check_bary)
        self.weights=self.tensor(model.weights[:,None]*model.volumes[None]);self.I=self.tensor(np.eye(3))
        self.chunk=1 if len(model.elements)>50000 else 4
    def tensor(self,x,dtype=None):return self.t.as_tensor(np.asarray(x),dtype=dtype or self.t.float64,device=self.device)
    def gradients(self,b,linear=None):
        linear=self.linear if linear is None else linear
        vertex=(4*b[:,None,:,None]-1)*linear[None]
        edge=self.t.stack([4*(b[:,None,i,None]*linear[None,:,j,:]+b[:,None,j,None]*linear[None,:,i,:]) for i,j in EDGES],dim=2)
        return self.t.cat([vertex,edge],dim=2)
    def elastic(self,positions,tangent=False):
        t=self.t;m=self.model;u=self.tensor(positions)-self.rest;ue=u[self.elements]
        mu,lam=lame(m.young,m.poisson);energy=t.zeros((),dtype=t.float64,device=self.device)
        ge=t.zeros_like(ue)
        Js=[]
        for start in range(0,len(self.check_bary),self.chunk):
            F=self.I+t.einsum('eai,qeaj->qeij',ue,self.gradients(self.check_bary[start:start+self.chunk]));Js.append(t.linalg.det(F).reshape(-1))
        for start in range(0,len(self.bary),self.chunk):
            g=self.gradients(self.bary[start:start+self.chunk]);v=self.weights[start:start+self.chunk]
            G=t.einsum('eai,qeaj->qeij',ue,g);F=self.I+G;J=t.linalg.det(F);Js.append(J.reshape(-1))
            if not bool(t.isfinite(J).all()) or float(J.min())<=m.minimum_J:raise ValueError('Quadrature Jacobian guard')
            tr=t.diagonal(G,dim1=-2,dim2=-1).sum(-1);second=.5*(tr*tr-t.einsum('qeij,qeji->qe',G,G));detG=t.linalg.det(G)
            jm=tr+second+detG;log=t.log1p(jm);invT=t.linalg.inv(F).transpose(-1,-2)
            small=sum((-1.)**k*jm**k/k for k in range(2,10));xm=t.where(abs(jm)<1e-3,small,jm-log)
            W=mu*(.5*(G*G).sum((-1,-2))-second-detG+xm)+.5*lam*log*log
            P=mu*G+(lam*log-mu)[...,None,None]*(invT-self.I)+lam*log[...,None,None]*self.I
            energy+=(v*W).sum();ge+=t.einsum('qe,qeiJ,qeaJ->eai',v,P,g)
        J=t.cat(Js)
        if not bool(t.isfinite(J).all()) or float(J.min())<=m.minimum_J:raise ValueError('Nodal Jacobian guard')
        grad=t.zeros_like(self.rest);grad.index_add_(0,self.elements.reshape(-1),ge.reshape(-1,3))
        H=None
        if tangent:
            # Accumulate one element block across all quadrature points, then
            # transfer it once. The complete element tangent lives on CPU.
            nodes=ue.shape[1];values=np.empty((len(ue),nodes,3,nodes,3),dtype=np.float64)
            for first in range(0,len(ue),8192):
                sl=slice(first,first+8192);ub=ue[sl];he=t.zeros((len(ub),nodes,3,nodes,3),dtype=t.float64,device=self.device)
                for q in range(0,len(self.bary),4):
                    g=self.gradients(self.bary[q:q+4],self.linear[sl]);v=self.weights[q:q+4,sl]
                    G=t.einsum('eai,qeaj->qeij',ub,g);F=self.I+G;invT=t.linalg.inv(F).transpose(-1,-2)
                    tr=t.diagonal(G,dim1=-2,dim2=-1).sum(-1);second=.5*(tr*tr-t.einsum('qeij,qeji->qe',G,G))
                    log=t.log1p(tr+second+t.linalg.det(G));D=t.einsum('qeiJ,qeaJ->qeai',invT,g)
                    he+=mu*t.einsum('qe,qeaJ,qebJ,ik->eaibk',v,g,g,self.I)
                    he+=t.einsum('qe,qeak,qebi->eaibk',v*(mu-lam*log),D,D)
                    he+=lam*t.einsum('qe,qeai,qebk->eaibk',v,D,D)
                values[sl]=he.cpu().numpy()
            H=sparse.coo_matrix((values.ravel(),(m.rows,m.cols)),shape=(m.rest.size,m.rest.size)).tocsr()
        return float(energy),grad.cpu().numpy(),H,J.cpu().numpy()
