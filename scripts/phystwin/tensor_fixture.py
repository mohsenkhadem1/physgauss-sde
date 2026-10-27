"""Independent Q2 tensor-product space for the same nonlinear Robin block."""
import numpy as np
from scipy import sparse
from numpy.polynomial.legendre import leggauss
from scripts.phystwin.consistent_fixture import ConsistentFixture,GPUQuadratic


def basis(points):
    p=np.asarray(points);b=np.stack([.5*p*(p-1),1-p*p,.5*p*(p+1)],axis=-1)
    db=np.stack([p-.5,-2*p,p+.5],axis=-1)
    N=np.einsum('qi,qj,qk->qijk',b[:,0],b[:,1],b[:,2]).reshape(len(p),27)
    G=np.stack([np.einsum('qi,qj,qk->qijk',db[:,0],b[:,1],b[:,2]),np.einsum('qi,qj,qk->qijk',b[:,0],db[:,1],b[:,2]),np.einsum('qi,qj,qk->qijk',b[:,0],b[:,1],db[:,2])],axis=-1).reshape(len(p),27,3)
    return N,G


class TensorFixture(ConsistentFixture):
    def __init__(self,n,E,K,layout,order=3,device='cuda:0',x_power=1.5,memory_limit_bytes=24*1024**3):
        if not isinstance(n,(int,np.integer)) or n<1:raise ValueError('Positive integer mesh size required')
        if x_power<=0 or order not in (3,4):raise ValueError('Positive grading and quadrature order 3 or 4 required')
        if 4*n**3*81**2*24>memory_limit_bytes:raise MemoryError('Tensor tangent assembly exceeds declared memory limit')
        if n%2 and layout.endswith('lower'):raise ValueError('Half-face patch must align with z=0')
        self.n=n;self.cells=np.array([4*n,n,n]);self.degree=2;self.layout=layout
        self.young=E;self.K=K;self.poisson=.35;self.minimum_J=.1;self.order=order
        t=np.linspace(0,1,4*n+1);x=.24*(t**x_power if layout.startswith('left') else 1-(1-t)**x_power)
        s=np.linspace(-1,1,n+1);z=np.sign(s)*.04*(1-np.cos(np.pi*abs(s)))/2
        self.axes=[x,np.linspace(-.04,.04,n+1),z]
        fine=[]
        for axis in self.axes:
            a=np.empty(2*len(axis)-1);a[::2]=axis;a[1::2]=(axis[:-1]+axis[1:])/2;fine.append(a)
        self.rest=np.stack(np.meshgrid(*fine,indexing='ij'),-1).reshape(-1,3);self.grid_shape=tuple(len(a) for a in fine)
        cell=np.stack(np.meshgrid(*[np.arange(k) for k in self.cells],indexing='ij'),-1).reshape(-1,3)
        offset=np.stack(np.meshgrid(*[np.arange(3)]*3,indexing='ij'),-1).reshape(-1,3)
        self.elements=np.ravel_multi_index((2*cell[:,None]+offset[None]).transpose(2,0,1),self.grid_shape)
        self.widths=np.stack([np.diff(a)[cell[:,i]] for i,a in enumerate(self.axes)],axis=1);self.volumes=self.widths.prod(1)
        dofs=(self.elements[:,:,None]*3+np.arange(3)).reshape(-1,81)
        self.rows=np.repeat(dofs,81,1).ravel().astype(np.int32);self.cols=np.tile(dofs,(1,81)).ravel().astype(np.int32)
        q,w=leggauss(order);self.quad=np.stack(np.meshgrid(q,q,q,indexing='ij'),-1).reshape(-1,3)
        self.weights=np.prod(np.stack(np.meshgrid(w,w,w,indexing='ij'),-1),axis=-1).ravel()/8
        self.check=np.stack(np.meshgrid(*[np.array([-1.,0.,1.])]*3,indexing='ij'),-1).reshape(-1,3)
        M,_,self.attachment_area=self.surface(layout.startswith('left'),layout.endswith('lower'))
        self.fixture_scalar=K*M/self.attachment_area;self.spring_matrix=sparse.kron(self.fixture_scalar,sparse.eye(3),format='csr')
        _,self.left,area=self.surface(True,False);self.left/=area
        _,self.right,area=self.surface(False,False);self.right/=area
        self.load=np.zeros_like(self.rest);self.mask=np.zeros_like(self.rest,dtype=bool);self.free=np.arange(self.rest.size)
        self.engine=GPUHex(self,device);self.linear_solver='gpu_multigrid';self.force=10.
    def surface(self,left,lower):
        cells=np.stack(np.meshgrid(np.arange(self.cells[1]),np.arange(self.cells[2]),indexing='ij'),-1).reshape(-1,2)
        if lower:cells=cells[self.axes[2][cells[:,1]+1]<=1e-13]
        offset=np.stack(np.meshgrid(np.arange(3),np.arange(3),indexing='ij'),-1).reshape(-1,2)
        yz=2*cells[:,None]+offset;ix=np.full(yz.shape[:2],0 if left else self.grid_shape[0]-1)
        nodes=np.ravel_multi_index((ix,yz[:,:,0],yz[:,:,1]),self.grid_shape)
        area=np.diff(self.axes[1])[cells[:,0]]*np.diff(self.axes[2])[cells[:,1]]
        mass=np.array([[4.,2.,-1.],[2.,16.,2.],[-1.,2.,4.]])/15
        local=np.kron(mass,mass)/4;integral=np.outer([1.,4.,1.],[1.,4.,1.]).ravel()/36
        rows=np.broadcast_to(nodes[:,:,None],(len(nodes),9,9));cols=np.broadcast_to(nodes[:,None,:],rows.shape)
        M=sparse.coo_matrix(((area[:,None,None]*local).ravel(),(rows.ravel(),cols.ravel())),shape=(len(self.rest),len(self.rest))).tocsr()
        load=np.zeros(len(self.rest));np.add.at(load,nodes,area[:,None]*integral)
        return M,load,float(area.sum())
    def elastic(self,positions,tangent=False):return self.engine.elastic(positions,tangent)
    def query_field(self,u,points,gradient=False):
        p=np.asarray(points);cell=np.stack([np.clip(np.searchsorted(a,p[:,i],side='right')-1,0,len(a)-2) for i,a in enumerate(self.axes)],axis=1)
        lo=np.stack([a[cell[:,i]] for i,a in enumerate(self.axes)],axis=1);width=np.stack([np.diff(a)[cell[:,i]] for i,a in enumerate(self.axes)],axis=1)
        local=2*(p-lo)/width-1
        if np.any(abs(local)>1+1e-9):raise ValueError('Query outside tensor block')
        ids=np.ravel_multi_index(cell.T,tuple(self.cells));N,G=basis(local)
        if gradient:return np.einsum('pai,paJ->piJ',u[self.elements[ids]],G*2/width[:,None,:])
        return np.einsum('pa,pai->pi',N,u[self.elements[ids]])


class GPUHex(GPUQuadratic):
    def __init__(self,model,device):
        import torch
        self.t=torch;self.model=model;self.device=device
        self.rest=self.tensor(model.rest);self.elements=self.tensor(model.elements,torch.int64);self.linear=self.tensor(2/model.widths)
        _,grad=basis(np.r_[model.quad,model.check]);self.reference_grad=self.tensor(grad)
        self.bary=self.tensor(np.arange(len(model.quad))[:,None],torch.int64);self.check_bary=self.tensor(np.arange(len(model.quad),len(grad))[:,None],torch.int64)
        self.weights=self.tensor(model.weights[:,None]*model.volumes[None]);self.I=self.tensor(np.eye(3));self.chunk=1 if len(model.elements)>50000 else 4
    def gradients(self,b,linear=None):
        linear=self.linear if linear is None else linear
        return self.reference_grad[b[:,0]][:,None]*linear[None,:,None,:]
