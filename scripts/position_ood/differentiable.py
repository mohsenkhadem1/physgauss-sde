"""Implicitly differentiated Q2 equilibrium for positional inverse mechanics.

Small dense tangents provide an independently checked inverse-model backend.
Unknown parameters describe material and spatial environmental fields; controller
poses are the action input. No measured forces are needed by this interface.
"""
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from scripts.phystwin.tensor_fixture import TensorFixture,basis

class DensePositionFEM:
    def __init__(self,n=1,device='cuda:0',order=3,field_frame='world'):
        self.device=device;self.dtype=torch.float64;self.n=n;self.field_frame=field_frame
        if field_frame not in ['world','material']:raise ValueError('field frame must be world or material')
        geo=TensorFixture(n,1.,1.,'left_full',device=device,order=order,x_power=1.)
        self.geo=geo;self.rest=self.tensor(geo.rest);self.elements=self.tensor(geo.elements,torch.long)
        self.grad=geo.engine.gradients(geo.engine.bary).detach();self.check_grad=geo.engine.gradients(geo.engine.check_bary).detach()
        N,_=basis(geo.quad);X=torch.einsum('qa,eai->qei',self.tensor(N),self.rest[self.elements]);self.material_feature=torch.exp(-((X[...,0]-.06)/.07)**2)-torch.exp(-((X[...,0]-.18)/.07)**2)
        self.weights=self.tensor(geo.weights[:,None]*geo.volumes[None]);self.eye=torch.eye(3,dtype=self.dtype,device=device)
        self.rows=self.tensor(geo.rows,torch.long);self.cols=self.tensor(geo.cols,torch.long);self.ndof=self.rest.numel()
        integral=np.einsum('i,j,k->ijk',[1,4,1],[1,4,1],[1,4,1]).ravel()/216
        weights=np.zeros(len(geo.rest));np.add.at(weights,geo.elements,geo.volumes[:,None]*integral);weights/=weights.sum()
        self.mass_weights=self.tensor(weights);self.mass=625.*geo.volumes.sum();self.field_center=self.tensor([.12,0,-.09]);self.field_width=.08
        self.matrices=[]
        for pair in [('lower','lower'),('lower','upper'),('yminus','yplus')]:self.matrices.append([self.tensor(self.patch(left,kind)) for left,kind in zip([True,False],pair)])
        self.eye_dof=torch.eye(self.ndof,dtype=self.dtype,device=device)
    def tensor(self,x,dtype=None):return torch.as_tensor(x,dtype=dtype or self.dtype,device=self.device)
    def patch(self,left,kind):
        g=self.geo;q,w=leggauss(3);M=np.zeros((len(g.rest),len(g.rest)));area=0.
        bounds={'lower':(-.04,.04,-.04,0),'upper':(-.04,.04,0,.04),'yminus':(-.04,0,-.04,.04),'yplus':(0,.04,-.04,.04)}[kind]
        for iy in range(g.cells[1]):
            for iz in range(g.cells[2]):
                ya=max(g.axes[1][iy],bounds[0]);yb=min(g.axes[1][iy+1],bounds[1]);za=max(g.axes[2][iz],bounds[2]);zb=min(g.axes[2][iz+1],bounds[3])
                if yb<=ya or zb<=za:continue
                yy=(q+1)*(yb-ya)/2+ya;zz=(q+1)*(zb-za)/2+za
                points=np.stack(np.meshgrid(yy,zz,indexing='ij'),-1).reshape(-1,2)
                local=np.c_[np.full(9,-1. if left else 1.),2*(points[:,0]-g.axes[1][iy])/(g.axes[1][iy+1]-g.axes[1][iy])-1,2*(points[:,1]-g.axes[2][iz])/(g.axes[2][iz+1]-g.axes[2][iz])-1]
                N,_=basis(local);ww=np.outer(w,w).ravel()*(yb-ya)*(zb-za)/4
                idx=np.ravel_multi_index((0 if left else g.cells[0]-1,iy,iz),tuple(g.cells));ids=g.elements[idx];M[np.ix_(ids,ids)]+=N.T@(ww[:,None]*N);area+=(yb-ya)*(zb-za)
        return M/area
    def rotations(self,r):
        zero=r.new_zeros(());x,y,z=r;K=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
        angle=torch.linalg.vector_norm(r);return self.eye+torch.sinc(angle/torch.pi)*K+.5*torch.sinc(angle/(2*torch.pi))**2*(K@K)
    def parts(self,theta,poses,layout):
        E,kl,kr=torch.exp(theta[:3]);mu=E/(2*1.35);lam=E*.35/(1.35*.3)
        Mleft=kl*self.matrices[layout][0];Mright=kr*self.matrices[layout][1];M=Mleft+Mright
        load=self.mass_weights[:,None]*self.mass*theta[3:6];constant=theta.new_zeros(())
        for i,B in enumerate([Mleft,Mright]):
            center=self.tensor([0. if i==0 else .24,0.,0.]);d=(self.rest-center)@self.rotations(poses[i,3:]).T+center+poses[i,:3]-self.rest
            load=load+B@d;constant=constant+.5*(d*(B@d)).sum()
        return mu,lam,M,load,constant
    def evaluate(self,u,theta,poses,layout,tangent=False):
        mu,lam,M,load,constant=self.parts(theta,poses,layout);ue=u[self.elements]
        coefficient=torch.exp(theta[7]*self.material_feature) if theta.numel()>7 else torch.ones_like(self.material_feature)
        mu=mu*coefficient;lam=lam*coefficient
        G=torch.einsum('eai,qeaj->qeij',ue,self.grad);F=self.eye+G;J=torch.linalg.det(F)
        Jcheck=torch.linalg.det(self.eye+torch.einsum('eai,qeaj->qeij',ue,self.check_grad))
        if min(float(J.detach().min()),float(Jcheck.detach().min()))<=.1:raise ValueError('Nonpositive/near-singular physical deformation')
        log=torch.log(J);invT=torch.linalg.inv(F).transpose(-1,-2)
        W=.5*mu*((F*F).sum((-1,-2))-3)-mu*log+.5*lam*log*log
        P=mu[...,None,None]*F+(lam*log-mu)[...,None,None]*invT
        ge=torch.einsum('qe,qeiJ,qeaJ->eai',self.weights,P,self.grad)
        residual=torch.zeros_like(u).index_add(0,self.elements.reshape(-1),ge.reshape(-1,3))+M@u-load
        energy=(self.weights*W).sum()+.5*(u*(M@u)).sum()-(u*load).sum()+constant
        center=theta[8:11] if theta.numel()>=12 else self.field_center
        width=torch.exp(theta[11]) if theta.numel()>=12 else self.field_width
        delta=(self.rest+u if self.field_frame=='world' else self.rest)-center;s2=width**2
        value=theta[6]*self.mass_weights*torch.exp(-(delta*delta).sum(1)/(2*s2))
        force_gradient=-value[:,None]*delta/s2
        energy=energy+value.sum()+(0. if self.field_frame=='world' else (force_gradient*u).sum());residual=residual+force_gradient
        H=None
        if tangent:
            D=torch.einsum('qeiJ,qeaJ->qeai',invT,self.grad)
            he=torch.einsum('qe,qeaJ,qebJ,ik->eaibk',self.weights*mu,self.grad,self.grad,self.eye)
            he=he+torch.einsum('qe,qeak,qebi->eaibk',self.weights*(mu-lam*log),D,D)+torch.einsum('qe,qeai,qebk->eaibk',self.weights*lam,D,D)
            H=torch.zeros((self.ndof,self.ndof),dtype=self.dtype,device=self.device).index_put((self.rows,self.cols),he.reshape(-1),accumulate=True)+torch.kron(M.contiguous(),self.eye)
            blocks=(1. if self.field_frame=='world' else 0.)*value[:,None,None]*(delta[:,:,None]*delta[:,None,:]/s2**2-self.eye[None]/s2)
            ids=torch.arange(self.ndof,device=self.device).reshape(-1,3);H=H.index_put((ids[:,:,None].expand_as(blocks).reshape(-1),ids[:,None,:].expand_as(blocks).reshape(-1)),blocks.reshape(-1),accumulate=True)
        return energy,residual,H
    @torch.no_grad()
    def solve(self,theta,poses,layout,initial=None,tolerance=1e-8):
        u=torch.zeros_like(self.rest) if initial is None else initial.clone()
        for iteration in range(45):
            energy,r,H=self.evaluate(u,theta,poses,layout,True);norm=torch.linalg.vector_norm(r)
            if float(norm)<tolerance:return u,H
            direction=torch.linalg.solve(H,-r.reshape(-1));slope=(direction*r.reshape(-1)).sum()
            if not torch.isfinite(direction).all() or float(slope)>=0:raise RuntimeError('Inverse equilibrium non-descent direction')
            for j in range(30):
                step=2.**(-j)
                try:newenergy,newr,_=self.evaluate(u+step*direction.reshape_as(u),theta,poses,layout)
                except ValueError:continue
                if float(newenergy)<=float(energy+1e-4*step*slope) or (abs(float(newenergy-energy))<1e-12 and float(torch.linalg.vector_norm(newr))<float(norm)):
                    u=u+step*direction.reshape_as(u);break
            else:raise RuntimeError('Inverse equilibrium line search failed')
        raise RuntimeError('Inverse equilibrium iteration limit')
    def predict(self,theta,poses,layout=0):return Equilibrium.apply(theta,poses,self,layout)

class Equilibrium(torch.autograd.Function):
    @staticmethod
    def forward(ctx,theta,poses,model,layout):
        u,H=model.solve(theta,poses,layout);ctx.save_for_backward(theta,poses,u,H);ctx.model=model;ctx.layout=layout;return u
    @staticmethod
    def backward(ctx,grad_u):
        theta,poses,u,H=ctx.saved_tensors;adjoint=torch.linalg.solve(H.T,grad_u.reshape(-1)).reshape_as(u)
        with torch.enable_grad():
            th=theta.detach().requires_grad_(True);ps=poses.detach().requires_grad_(True)
            _,r,_=ctx.model.evaluate(u.detach(),th,ps,ctx.layout)
            gt,gp=torch.autograd.grad(r,(th,ps),grad_outputs=-adjoint)
        return gt,gp,None,None
