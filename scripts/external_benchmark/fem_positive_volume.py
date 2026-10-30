# Numerical-domain extension of frozen fem.py. Identical Neo-Hookean energy,
# residual and Hessian; reject J <= 1e-6 instead of the artificial J <= .03 cutoff.
# Kept separate to preserve and report the original frozen benchmark unchanged.
"""Arbitrary-cloud Neo-Hookean FEM with material and world-space boundaries.

Dimensionless energy: material coordinates define tetrahedra/stiffness; world
coordinates define contact/potential. Only prescribed positions drive attachments.
The initial observed shape is treated as rest geometry, a disclosed approximation.
"""
import numpy as np
import torch
from scipy.spatial import Delaunay, cKDTree


class PositiveVolumeFEM:
    def __init__(self,data,device='cuda:0',field=True,contact=False,nonconvex=False,materials=1):
        self.device=device;self.dtype=torch.float64;self.field=field;self.contact=contact
        nodes=data['nodes'];self.origin=nodes.mean(0);self.scale=float(np.max(np.ptp(nodes,axis=0)))
        rest=(nodes-self.origin)/self.scale
        elements=Delaunay(rest).simplices.copy()
        self.mesh_filter='convex_hull'
        if nonconvex and 'support_geometry' in data:
            support=(data['support_geometry']-self.origin)/self.scale
            tree=cKDTree(support)
            # Reject tetrahedra crossing empty space between limbs using ONLY
            # the released initial surface/interior reconstruction.
            probes=np.concatenate([rest[elements].mean(1)[:,None],
                np.stack([(rest[elements[:,i]]+rest[elements[:,j]])/2 for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]],1)],1)
            distances=tree.query(probes.reshape(-1,3))[0].reshape(len(elements),-1)
            elements=elements[(distances<.045).all(1)]
            if len(elements)<10:raise ValueError('Initial geometry filtering removed almost all tetrahedra')
            self.mesh_filter='initial_geometry_distance_0.045_extent'
        dm=np.stack([rest[elements[:,i]]-rest[elements[:,0]] for i in [1,2,3]],axis=-1)
        det=np.linalg.det(dm);good=np.abs(det)>1e-9;elements=elements[good];dm=dm[good];det=det[good]
        inv=np.linalg.inv(dm)
        grad=np.concatenate([-inv.sum(1,keepdims=True),inv],axis=1)
        volume=np.abs(det)/6
        self.t=lambda x:torch.as_tensor(x,dtype=self.dtype,device=self.device)
        self.rest=self.t(rest);self.elements=torch.as_tensor(elements,device=device)
        self.grad=self.t(grad);self.vol=self.t(volume);self.n=len(rest);self.ndof=self.n*3
        self.eye=torch.eye(3,dtype=self.dtype,device=device)
        self.mass=torch.zeros(self.n,dtype=self.dtype,device=device).index_add(0,self.elements.ravel(),self.vol[:,None].expand(-1,4).reshape(-1)/4)
        self.mass=self.mass/self.mass.sum()
        edof=(self.elements[:,:,None]*3+torch.arange(3,device=device)).reshape(-1,12)
        self.rows=edof[:,:,None].expand(-1,12,12).reshape(-1);self.cols=edof[:,None,:].expand(-1,12,12).reshape(-1)
        self.weights=self.t(data['attachment_weights']);self.weights=self.weights/self.weights.sum(1,keepdim=True).clamp_min(1e-9)
        self.targets=self.t((data['controller_targets']-self.origin)/self.scale)
        self.active=self.t(data['attachment_active']);self.floor=(data['floor']-self.origin[2])/self.scale
        centers=self.rest[self.elements].mean(1)
        self.n_material=materials
        if materials==1:
            self.feature=(torch.sin(centers[:,0]*3)*torch.cos(centers[:,1]*3))[:,None]
        else:
            from scripts.external_benchmark.data import fps
            kernels=self.rest[fps(rest,materials)]
            features=torch.exp(-((centers[:,None]-kernels[None])**2).sum(-1)/(2*.25**2))
            features=features/features.sum(1,keepdim=True).clamp_min(1e-12)
            self.feature=features-(self.vol[:,None]*features).sum(0)/self.vol.sum()
        self.field_center=self.t([0,0,self.floor]);self.width=.45
        self.nodes=nodes
        self.i_floor=1+len(self.weights);self.i_material=self.i_floor+1;self.i_field=self.i_material+materials
        self.inertial_target=None;self.inertial_dt=None

    def evaluate(self,x,theta,frame,tangent=False):
        # Dimensionless gravity/material ratio; fixed nu=.35. Two attachment strengths.
        gravity=torch.exp(theta[0]); stiffness=torch.exp(theta[1:1+len(self.weights)])
        material=torch.exp(self.feature@theta[self.i_material:self.i_field])
        mu=material/2.7;lam=material*.35/(1.35*.3)
        F=torch.einsum('eai,eaj->eij',x[self.elements],self.grad)
        J=torch.linalg.det(F)
        if float(J.detach().min())<=1e-6:raise ValueError('Nonpositive or near-zero tetrahedral volume')
        logJ=torch.log(J); invT=torch.linalg.inv(F).transpose(-1,-2)
        W=.5*mu*((F*F).sum((-1,-2))-3)-mu*logJ+.5*lam*logJ**2
        P=mu[:,None,None]*F+(lam*logJ-mu)[:,None,None]*invT
        local=torch.einsum('e,eiJ,eaJ->eai',self.vol,P,self.grad)
        r=torch.zeros_like(x).index_add(0,self.elements.reshape(-1),local.reshape(-1,3))
        energy=(self.vol*W).sum()
        w=(self.weights*stiffness[:,None]*self.active[frame,:,None])
        delta=x[None]-self.targets[frame];energy=energy+.5*(w[:,:,None]*delta**2).sum();r=r+(w[:,:,None]*delta).sum(0)
        # Gravity is a world-space body load; mass ratios fixed from initial geometry.
        energy=energy+gravity*(self.mass*x[:,2]).sum();r=r+gravity*self.mass[:,None]*self.t([0,0,1])
        # Unilateral floor. Initial minimum geometry is known, not future floor labels.
        floor_k=torch.exp(theta[self.i_floor]);penetration=torch.minimum(x[:,2]-self.floor,torch.zeros(self.n,device=self.device))
        energy=energy+.5*floor_k*(self.mass*penetration**2).sum()
        r=r+floor_k*(self.mass*penetration)[:,None]*self.t([0,0,1])
        # Very weak positional gauge removes free horizontal nullspaces before grasp.
        pin=1e-5;energy=energy+.5*pin*((x-self.rest)**2).sum();r=r+pin*(x-self.rest)
        value=torch.zeros(self.n,device=self.device,dtype=self.dtype)
        d=x-self.field_center;s2=self.width**2
        if self.field:
            value=.1*theta[self.i_field]*self.mass*torch.exp(-(d*d).sum(1)/(2*s2))
            energy=energy+value.sum();r=r-value[:,None]*d/s2
        # Reversible tangential compliance localized near a world-space plane.
        # This is a compliant-support hypothesis, not a Coulomb-friction solver.
        contact_blocks=torch.zeros((self.n,3,3),device=self.device,dtype=self.dtype)
        if self.contact:
            h2=.06**2;z=x[:,2]-self.floor;xy=x[:,:2]-self.rest[:,:2];rr=(xy*xy).sum(1)
            displacement=torch.cat([xy,z[:,None]],1)
            inverse_width=self.t([1/.3**2,1/.3**2,1/h2])
            a=displacement*inverse_width
            cw=torch.exp(theta[self.i_field+1])*self.mass*torch.exp(-.5*(displacement*a).sum(1))
            energy=energy+.5*(cw*rr).sum()
            dq=torch.cat([2*xy,torch.zeros((self.n,1),device=self.device,dtype=self.dtype)],1)
            r=r+.5*cw[:,None]*(dq-rr[:,None]*a)
            ddq=torch.diag(self.t([2,2,0]))
            contact_blocks=.5*cw[:,None,None]*(ddq-dq[:,:,None]*a[:,None,:]-a[:,:,None]*dq[:,None,:]+rr[:,None,None]*(a[:,:,None]*a[:,None,:]-torch.diag(inverse_width)))
        inertia=torch.zeros_like(self.mass)
        if self.inertial_target is not None:
            # Implicit Euler in nondimensional coordinates; the gravity ratio
            # fixes mass relative to elastic energy without measured forces.
            inertia=gravity*self.scale/(9.81*self.inertial_dt**2)*self.mass
            difference=x-self.inertial_target
            energy=energy+.5*(inertia[:,None]*difference**2).sum()
            r=r+inertia[:,None]*difference
        H=None
        if tangent:
            D=torch.einsum('eiJ,eaJ->eai',invT,self.grad)
            h=torch.einsum('e,eaJ,ebJ,ik->eaibk',self.vol*mu,self.grad,self.grad,self.eye)
            h=h+torch.einsum('e,eak,ebi->eaibk',self.vol*(mu-lam*logJ),D,D)+torch.einsum('e,eai,ebk->eaibk',self.vol*lam,D,D)
            H=torch.zeros((self.ndof,self.ndof),dtype=self.dtype,device=self.device).index_put((self.rows,self.cols),h.reshape(-1),accumulate=True)
            blocks=(w.sum(0)+pin+inertia)[:,None,None]*self.eye[None]
            blocks=blocks+value[:,None,None]*(d[:,:,None]*d[:,None,:]/s2**2-self.eye/s2)
            blocks[:,2,2]+=floor_k*self.mass*(x[:,2]<self.floor)
            blocks=blocks+contact_blocks
            ids=torch.arange(self.ndof,device=self.device).reshape(-1,3)
            H=H.index_put((ids[:,:,None].expand_as(blocks).reshape(-1),ids[:,None,:].expand_as(blocks).reshape(-1)),blocks.reshape(-1),accumulate=True)
        return energy,r,H

    @torch.no_grad()
    def solve(self,theta,frame,initial=None,tolerance=1e-7,maxiter=80):
        x=self.rest.clone() if initial is None else initial.clone()
        iterations=0
        for iteration in range(maxiter):
            E,r,H=self.evaluate(x,theta,frame,True);norm=float(torch.linalg.vector_norm(r));iterations=iteration+1
            if norm<tolerance:return x,H,dict(iterations=iterations,residual=norm,minJ=self.min_jacobian(x))
            diag=torch.diag(H).abs().mean().clamp_min(1e-5)
            accepted=False
            for damping in [0.,1e-5,1e-3,.1,1.,10.]:
                matrix=H+damping*diag*torch.eye(self.ndof,device=self.device,dtype=self.dtype)
                try:direction=torch.linalg.solve(matrix,-r.flatten()).reshape_as(x)
                except RuntimeError:continue
                slope=(direction*r).sum()
                if not torch.isfinite(direction).all() or float(slope)>=0:continue
                for j in range(22):
                    step=2.**(-j)
                    try:newE,newr,_=self.evaluate(x+step*direction,theta,frame)
                    except ValueError:continue
                    if float(newE)<=float(E+1e-4*step*slope):
                        x=x+step*direction;accepted=True;break
                if accepted:break
            if not accepted:raise RuntimeError(f'Equilibrium line search failed frame={frame} residual={norm:g}')
        E,r,H=self.evaluate(x,theta,frame,True);norm=float(torch.linalg.vector_norm(r))
        if norm>1e-5:raise RuntimeError(f'Equilibrium iteration limit frame={frame} residual={norm:g}')
        return x,H,dict(iterations=iterations,residual=norm,minJ=self.min_jacobian(x))

    def min_jacobian(self,x):
        return float(torch.linalg.det(torch.einsum('eai,eaj->eij',x[self.elements],self.grad)).detach().min())

    def predict(self,theta,frame,initial=None):
        return Equilibrium.apply(theta,self,frame,initial)


class Equilibrium(torch.autograd.Function):
    @staticmethod
    def forward(ctx,theta,model,frame,initial):
        x,H,diagnostic=model.solve(theta,frame,initial)
        model.last_diagnostic=diagnostic
        ctx.save_for_backward(theta,x,H);ctx.model=model;ctx.frame=frame
        return x
    @staticmethod
    def backward(ctx,grad):
        theta,x,H=ctx.saved_tensors
        adjoint=torch.linalg.solve(H.T,grad.reshape(-1)).reshape_as(x)
        with torch.enable_grad():
            th=theta.detach().requires_grad_(True)
            _,r,_=ctx.model.evaluate(x.detach(),th,ctx.frame)
            gt,=torch.autograd.grad(r,th,grad_outputs=-adjoint)
        return gt,None,None,None
