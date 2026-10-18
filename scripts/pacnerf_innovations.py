"""Exact causal innovation coordinates for the frozen PAC transition density.

For marginalized loads, conjugate Gaussian filtering integrates the same global
coefficients sequentially. Each next state has a Gaussian conditional, even
when diffusion/contact depend nonlinearly on previous states. Its Cholesky maps
independent standard innovations into the trajectory. The triangular Jacobian
cancels all conditional Gaussian covariance normalizers in innovation space.
This changes coordinates, not the physical model, force prior or image data.
"""
import math
import torch


class InnovationModel:
    def __init__(self,model):
        self.model=model;self.dimension=model.dimension;self.modes=model.modes

    def operators(self,z):
        model=self.model;m=model.modes
        transition=model.transition(z,model.dt)
        drive=z.new_zeros(len(z),2*m,m);quadrature=[]
        for point,weight in zip([-.8611363115940526,-.3399810435848563,.3399810435848563,.8611363115940526],
            [.3478548451374538,.6521451548625461,.6521451548625461,.3478548451374538]):
            f=model.transition(z,model.dt*(point+1)/2)[:,:,m:]
            w=weight*model.dt/2;drive=drive+w*f;quadrature.append((w,f))
        mapping=drive[:,None]@model.acceleration_mapping[None]
        return transition,drive,quadrature,mapping

    def step(self,q,v,drive,quadrature):
        model=self.model;m=model.modes
        x=model.data['rest'][None]+torch.einsum('bm,ncm->bnc',q,model.basis)
        velocity=torch.einsum('bm,ncm->bnc',v,model.basis)
        delta=x[:,model.edges[:,1]]-x[:,model.edges[:,0]]
        direction=delta/(delta.square().sum(-1,keepdim=True)+1e-12).sqrt()
        strain=(delta.square().sum(-1)+1e-12).sqrt()/model.length-1
        radial=((velocity[:,model.edges[:,1]]-velocity[:,model.edges[:,0]])*direction).sum(-1)
        features=torch.stack((strain,radial),-1)
        amp=model.network(features)/model.constant if model.noise=='neural' else torch.ones_like(strain)
        edge_basis=model.basis[model.edges[:,1]]-model.basis[model.edges[:,0]]
        factor=torch.einsum('ecm,bec->bme',edge_basis,direction)*amp[:,None,:]
        factor=factor*model.internal_scale/math.sqrt(len(model.edges)/len(model.data['rest']))
        spectral=factor@factor.transpose(-1,-2)+model.acc_noise_floor**2*torch.eye(m,dtype=q.dtype)
        covariance=sum(w*(f@spectral@f.transpose(-1,-2)) for w,f in quadrature)
        nodal=x.new_zeros(x.shape);nodal[...,1]=-9.8+6*torch.nn.functional.softplus(-x[...,1]/.01)
        acceleration=torch.einsum('bnc,ncm->bm',nodal,model.basis)
        return (drive@acceleration[...,None]).squeeze(-1),covariance

    def transform(self,z,inverse=False):
        """Return other coordinates and log|d old_path / d innovations|.

        Forward input is innovations; inverse input is the original standardized
        path coordinate. Parameters/initial velocity/explicit loads are shared.
        """
        model=self.model;m=model.modes;b=len(z);k=model.force_dimension
        transition,drive,quadrature,mapping=self.operators(z)
        if inverse:
            q_path,v_path=model.decode(z)
            supplied=torch.cat((q_path[:,1:],v_path[:,1:]),-1)
        else:supplied=z[:,2+m:model.base_dimension].reshape(b,13,2*m)
        current=torch.cat((z.new_zeros(b,m),.5*z[:,2:2+m]),-1)
        precision=torch.eye(k,dtype=z.dtype).expand(b,k,k).clone();rhs=z.new_zeros(b,k)
        converted=[];log_jac=z.new_zeros(b)
        coefficients=z[:,model.base_dimension:] if model.force=='joint' else None
        for time in range(13):
            external,covariance=self.step(current[:,:m],current[:,m:],drive,quadrature)
            base=(transition@current[...,None]).squeeze(-1)+external
            g=mapping[:,time]
            if coefficients is None:
                chol_p=torch.linalg.cholesky(precision)
                conditional_mean=torch.cholesky_solve(rhs[...,None],chol_p).squeeze(-1)
                sg=torch.cholesky_solve(g.transpose(-1,-2),chol_p)
                mean=base+(g@conditional_mean[...,None]).squeeze(-1)
                conditional_covariance=covariance+g@sg
                conditional_covariance=(conditional_covariance+conditional_covariance.transpose(-1,-2))/2
            else:
                mean=base+(g@coefficients[...,None]).squeeze(-1)
                conditional_covariance=covariance
            chol=torch.linalg.cholesky(conditional_covariance)
            log_jac=log_jac+chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)
            if inverse:
                following=supplied[:,time]
                converted.append(torch.linalg.solve_triangular(chol,(following-mean)[...,None],upper=False).squeeze(-1))
            else:
                following=mean+(chol@supplied[:,time,:,None]).squeeze(-1)
                converted.append(following)
            if coefficients is None:
                chol_q=torch.linalg.cholesky(covariance)
                white_g=torch.linalg.solve_triangular(chol_q,g,upper=False)
                white_r=torch.linalg.solve_triangular(chol_q,(following-base)[...,None],upper=False).squeeze(-1)
                precision=precision+white_g.transpose(-1,-2)@white_g
                rhs=rhs+(white_g.transpose(-1,-2)@white_r[...,None]).squeeze(-1)
            current=following
        middle=torch.stack(converted,1).flatten(1)
        if not inverse:middle=middle/model.scale[2+m:model.base_dimension]
        result=torch.cat((z[:,:2+m],middle,z[:,model.base_dimension:]),-1)
        log_jac=log_jac-model.scale[2+m:model.base_dimension].log().sum()
        return result,log_jac

    def to_physical(self,z):return self.transform(z)[0]

    def objective(self,z):
        physical=self.to_physical(z)
        residual=self.model.image_residual(physical)
        n=self.dimension-(2+self.modes)
        return .5*z.square().sum(-1)+.5*residual.square().sum(-1)+.5*n*math.log(2*math.pi)
