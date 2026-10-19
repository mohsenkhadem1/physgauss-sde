"""Exact posterior coordinates using a parameter-dependent Gaussian bridge.

Frozen contact/diffusion geometry and a frozen RGB linearization define ONLY
a guide. At every material/noise/load scale, analytically condition the whole
Gaussian state path on that image surrogate. The full nonlinear physical
target is evaluated with the guide's triangular Jacobian. No approximate
likelihood, mechanics, force prior or linearized model becomes the target.
"""
import math
import torch
from physgauss.pac_integrals import covariance as integrated_covariance
from physgauss.pacnerf_model import PACRenderer
from pacnerf_discrepancy import DiscrepancyModel,DiscrepancyInnovation,spatial_whiten,DiscrepancyTarget


class PrecisionTailRenderer(PACRenderer):
    """Avoid exp/log on spatial tails below the float64 RGB error budget.

    Only no-gradient evaluation uses this path; reference derivatives use the
    original dense renderer. With 128 splats and opacity<=.7, tail/series error
    is below 1e-21 RGB, apart from ordinary float64 rounding (separately tested).
    Pixels are not subsampled; volumetric depth compositing is unchanged.
"""
    tail_alpha=2.**-80

    def __init__(self,source):self.__dict__.update(source.__dict__)

    def components(self,q,specs,pixel_indices=None):
        if pixel_indices is not None or torch.is_grad_enabled() or torch.autograd.forward_ad.unpack_dual(q).tangent is not None:
            return super().components(q,specs,pixel_indices)
        centers=self.data['gaussians'][None,None]+torch.einsum('btm,gcm->btgc',q,self.gaussian_basis)
        eyes,rotations,intrinsics=[torch.stack([self.data['cameras'][c][i] for _,c in specs]) for i in range(3)]
        camera=torch.einsum('tij,btgj->btgi',rotations,centers-eyes[None,:,None])
        depth=camera[...,2];safe=depth.clamp_min(.1)
        uv=camera[...,:2]/safe[...,None]*intrinsics[None,:,None,[0,1],[0,1]]+intrinsics[None,:,None,:2,2]-.5
        sd=self.radius*intrinsics[None,:,None,0,0]/safe
        # The isotropic screen Gaussian separates exactly into two 1D kernels:
        # 2*size exponentials rather than size**2 for every projected splat.
        variance=sd.square()+.3
        ex=-.5*(self.pixels[0,:,0][None,None,None,:]-uv[...,0,None]).square()/variance[...,None]
        ey=-.5*(self.pixels[:,0,1][None,None,None,:]-uv[...,1,None]).square()/variance[...,None]
        cutoff=math.log(self.tail_alpha)
        wx=torch.where(ex>cutoff,torch.exp(ex.clamp_min(cutoff)),0.)
        wy=torch.where(ey>cutoff,torch.exp(ey.clamp_min(cutoff)),0.)
        alpha=self.opacity*(wy[..., :,None]*wx[...,None,:]).flatten(-2)
        # Three log1p-series terms below 1e-6 omit <=2.51e-25 optical depth.
        # This is smaller than the existing renderer's expm1-series error.
        optical=alpha+alpha.square()*(.5+alpha/3)
        large=alpha>1e-6
        optical[large]=-torch.log1p(-alpha[large])
        cdf=.5*(1+torch.erf((self.boundaries[None,None,:,None]-depth[:,:,None])/(math.sqrt(2)*self.radius)))
        cdf=torch.cat((torch.zeros_like(cdf[:,:,:1]),cdf,torch.ones_like(cdf[:,:,:1])),2)
        return optical,cdf[:,:,1:]-cdf[:,:,:-1]


class CachedPhysicalModel(DiscrepancyModel):
    """Exact cache, used only without gradients and at identical parameters."""
    def __init__(self,source):
        self.__dict__.update(source.__dict__);self._integral_cache=None;self._transition_cache=None
        self.renderer=PrecisionTailRenderer(source.renderer)
        self._image_cache={}
        self._conditional_context=None

    def _check_conditional_parameters(self,z,hyper=False):
        context=self._conditional_context
        if not torch.equal(z[:,:2].detach(),context['theta']):raise ValueError('Conditional state update changed material parameters')
        if hyper and not torch.equal(z[:,-3:].detach(),context['hyper']):raise ValueError('Conditional state update changed noise parameters')
        return context

    def scales(self,z):
        if self._conditional_context is not None:return self._check_conditional_parameters(z,True)['scales']
        return super().scales(z)

    def specification(self):
        result=super().specification()
        result['no_grad_spatial_tail_alpha_cutoff']=self.renderer.tail_alpha
        result['tail_rgb_absolute_error_bound']=128*(2*self.renderer.tail_alpha+2.51e-25)/(1-.7)
        result['tail_policy']='separable full-grid Gaussian evaluation, tiny 1D tail/log-series error below float64 RGB budget; dense forward/reverse derivatives; no pixel subsampling or changed depth compositing; ordinary float64 rounding separately checked'
        return result

    def integration_operators(self,z):
        if self._conditional_context is not None:return self._check_conditional_parameters(z)['integrals']
        if torch.is_grad_enabled():return super().integration_operators(z)
        key=z[:,:2]
        if self._integral_cache is None or not torch.equal(key,self._integral_cache[0]):
            self._integral_cache=(key.clone(),super().integration_operators(z))
        return self._integral_cache[1]

    def transition(self,z,duration):
        if self._conditional_context is not None:
            if duration!=self.dt:raise ValueError('Conditional transition duration changed')
            return self._check_conditional_parameters(z)['transition']
        if torch.is_grad_enabled():return super().transition(z,duration)
        key=z[:,:2]
        if self._transition_cache is None or duration!=self._transition_cache[1] or not torch.equal(key,self._transition_cache[0]):
            self._transition_cache=(key.clone(),duration,super().transition(z,duration))
        return self._transition_cache[2]

    def image_residual(self,z):
        if torch.is_grad_enabled() or torch.autograd.forward_ad.unpack_dual(z).tangent is not None:
            return super().image_residual(z)
        q,_=self.decode(z);missing=[]
        for slot,(frame,camera) in enumerate(self.specs):
            cached=self._image_cache.get(slot)
            if cached is None or not torch.equal(cached[0],q[:,frame]):missing.append(slot)
        if missing:
            specs=[self.specs[i] for i in missing]
            rgb=self.renderer(q[:,[f for f,c in specs]],specs)
            raw=spatial_whiten(rgb-self.obs[missing][None],self.rho_x,self.rho_y)
            for j,slot in enumerate(missing):
                self._image_cache[slot]=(q[:,self.specs[slot][0]].clone(),raw[:,j].clone())
        error=torch.stack([self._image_cache[i][1] for i in range(len(self.specs))],1)
        return (error/self.scales(z)[2][:,None,None,None,None]).flatten(1)


class ConditionedTransport:
    def __init__(self,model,anchor,guide=None):
        self.model=model;self.dimension=model.dimension;self.modes=model.modes
        self.anchor=anchor.detach().clone()
        self.guide=self.build_guide() if guide is None else guide
        self._cache=None

    def build_guide(self):
        model=self.model;m=model.modes;s=model.base_dimension-2
        q,v=model.decode(self.anchor[None]);q=q[0].detach();v=v[0].detach()
        with torch.no_grad():
            x=model.data['rest'][None]+torch.einsum('tm,ncm->tnc',q[:-1],model.basis)
            velocity=torch.einsum('tm,ncm->tnc',v[:-1],model.basis)
            delta=x[:,model.edges[:,1]]-x[:,model.edges[:,0]]
            length=(delta.square().sum(-1)+1e-12).sqrt();direction=delta/length[...,None]
            speed=((velocity[:,model.edges[:,1]]-velocity[:,model.edges[:,0]])*direction).sum(-1)
            features=torch.stack((length/model.length-1,speed),-1)
            amp=model.network(features)/model.constant if model.noise=='neural' else torch.ones_like(speed)
            eb=model.basis[model.edges[:,1]]-model.basis[model.edges[:,0]]
            factor=torch.einsum('ecm,tec->tme',eb,direction)*amp[:,None,:]
            factor=factor*model.internal_scale/math.sqrt(len(model.edges)/len(model.data['rest']))
            spectral=factor@factor.transpose(-1,-2)+model.acc_noise_floor**2*torch.eye(m,dtype=q.dtype)
            nodal=torch.zeros_like(x);nodal[...,1]=-9.8+6*torch.nn.functional.softplus(-x[...,1]/.01)
            external=torch.einsum('tnc,ncm->tm',nodal,model.basis)
            contact=-600*torch.einsum('ni,tn,nj->tij',model.basis[:,1],torch.sigmoid(-x[...,1]/.01),model.basis[:,1])
            external=external-torch.einsum('tij,tj->ti',contact,q[:-1])
        following=q.new_zeros(13,2*m,s);previous=torch.zeros_like(following)
        scale=torch.tensor([.1]*m+[.5]*m,dtype=q.dtype)
        for time in range(13):
            following[time,:,m+time*2*m:m+(time+1)*2*m]=torch.diag(scale)
            if time==0:previous[time,m:,:m]=.5*torch.eye(m,dtype=q.dtype)
            else:previous[time,:,m+(time-1)*2*m:m+time*2*m]=torch.diag(scale)
        image_precision=q.new_zeros(s,s);image_rhs=q.new_zeros(s)
        for slot,spec in enumerate(model.specs):
            frame=spec[0]
            if frame==0:continue
            def image(pose):
                return spatial_whiten(model.renderer(pose[None,None],[spec])[0,0]-model.obs[slot],model.rho_x,model.rho_y).flatten()
            jac=torch.autograd.functional.jacobian(image,q[frame],vectorize=True,strategy='forward-mode').detach()
            residual=image(q[frame]).detach()
            target=jac@q[frame]-residual
            start=m+(frame-1)*2*m
            image_precision[start:start+m,start:start+m]+=.1**2*(jac.T@jac)
            image_rhs[start:start+m]+=.1*(jac.T@target)
        constraint_precision=torch.zeros_like(image_precision);constraint_rhs=torch.zeros_like(image_rhs)
        if hasattr(model,'floor_quadratic'):
            gradient,curvature=model.floor_quadratic(q[1:])
            for t in range(13):
                start=m+t*2*m
                constraint_precision[start:start+m,start:start+m]+=.1**2*curvature[t]
                constraint_rhs[start:start+m]+=.1*(curvature[t]@q[t+1]-gradient[t])
        return {'original_spectral_at_anchor':spectral,'external_at_anchor':external,'contact_derivative_at_anchor':contact,
            'following':following,'previous':previous,'image_precision':image_precision,'image_rhs':image_rhs,
            'constraint_precision':constraint_precision.detach(),'constraint_rhs':constraint_rhs.detach(),
            'anchor_modal_poses':q,'anchor_modal_velocities':v}

    def gaussian(self,z):
        model=self.model;m=model.modes;b=len(z);s=model.core_dimension-2
        if getattr(model,'_conditional_context',None) is not None:
            return model._check_conditional_parameters(z,True)['gaussian']
        key=torch.cat((z[:,:2],z[:,-3:]),-1)
        if not torch.is_grad_enabled() and self._cache is not None and torch.equal(key,self._cache[0]):
            return self._cache[1]
        transition=model.transition(z,model.dt);drive,kernel=model.integration_operators(z)
        diffusion,smooth,sigma=model.scales(z)
        spectral=self.guide['original_spectral_at_anchor'][None]+diffusion[:,None,None,None].square()*torch.eye(m,dtype=z.dtype)
        covariance=integrated_covariance(spectral,kernel)
        mapping=drive[:,None]@model.acceleration_mapping[None]
        factors=torch.cat((torch.ones(b,model.original_force_dimension,dtype=z.dtype),
            smooth[:,None].expand(-1,model.force_dimension-model.original_force_dimension)),-1)
        mapping=mapping*factors[:,None,None,:]
        contact=drive[:,None]@self.guide['contact_derivative_at_anchor'][None]
        linear_transition=transition[:,None]+torch.cat((contact,torch.zeros_like(contact)),-1)
        residual_matrix=self.guide['following'][None]-linear_transition@self.guide['previous'][None]
        external=(drive[:,None]@self.guide['external_at_anchor'][None,:,:,None]).squeeze(-1)
        if model.force=='joint':residual_matrix=torch.cat((residual_matrix,-mapping),-1)
        chol_q=torch.linalg.cholesky(covariance)
        wb=torch.linalg.solve_triangular(chol_q,residual_matrix,upper=False)
        wr=torch.linalg.solve_triangular(chol_q,external[...,None],upper=False).squeeze(-1)
        precision=torch.einsum('btdi,btdj->bij',wb,wb)
        rhs=torch.einsum('btdi,btd->bi',wb,wr)
        if model.force=='marginal':
            wg=torch.linalg.solve_triangular(chol_q,mapping,upper=False)
            a=torch.eye(model.force_dimension,dtype=z.dtype)+torch.einsum('btdk,btdl->bkl',wg,wg)
            root=torch.linalg.cholesky(a)
            cross=torch.einsum('btdi,btdk->bik',wb,wg)
            arhs=torch.einsum('btdk,btd->bk',wg,wr)
            precision=precision-cross@torch.cholesky_solve(cross.transpose(-1,-2),root)
            rhs=rhs-(cross@torch.cholesky_solve(arhs[...,None],root)).squeeze(-1)
        diagonal=z.new_zeros(s);diagonal[:m]=1
        if model.force=='joint':diagonal[model.base_dimension-2:]=1
        precision=precision+torch.diag(diagonal)[None]
        # Fixed local image surrogate, with the current sampled RGB scale.
        pad=s-(model.base_dimension-2)
        ip=torch.nn.functional.pad(self.guide['image_precision'],(0,pad,0,pad))
        ir=torch.nn.functional.pad(self.guide['image_rhs'],(0,pad))
        precision=precision+ip[None]/sigma[:,None,None].square()
        rhs=rhs+ir[None]/sigma[:,None].square()
        if 'constraint_precision' in self.guide:
            precision=precision+torch.nn.functional.pad(self.guide['constraint_precision'],(0,pad,0,pad))[None]
            rhs=rhs+torch.nn.functional.pad(self.guide['constraint_rhs'],(0,pad))[None]
        precision=(precision+precision.transpose(-1,-2))/2
        # Proposal-only numerical ridge; the exact Jacobian corrects it too.
        ridge=1e-6+precision.diagonal(dim1=-2,dim2=-1).abs().amax(-1)*1e-10
        precision=precision+ridge[:,None,None]*torch.eye(s,dtype=z.dtype)
        root=torch.linalg.cholesky(precision)
        mean=torch.cholesky_solve(rhs[...,None],root).squeeze(-1)
        result=(mean,root)
        if not torch.is_grad_enabled():self._cache=(key.clone(),result)
        return result

    def transform(self,z,inverse=False):
        mean,root=self.gaussian(z)
        if inverse:
            state=(root.transpose(-1,-2)@(z[:,2:-3]-mean)[...,None]).squeeze(-1)
        else:
            state=mean+torch.linalg.solve_triangular(root.transpose(-1,-2),z[:,2:-3,None],upper=True).squeeze(-1)
        physical=torch.cat((z[:,:2],state,z[:,-3:]),-1)
        log_jac=-root.diagonal(dim1=-2,dim2=-1).log().sum(-1)
        return physical,log_jac

    def to_physical(self,z):return self.transform(z)[0]

    def objective(self,z):
        physical,jac=self.transform(z)
        return self.model.objective(physical)-jac


def make_target(model,reference):
    guide=ConditionedTransport(model,reference['anchor'],reference['guide'])
    mean=reference.get('sampling_mean',reference['mean'])
    precision=reference.get('sampling_precision',reference['precision'])
    target=DiscrepancyTarget(guide,mean,precision)
    if reference.get('parameter_blocked_reference'):
        from pacnerf_hybrid import conditional_reference_transform,ConditionalStateEvaluator
        target.transform=conditional_reference_transform(precision,model.core_dimension)
        target.state_target=lambda fixed:ConditionalStateEvaluator(target,fixed,model.core_dimension)
    return target
