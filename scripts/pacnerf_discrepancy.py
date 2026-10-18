"""A separately versioned hierarchical model; never changes frozen old targets.

Three N(0,1) hyperparameters control additional full-rank acceleration diffusion,
smooth modal acceleration discrepancy, and RGB discrepancy. Their dimensional
prior centers follow gravity, frame duration and the supplied rest geometry,
NOT evaluation trajectories. The Gaussian load prior remains identical between
joint and analytically marginalized arms. RGB residuals have a normalized,
spatially correlated Gaussian density, including its learned-scale determinant.
"""
import math
import torch
from physgauss.pacnerf_model import PACModel
from physgauss.pac_integrals import covariance as integrated_covariance
from pacnerf_innovations import InnovationModel
from pacnerf_comparison import Target, make_model as old_make_model


def spatial_whiten(residual, rho_x, rho_y):
    """Inverse Cholesky for separable unit-diagonal AR(1) image covariance."""
    x=torch.cat((residual[...,:1,:],(residual[...,1:,:]-rho_x*residual[...,:-1,:])/math.sqrt(1-rho_x*rho_x)),-2)
    return torch.cat((x[...,:1,:,:],(x[...,1:,:,:]-rho_y*x[...,:-1,:,:])/math.sqrt(1-rho_y*rho_y)),-3)


def initial_residual_correlation(model):
    """Initial appearance diagnostics only; no later or withheld image access."""
    specs=[(0,c) for c in range(11)]
    with torch.no_grad():
        prediction=model.renderer(torch.zeros(1,11,model.modes,dtype=torch.float64),specs)[0]
        observed=torch.stack([model.data['images'][c,0] for c in range(11)])
        e=prediction-observed
        def corr(a,b):
            a=a.flatten()-a.mean();b=b.flatten()-b.mean()
            return min(.95,max(0.,float((a@b)/((a@a)*(b@b)).clamp_min(1e-30).sqrt())))
        return corr(e[:, :,1:],e[:,:,:-1]),corr(e[:,1:],e[:,:-1])


class DiscrepancyModel(PACModel):
    def __init__(self,base):
        self.__dict__.update(base.__dict__)
        self.original_force_dimension=base.force_dimension
        self.rho_x,self.rho_y=initial_residual_correlation(base)
        # A modal acceleration with independent equal-variance coefficients has
        # average particle-coordinate variance rms_projection**2. Normalize its
        # units so prior centers mean RMS acceleration over rendered coordinates.
        self.rms_projection=float(self.renderer.gaussian_basis.square().sum(-1).mean().sqrt())
        self.diffusion_center=9.8*math.sqrt(self.dt)/self.rms_projection
        self.smooth_acceleration_center=(9.8/3)/self.rms_projection
        times=self.data['times'][:-1]
        knots=torch.linspace(0,13/24,4,dtype=torch.float64)
        temporal=torch.exp(-.5*((times[:,None]-knots)/.14).square())
        temporal=temporal/temporal.square().sum(-1,keepdim=True).sqrt()
        extra=torch.einsum('tk,mn->tmkn',temporal,torch.eye(self.modes,dtype=torch.float64)).flatten(2)
        self.acceleration_mapping=torch.cat((base.acceleration_mapping,extra*self.smooth_acceleration_center),-1)
        self.force_dimension=self.acceleration_mapping.shape[-1]
        self.core_dimension=self.base_dimension+(self.force_dimension if self.force=='joint' else 0)
        self.dimension=self.core_dimension+3

    def scales(self,z):
        h=z[:,-3:]
        diffusion=self.diffusion_center*torch.exp(h[:,0])
        smooth=torch.exp(h[:,1])
        # A fixed .08 observation component plus an uncertain renderer-error
        # component. This cannot fit a zero noise scale to empty backgrounds.
        rgb=.08*torch.sqrt(1+torch.exp(h[:,2]))
        return diffusion,smooth,rgb

    def terms(self,z):
        residual,covariance,mapping,features=super().terms(z)
        diffusion,smooth,_=self.scales(z)
        _,kernel=self.integration_operators(z)
        unit=torch.eye(self.modes,dtype=z.dtype)[None,None].expand(len(z),1,-1,-1)
        covariance=covariance+diffusion[:,None,None,None].square()*integrated_covariance(unit,kernel)
        factors=torch.cat((torch.ones(len(z),self.original_force_dimension,dtype=z.dtype),
                           smooth[:,None].expand(-1,self.force_dimension-self.original_force_dimension)),-1)
        return residual,covariance,mapping*factors[:,None,None,:],features

    def dynamics(self,z):
        residual,covariance,mapping,_=self.terms(z)
        chol=torch.linalg.cholesky(covariance)
        g=torch.linalg.solve_triangular(chol,mapping,upper=False)
        r=torch.linalg.solve_triangular(chol,residual[...,None],upper=False).squeeze(-1)
        logdet=2*chol.diagonal(dim1=-2,dim2=-1).log().sum((-1,-2))
        n=residual.shape[1]*residual.shape[2]
        if self.force=='joint':
            a=z[:,self.base_dimension:self.core_dimension]
            square=(r-torch.einsum('btdk,bk->btd',g,a)).square().sum((-1,-2))+a.square().sum(-1)
            n+=self.force_dimension
        else:
            precision=torch.eye(self.force_dimension,dtype=z.dtype)+torch.einsum('btdk,btdl->bkl',g,g)
            chol_a=torch.linalg.cholesky(precision)
            rhs=torch.einsum('btdk,btd->bk',g,r)
            a=torch.cholesky_solve(rhs[...,None],chol_a).squeeze(-1)
            square=(r-torch.einsum('btdk,bk->btd',g,a)).square().sum((-1,-2))+a.square().sum(-1)
            logdet=logdet+2*chol_a.diagonal(dim1=-2,dim2=-1).log().sum(-1)
        prior=.5*z[:,:2+self.modes].square().sum(-1)+.5*z[:,-3:].square().sum(-1)
        return .5*(square+logdet+n*math.log(2*math.pi))+prior-self.scale[2+self.modes:].log().sum()

    def image_residual(self,z):
        q,_=self.decode(z)
        error=self.renderer(q[:,self.frames],self.specs)-self.obs[None]
        sigma=self.scales(z)[2]
        return (spatial_whiten(error,self.rho_x,self.rho_y)/sigma[:,None,None,None,None]).flatten(1)

    def image_normalizer(self,z):
        size=self.data['size'];n=len(self.specs)*size*size*3
        logdet=len(self.specs)*3*(size*(size-1)*math.log(1-self.rho_x**2)+size*(size-1)*math.log(1-self.rho_y**2))
        return n*torch.log(self.scales(z)[2])+.5*logdet+.5*n*math.log(2*math.pi)

    def objective(self,z):
        return self.dynamics(z)+.5*self.image_residual(z).square().sum(-1)+self.image_normalizer(z)

    def specification(self):
        return {'version':'hierarchical-discrepancy-v1','latent_dimension':self.dimension,
            'force_dimension':self.force_dimension,'original_force_dimension':self.original_force_dimension,
            'hyper_prior':'three independent standard normals; no plug-in scale fitting',
            'diffusion_modal_center':self.diffusion_center,'diffusion_coordinate_rms_center':9.8*math.sqrt(self.dt),
            'smooth_acceleration_modal_center':self.smooth_acceleration_center,'smooth_acceleration_coordinate_rms_center':9.8/3,
            'rgb_sigma':'.08 * sqrt(1 + exp(h_rgb))','rgb_spatial_correlation':[self.rho_x,self.rho_y],
            'correlation_fit_data':'eleven supplied frame-zero images and supplied frozen appearance only',
            'smooth_discrepancy':'four normalized temporal RBFs of width .14s in each of twelve modal accelerations',
            'remaining_limits':'twelve-mode approximation, supplied geometry/appearance, frame-held contact; excludes uncertainty outside the modal subspace; not guaranteed calibrated'}


class ConditionalInnovation(InnovationModel):
    def __init__(self,model,hyper):
        super().__init__(model);self.hyper=hyper

    def operators(self,z):
        transition,drive,kernel,mapping=super().operators(z)
        factors=torch.cat((torch.ones(len(z),self.model.original_force_dimension,dtype=z.dtype),
            torch.exp(self.hyper[:,1:2]).expand(-1,self.model.force_dimension-self.model.original_force_dimension)),-1)
        unit=torch.eye(self.modes,dtype=z.dtype)[None,None].expand(len(z),1,-1,-1)
        self.extra_covariance=integrated_covariance(unit,kernel)[:,0]*(self.model.diffusion_center*torch.exp(self.hyper[:,0]))[:,None,None].square()
        return transition,drive,kernel,mapping*factors[:,None,None,:]

    def step(self,*args):
        mean,covariance=super().step(*args)
        return mean,covariance+self.extra_covariance

    def transform(self,z,inverse=False):
        if self.model.force=='joint':return super().transform(z,inverse)
        # Covariance-form Gaussian filtering uses only 24x24 innovation
        # Cholesky factors, rather than thirteen 72x72 precision factorizations.
        # Its physical target and triangular Jacobian match the precision form.
        model=self.model;m=model.modes;b=len(z);k=model.force_dimension
        transition,drive,kernel,mapping=self.operators(z)
        if inverse:
            q,v=model.decode(z);supplied=torch.cat((q[:,1:],v[:,1:]),-1)
        else:supplied=z[:,2+m:model.base_dimension].reshape(b,13,2*m)
        current=torch.cat((z.new_zeros(b,m),.5*z[:,2:2+m]),-1)
        covariance_a=torch.eye(k,dtype=z.dtype).expand(b,k,k).clone();mean_a=z.new_zeros(b,k)
        converted=[];log_jac=z.new_zeros(b)
        for time in range(13):
            external,covariance=self.step(current[:,:m],current[:,m:],drive,kernel)
            base=(transition@current[...,None]).squeeze(-1)+external
            g=mapping[:,time];gs=g@covariance_a
            mean=base+(g@mean_a[...,None]).squeeze(-1)
            conditional=covariance+gs@g.transpose(-1,-2)
            chol=torch.linalg.cholesky((conditional+conditional.transpose(-1,-2))/2)
            log_jac+=chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)
            if inverse:
                following=supplied[:,time]
                standardized=torch.linalg.solve_triangular(chol,(following-mean)[...,None],upper=False).squeeze(-1)
                converted.append(standardized)
            else:
                standardized=supplied[:,time]
                following=mean+(chol@standardized[...,None]).squeeze(-1)
                converted.append(following)
            whitened_gs=torch.linalg.solve_triangular(chol,gs,upper=False)
            mean_a=mean_a+(whitened_gs.transpose(-1,-2)@standardized[...,None]).squeeze(-1)
            covariance_a=covariance_a-whitened_gs.transpose(-1,-2)@whitened_gs
            covariance_a=(covariance_a+covariance_a.transpose(-1,-2))/2
            current=following
        middle=torch.stack(converted,1).flatten(1)
        if not inverse:middle=middle/model.scale[2+m:model.base_dimension]
        result=torch.cat((z[:,:2+m],middle),-1)
        return result,log_jac-model.scale[2+m:model.base_dimension].log().sum()


class DiscrepancyInnovation:
    def __init__(self,model):
        self.model=model;self.dimension=model.dimension;self.modes=model.modes

    def transform(self,z,inverse=False):
        physical,jac=ConditionalInnovation(self.model,z[:,-3:]).transform(z[:,:-3],inverse)
        return torch.cat((physical,z[:,-3:]),-1),jac

    def to_physical(self,z):return self.transform(z)[0]

    def objective(self,z):
        physical=self.to_physical(z)
        n=self.model.core_dimension-(2+self.modes)
        return (.5*z.square().sum(-1)+.5*self.model.image_residual(physical).square().sum(-1)
                +self.model.image_normalizer(physical)+.5*n*math.log(2*math.pi))


class DiscrepancyTarget(Target):
    @torch.no_grad()
    def decode(self,w):
        return torch.cat([self.model.to_physical(self.coordinates(batch)) for batch in w.split(32)])


def make_model(inputs,index):
    old,bundle=old_make_model(inputs,index)
    return DiscrepancyModel(old),bundle


def make_target(model,reference):
    return DiscrepancyTarget(DiscrepancyInnovation(model),reference['mean'],reference['precision'])


def gauss_newton(innovation,mean):
    """Exact full-residual first derivative; fixed proposal metric only.

Chain rule via observed poses avoids a 400-direction full trajectory renderer.
The learned RGB scale derivative is included, even at the fixed initial pose.
"""
    from pacnerf_innovation_metric import pose_jacobian
    model=innovation.model;m=model.modes
    poses,jac=pose_jacobian(innovation,mean)
    physical=innovation.to_physical(mean[None])
    q,_=model.decode(physical)
    sigma=model.scales(mean[None])[2][0].detach()
    derivative=.5*torch.sigmoid(mean[-1]).detach()
    precision=torch.eye(len(mean),dtype=mean.dtype)
    dynamic_slot=0
    for slot,spec in enumerate(model.specs):
        def residual(pose):
            error=model.renderer(pose[None,None],[spec])[0,0]-model.obs[slot]
            return (spatial_whiten(error,model.rho_x,model.rho_y)/sigma).flatten()
        value=residual(q[0,spec[0]].detach()).detach()
        local=mean.new_zeros(len(value),len(mean))
        if spec[0]:
            image_jac=torch.autograd.functional.jacobian(residual,poses[dynamic_slot],vectorize=True,strategy='forward-mode').detach()
            local=image_jac@jac[dynamic_slot*m:(dynamic_slot+1)*m]
            dynamic_slot+=1
        local[:,-1]-=value*derivative
        precision+=local.T@local
    precision=(precision+precision.T)/2
    return precision,{'metric':'I + full correlated, scale-dependent RGB residual Jacobian Gram; proposal only',
        'minimum_eigenvalue':float(torch.linalg.eigvalsh(precision)[0])}
