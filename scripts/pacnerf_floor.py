"""Known opaque floor and explicit physical-feasibility conditioning.

The event likelihood P(F=1|path)=prod sigmoid(height/0.002) is an explicit
relaxation of the known non-penetration constraint, not learned measurement
noise or a renormalized transition. Conditioning the original joint prior on
this event keeps Gaussian forces conjugate: the event depends on states only.
"""
import math
import torch
from pacnerf_conditioned_transport import CachedPhysicalModel
from pacnerf_discrepancy import DiscrepancyModel,DiscrepancyInnovation
from pacnerf_comparison import make_model as make_base_model


class FloorRenderer:
    """Clip depth emission at the camera and the known opaque y=0 plane.

    Each pixel has only one partially visible depth slab. Compute that slab
    directly rather than allocating batch*time*slab*splat*pixel fractions.
    """
    def __init__(self,source):
        self.__dict__.update(source.__dict__);self.source=source;self._floor_cache={}

    @property
    def tail_alpha(self):return self.source.tail_alpha

    def geometry(self,q,specs,pixel_indices=None):
        centers=self.data['gaussians'][None,None]+torch.einsum('btm,gcm->btgc',q,self.gaussian_basis)
        eyes,rotations,intrinsics=[torch.stack([self.data['cameras'][c][i] for _,c in specs]) for i in range(3)]
        depth=torch.einsum('tij,btgj->btgi',rotations,centers-eyes[None,:,None])[...,2]
        key=tuple(c for _,c in specs)
        if pixel_indices is None and key in self._floor_cache:
            return depth,*self._floor_cache[key]
        pixels=self.pixels.flatten(0,1)
        pixels=pixels[None].expand(len(specs),-1,-1) if pixel_indices is None else pixels[pixel_indices]
        ray=(pixels+.5-intrinsics[:,None,:2,2])/intrinsics[:,None,[0,1],[0,1]]
        ray=torch.cat((ray,torch.ones_like(ray[...,:1])),-1)
        dy=(ray*rotations[:,:,1][:,None]).sum(-1)
        if bool((eyes[:,1]<=0).any()):raise ValueError('Floor compositor requires cameras above y=0')
        candidate=-eyes[:,1,None]/torch.where(dy<0,dy,torch.ones_like(dy))
        ground=torch.where((dy<0)&(candidate>0),candidate,torch.full_like(candidate,torch.inf))
        slab=torch.searchsorted(self.boundaries,ground.contiguous())
        if pixel_indices is None:self._floor_cache[key]=(ground,slab)
        return depth,ground,slab

    def __call__(self,q,specs,colors=None,pixel_indices=None):
        optical,fractions=self.source.components(q,specs,pixel_indices)
        depth,ground,slab=self.geometry(q,specs,pixel_indices)
        cdf=torch.cat((torch.zeros_like(fractions[:,:,:1]),fractions.cumsum(2)),2)
        near=.5*(1+torch.erf(-depth/(math.sqrt(2)*self.radius)))
        cdf=torch.maximum(cdf,near[:,:,None]);fractions=cdf[:,:,1:]-cdf[:,:,:-1]
        # Avoid infinities in second derivatives of erf at rays missing ground.
        safe_ground=torch.where(torch.isfinite(ground),ground,torch.full_like(ground,1e6))
        limit=.5*(1+torch.erf((safe_ground[None,:,None]-depth[...,None])/(math.sqrt(2)*self.radius)))
        index=slab[None,:,None].expand(len(q),-1,len(self.data['gaussians']),-1)
        lower=torch.gather(cdf.transpose(-1,-2),-1,index)
        partial=(limit-lower).clamp_min(0)
        colors=self.colors if colors is None else colors
        extinction=torch.einsum('btsg,btgp->btsp',fractions,optical)
        source=torch.einsum('btsg,btgp,gc->btspc',fractions,optical,colors)
        ordinal=torch.arange(fractions.shape[2])[None,:,None]
        complete=(ordinal<slab[:,None]);at_floor=(ordinal==slab[:,None])
        clipped=optical*partial
        extinction=extinction*complete[None]+clipped.sum(2)[:,:,None]*at_floor[None]
        source=source*complete[None,...,None]+torch.einsum('btgp,gc->btpc',clipped,colors)[:,:,None]*at_floor[None,...,None]
        before=extinction.cumsum(2)-extinction;small=extinction<1e-3
        series=1+extinction*(-.5+extinction*(1/6+extinction*(-1/24+extinction/120)))
        safe=extinction.clamp_min(1e-3)
        phi=torch.where(small,series,-torch.expm1(-safe)/safe)
        bg=torch.stack([self.data['backgrounds'][c].flatten(0,1) for _,c in specs])
        if pixel_indices is not None:bg=torch.gather(bg,1,pixel_indices[...,None].expand(-1,-1,3))
        rgb=(torch.exp(-before)[...,None]*phi[...,None]*source).sum(2)+torch.exp(-extinction.sum(2))[...,None]*bg[None]
        return rgb.reshape(*rgb.shape[:2],self.data['size'],self.data['size'],3) if pixel_indices is None else rgb


class FloorConditionedModel(CachedPhysicalModel):
    floor_tolerance=.002

    def __init__(self,source):
        super().__init__(source);self.renderer=FloorRenderer(self.renderer)
        self.floor_basis=torch.cat((self.basis[:,1],self.renderer.gaussian_basis[:,1]))
        self.floor_rest=torch.cat((self.data['rest'][:,1],self.data['gaussians'][:,1]))

    def floor_heights(self,z):
        q,_=self.decode(z)
        return self.floor_rest[None,None]+torch.einsum('btm,nm->btn',q[:,1:],self.floor_basis)

    def floor_penalty(self,z):
        return torch.nn.functional.softplus(-self.floor_heights(z)/self.floor_tolerance).sum((1,2))

    def objective(self,z):return super().objective(z)+self.floor_penalty(z)

    def floor_quadratic(self,q):
        height=self.floor_rest[None]+q@self.floor_basis.T
        probability=torch.sigmoid(-height/self.floor_tolerance)
        gradient=-torch.einsum('tn,nm->tm',probability,self.floor_basis)/self.floor_tolerance
        curvature=torch.einsum('nm,tn,nk->tmk',self.floor_basis,probability*(1-probability),self.floor_basis)/self.floor_tolerance**2
        return gradient,curvature

    def specification(self):
        result=super().specification();result.update({'version':'floor-conditioned-hierarchical-v2',
            'floor_y_m':0.,'opaque_floor_in_renderer':True,'camera_near_depth_m':0.,
            'floor_feasibility_event':'P(F=1|path)=exp(-sum softplus(-height/.002)); all later handle and rendering-center heights',
            'floor_tolerance_m':self.floor_tolerance,
            'prior_scope':'Original normalized SDE/mechanics/load priors BEFORE conditioning on the known physical-feasibility event; no omitted parameter-specific truncated-transition normalizer.',
            'force_conjugacy':'Floor event and visibility depend only on the path; same Gaussian force prior remains exactly integrable.'})
        return result


class FloorInnovation(DiscrepancyInnovation):
    def objective(self,z):
        physical=self.to_physical(z);n=self.model.core_dimension-(2+self.modes)
        return (.5*z.square().sum(-1)+.5*self.model.image_residual(physical).square().sum(-1)
            +self.model.image_normalizer(physical)+.5*n*math.log(2*math.pi)+self.model.floor_penalty(physical))


def make_floor_model(inputs,index):
    base,bundle=make_base_model(inputs,index)
    base.renderer=FloorRenderer(base.renderer)
    return FloorConditionedModel(DiscrepancyModel(base)),bundle
