"""Untrained interface prototype: OUR ConditionalFlow -> existing Isaac teacher.

A global25D coefficient posterior with sampled4-way boundary/force categories.
This is not the current six-parameter FEM field, spatial scene discovery, a
retrained old normalizing flow, or a physically validated forward simulator.
No file/dataset access; all inputs are already permitted five-frame features.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from scripts.gaussian_boundary_flow.flow import ConditionalFlow
from experiments.gaussian_observed_features import ObservedFeatureEncoder
from experiments.gaussian_operator import PhysicalParameterTransform,canonicalize_physical


class IsaacConditionalFlowAdapter(nn.Module):
    def __init__(self,width=160):
        super().__init__()
        self.encoder=ObservedFeatureEncoder(119,128,64,'features')
        self.categories=nn.Linear(128,8)
        self.transform=PhysicalParameterTransform()
        self.register_buffer("logit_mean",torch.zeros(25))
        self.register_buffer("logit_scale",torch.ones(25))
        self.flow=ConditionalFlow(context_dim=136,parameter_dim=25,width=width)

    def set_standardization(self,mean,scale):
        if mean.shape!=(25,) or scale.shape!=(25,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale>0).all():
            raise ValueError("Finite25D mean and positive scale required")
        self.logit_mean.copy_(mean);self.logit_scale.copy_(scale)

    def context(self,features):
        if features.ndim!=3 or features.shape[1:]!=(5,119) or not torch.isfinite(features).all():
            raise ValueError('Exactly five finite119-feature RGB-D prefix rows required')
        return self.encoder(features)

    @staticmethod
    def conditional_context(context,boundary,force):
        return torch.cat([context,F.one_hot(boundary,4).to(context),F.one_hot(force,4).to(context)],-1)

    def loss(self,features,physical,boundary,force,category_weight=3.):
        """TRAIN labels only; caller applies declared training-label tolerance.

        Validation/test inference never accepts these labels. Flow matching loss
        is not a likelihood and must not be compared as NLL to old flow models.
        """
        if not math.isfinite(category_weight) or category_weight<=0:raise ValueError("Positive finite category weight required")
        context=self.context(features);logits=self.categories(context)
        target=self.transform.physical_to_unconstrained(physical,strict=True)
        target=(target-self.logit_mean)/self.logit_scale
        loss=self.flow.loss(target,self.conditional_context(context,boundary,force))
        return loss+category_weight*(F.cross_entropy(logits[:,:4],boundary)+F.cross_entropy(logits[:,4:],force))

    @torch.no_grad()
    def sample(self,features,samples=32,steps=32,generator=None):
        if type(samples)is not int or samples<1:raise ValueError('Positive integer sample count required')
        if type(steps)is not int or steps<1:raise ValueError("Positive integer integration steps required")
        context=self.context(features);logits=self.categories(context);batch=len(context)
        if not torch.isfinite(logits).all():raise ValueError("Nonfinite category logits")
        boundary=torch.multinomial(logits[:,:4].softmax(-1),samples,replacement=True,generator=generator)
        force=torch.multinomial(logits[:,4:].softmax(-1),samples,replacement=True,generator=generator)
        conditioned=self.conditional_context(context.repeat_interleave(samples,0),boundary.flatten(),force.flatten())
        z=self.flow.sample(conditioned,count=1,steps=steps,generator=generator).reshape(batch,samples,25)
        if not torch.isfinite(z).all() or not torch.isfinite(logits).all():raise ValueError("Nonfinite posterior output")
        z=z*self.logit_scale+self.logit_mean
        if not torch.isfinite(z).all():raise ValueError("Nonfinite affine logit output")
        physical=self.transform.unconstrained_to_physical(z)
        return dict(physical=physical,effective_physical=canonicalize_physical(physical,boundary,force),
                    boundary_type=boundary,force_type=force)


class IsaacDiagonalMixtureAdapter(IsaacConditionalFlowAdapter):
    """Four-component diagonal Gaussian on the SAME standardized25D logits."""
    def __init__(self,width=200):
        super().__init__();del self.flow
        self.mixture=nn.Sequential(nn.Linear(136,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,204))

    def distribution(self,context):
        output=self.mixture(context)
        if not torch.isfinite(output).all():raise ValueError('Nonfinite mixture output')
        weight=output[:,:4].log_softmax(-1)
        mean=output[:,4:104].reshape(-1,4,25)
        logstd=output[:,104:].reshape(-1,4,25).clamp(-5,3)
        return weight,mean,logstd

    def loss(self,features,physical,boundary,force,category_weight=3.):
        if not math.isfinite(category_weight) or category_weight<=0:raise ValueError('Positive finite category weight required')
        context=self.context(features);logits=self.categories(context)
        target=(self.transform.physical_to_unconstrained(physical,strict=True)-self.logit_mean)/self.logit_scale
        weight,mean,logstd=self.distribution(self.conditional_context(context,boundary,force))
        component=-.5*(((target[:,None]-mean)*torch.exp(-logstd)).square()+2*logstd+math.log(2*math.pi)).sum(-1)
        # Per-coordinate NLL gives category CE the same declared nominal weight.
        nll=-torch.logsumexp(weight+component,-1).mean()/25
        return nll+category_weight*(F.cross_entropy(logits[:,:4],boundary)+F.cross_entropy(logits[:,4:],force))

    @torch.no_grad()
    def sample(self,features,samples=32,steps=32,generator=None):
        if type(samples)is not int or samples<1:raise ValueError('Positive integer sample count required')
        if type(steps)is not int or steps<1:raise ValueError('Positive integer integration steps required')
        context=self.context(features);logits=self.categories(context);batch=len(context)
        if not torch.isfinite(logits).all():raise ValueError("Nonfinite category logits")
        boundary=torch.multinomial(logits[:,:4].softmax(-1),samples,replacement=True,generator=generator)
        force=torch.multinomial(logits[:,4:].softmax(-1),samples,replacement=True,generator=generator)
        conditioned=self.conditional_context(context.repeat_interleave(samples,0),boundary.flatten(),force.flatten())
        weight,mean,logstd=self.distribution(conditioned)
        component=torch.multinomial(weight.exp(),1,generator=generator).squeeze(-1)
        index=torch.arange(len(component),device=component.device)
        z=mean[index,component]+logstd[index,component].exp()*torch.randn(len(component),25,device=context.device,generator=generator)
        z=z.reshape(batch,samples,25)*self.logit_scale+self.logit_mean
        if not torch.isfinite(z).all():raise ValueError('Nonfinite posterior output')
        physical=self.transform.unconstrained_to_physical(z)
        return dict(physical=physical,effective_physical=canonicalize_physical(physical,boundary,force),boundary_type=boundary,force_type=force)
