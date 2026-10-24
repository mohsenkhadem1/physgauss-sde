"""Exact grouped reconstruction of the same weighted full static field.

Conditioning on r=K/E makes the full displacement linear in a=1/E. Thus
the conditional amplitude mean suffices; no physical parameter interpolation
or replacement by a single mean material is involved.
"""
import numpy as np
from experiments.campaign_surface_posterior_field_v01 import PosteriorMeanField


class GroupedPosteriorMeanField(PosteriorMeanField):
    def mean_grouped(self,grouped,*,chunk=256):
        z=np.asarray(grouped['log_ratio'],float)
        p=np.asarray(grouped['probability'],float)
        a=np.asarray(grouped['amplitude_mean'],float)
        if z.ndim!=1 or not len(z) or p.shape!=z.shape or a.shape!=z.shape:
            raise ValueError('Common nonempty conditional ratio groups required')
        if not all(np.isfinite(x).all() for x in(z,p,a))or np.any(p<0)or np.any(a<=0)or abs(p.sum()-1)>1e-12:
            raise ValueError('Normalized positive-amplitude grouped posterior required')
        if type(chunk)is not int or chunk<1:raise ValueError('Positive bounded chunk required')
        modal_mean=np.zeros_like(self.eigenvalues)
        for first in range(0,len(z),chunk):
            ratio=np.exp(z[first:first+chunk])
            denominator=1+ratio[:,None]*self.eigenvalues
            if not np.isfinite(denominator).all()or np.any(denominator<=0):
                raise FloatingPointError('Unmodified modal denominator must be positive')
            modal_mean+=(p[first:first+chunk]*a[first:first+chunk])@(1/denominator)
        boundary=self.V@(modal_mean[:,None]*self.projected_force)
        field=np.empty(self.shape)
        field[self.boundary]=boundary
        field[self.interior]=self.particular*float(p@a)-self.M@boundary
        if not np.isfinite(field).all():raise FloatingPointError('Nonfinite grouped mean field')
        return field.T
