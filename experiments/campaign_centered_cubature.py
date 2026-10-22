"""Centered normalized moments with COMPONENTWISE cubature-error diagnostics.

These error estimates are heuristic numerical diagnostics, not rigorous bounds.
No scientific noise or posterior target changes; center and numerical units are
fixed before either tolerance pass. Only roundoff-scale negative variance may
be zeroed, and that event is reported.
"""
import numpy as np
from scipy.integrate import cubature


def normalize_moments(value,error,center,scale):
    value,error,center=np.asarray(value,dtype=float),np.asarray(error,dtype=float),np.asarray(center,dtype=float)
    dimension=len(center)
    if value.shape!=(1+2*dimension,) or error.shape!=value.shape or not np.isfinite(value).all() or not np.isfinite(error).all() or (error<0).any() or not np.isfinite(scale) or scale<=0:raise ValueError('Finite moment/error vectors required')
    Z=value[0];eZ=error[0]
    if not Z>eZ:raise ValueError('Cubature mass estimate not resolved relative to error')
    first=value[1:1+dimension]/Z;second=value[1+dimension:]/Z
    efirst=(error[1:1+dimension]+abs(first)*eZ)/(Z-eZ)
    esecond=(error[1+dimension:]+abs(second)*eZ)/(Z-eZ)
    variance=second-first**2;evariance=esecond+2*abs(first)*efirst+efirst**2
    roundoff=64*np.finfo(float).eps*(abs(second)+first**2+np.finfo(float).tiny)
    if np.any(variance < -roundoff):raise ValueError('Materially negative centered variance; no repair')
    clamped=variance<0;variance=np.maximum(variance,0)
    std=scale*np.sqrt(variance);lo=scale*np.sqrt(np.maximum(variance-evariance,0));hi=scale*np.sqrt(variance+evariance)
    return dict(Z_scaled=Z,Z_estimated_absolute_error=eZ,Z_estimated_relative_error=eZ/Z,
        fixed_center_m=center,scale_m=scale,integral_vector=value,component_error_estimates=error,
        normalized_centered_first=first,normalized_centered_second=second,estimated_variance_error_scaled=evariance,
        held_mean_m=center+scale*first,held_std_m=std,estimated_mean_error_m=scale*efirst,
        estimated_std_bracket_m=np.stack([lo,hi],axis=1),estimated_std_error_m=np.maximum(std-lo,hi-std),
        roundoff_variance_clamped=clamped,error_estimates_are_NOT_rigorous_bounds=True)


def integrate_centered(density,predict,center,scale,*,rtol,atol):
    center=np.asarray(center,dtype=float)
    def integrand(x):
        w=np.asarray(density(x),dtype=float);delta=(predict(x)-center)/scale
        return np.column_stack([w,w[:,None]*delta,w[:,None]*delta**2])
    result=cubature(integrand,np.zeros(2),np.ones(2),rule='gk21',rtol=rtol,atol=atol,max_subdivisions=512)
    if result.status!='converged':raise RuntimeError('Centered vector cubature did not meet declared tolerance')
    report=normalize_moments(result.estimate,result.error,center,scale)
    report.update(rtol=rtol,atol=atol,status=result.status,subdivisions=result.subdivisions)
    return report
