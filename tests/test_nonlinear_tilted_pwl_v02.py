from decimal import Decimal,localcontext
import numpy as np
import pytest
from scipy.integrate import quad
from experiments.campaign_positive_amplitude_pwl_v01 import PositiveAmplitudePWL
from experiments.campaign_nonlinear_tilted_pwl_v01 import NonlinearTiltedPWL
from experiments.campaign_nonlinear_tilted_pwl_v02 import NonlinearTiltedPWLV02,positive_panel_probability
from experiments.campaign_nonlinear_conditional_reference_v01 import NonlinearConditionalReference
from experiments.campaign_nonlinear_conditional_reference_v02 import NonlinearConditionalReferenceV02


def panel_data(n,h):
    k=np.sin(np.pi*np.arange(n+1)/(2*n))**2;k[0]=0.;k[-1]=1.
    xi=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
    weights=(((1-xi)[None,None,:]*h[:,:-1,None]+xi[None,None,:]*h[:,1:,None])*np.diff(k)[None,:,None]/2)
    return k,weights.sum(axis=2)


def decimal_probability(k,h,m,x,reverse=False):
    with localcontext()as ctx:
        ctx.prec=80;D=Decimal.from_float;cut=Decimal(1)-D(float(x))if reverse else D(float(x));total=sum((D(float(v))for v in m),Decimal(0));answer=Decimal(0)
        for i in range(len(m)):
            left=D(float(k[i]));right=D(float(k[i+1]));h0=D(float(h[i]));h1=D(float(h[i+1]));mass=D(float(m[i]))
            if not mass:continue
            lo=max(left,cut)if reverse else left;hi=right if reverse else min(right,cut)
            if hi<=lo:continue
            a=(lo-left)/(right-left);b=(hi-left)/(right-left)
            integral=h0*(b-a)+(h1-h0)*(b*b-a*a)/2
            answer+=mass*integral/((h0+h1)/2)
        return float(answer/total)


def test_old_reflected_width_failure_reproduced_new_exact_endpoints():
    n=512;h=np.zeros((1,n+1));h[0,0]=1.;k,m=panel_data(n,h)
    old=PositiveAmplitudePWL._primitive(1-k[::-1],h[:,::-1],m[:,::-1],np.array([1.]))/m.sum(axis=1)
    assert old[0]>1+32*np.finfo(float).eps
    np.testing.assert_array_equal(positive_panel_probability(k,h,m,np.array([1.]),reverse=True),[1.])
    for reverse in(False,True):
        np.testing.assert_array_equal(positive_panel_probability(k,h,m,[0.],reverse=reverse),[0.])
        np.testing.assert_array_equal(positive_panel_probability(k,h,m,[1.],reverse=reverse),[1.])


def test_strong_tilt_monotone_complement_and_independent_decimal():
    for n in(128,256,512):
        k=np.sin(np.pi*np.arange(n+1)/(2*n))**2;k[0]=0;k[-1]=1
        h=np.vstack((np.exp(-600*k),np.exp(-600*(1-k)),np.exp(-((k-.37)/.007)**2),np.linspace(.01,1,n+1)))
        k,m=panel_data(n,h)
        coordinates=np.unique(np.r_[np.linspace(0,1,1025),k,np.nextafter(k[1:-1],0),np.nextafter(k[1:-1],1)])
        for reverse in(False,True):
            values=np.array([positive_panel_probability(k,h,m,np.full(4,x),reverse=reverse)for x in coordinates])
            assert np.isfinite(values).all()and values.min()>=0 and values.max()<=1
            assert np.diff(values,axis=0).min()>=-2e-15
            for x in [0.,2**-30,.001,.3,.5,.9,np.nextafter(1.,0),1.]:
                actual=positive_panel_probability(k,h,m,np.full(4,x),reverse=reverse)
                expected=[decimal_probability(k,h[i],m[i],x,reverse)for i in range(4)]
                np.testing.assert_allclose(actual,expected,rtol=0,atol=5e-14)
        for x in np.linspace(0,1,129):
            np.testing.assert_allclose(positive_panel_probability(k,h,m,np.full(4,x))+positive_panel_probability(k,h,m,np.full(4,1-x),reverse=True),1.,rtol=0,atol=5e-14)


def test_malformed_positive_mass_on_zero_density_refused():
    with pytest.raises(ValueError,match='nonzero density'):positive_panel_probability([0,.5,1],np.zeros((1,3)),np.ones((1,2)),[.3])


def test_quadrature_hypotheses_weights_unchanged_and_zero_branch_correct():
    lo=.000125;hi=.0004;z=np.log([.002]);mu=.00024;sd=.00003
    def a_of_unit(u):return np.exp(-np.log(2000)-u[:,0]*np.log(4))
    for sigma in(sd,np.inf):
        def ll(u):
            a=a_of_unit(u)
            return 3*np.log(a/lo)-5 if np.isinf(sigma)else -7-.5*((a-mu)/sigma)**2-.05*((a-mu)/sigma)**4
        args=([mu],[sigma],[lo],[hi]);kwargs=dict(log_ratio=z,linear_peak=[-7.],full_loglik=ll,panels=512)
        old=NonlinearTiltedPWL(*args,**kwargs);new=NonlinearTiltedPWLV02(*args,**kwargs)
        for field in('amplitude','log_weights','density_weights','panel_mass','mass0','h','log_kernel_mass'):np.testing.assert_array_equal(getattr(new,field),getattr(old,field))
        if np.isinf(sigma):
            for cut in np.linspace(lo,hi,33):
                expected=(cut**3-lo**3)/(hi**3-lo**3)
                assert abs(new.cdf([cut])[0]-expected)<2e-6
                np.testing.assert_allclose(new.cdf([cut])+new.tail([cut]),1.,atol=5e-14)


def test_new_reference_moments_evidence_and_weights_bit_identical():
    def ratio(z,labels):return np.broadcast_to((1+.03*z)[:,None,None],(len(z),len(labels),1))
    def residual(u,labels):return np.broadcast_to((2e-6*u[:,0]**2*u[:,1])[:,None,None],(len(u),len(labels),1))
    args=(ratio,None,[('observe',np.array([150e-6]))],['target']);kwargs=dict(residual_predict=residual,retain_hypotheses=True)
    old=NonlinearConditionalReference(*args,**kwargs).integrate(outer=8,inner=128,intrinsic=False,keep=True,moment_chunk=1024)
    new=NonlinearConditionalReferenceV02(*args,**kwargs).integrate(outer=8,inner=128,intrinsic=False,keep=True,moment_chunk=1024)
    for field in('log_evidence','unit_mean','unit_covariance','predictive_mean','predictive_SD','unit_grid','quadrature_probability','log_likelihood'):np.testing.assert_array_equal(new[field],old[field])
