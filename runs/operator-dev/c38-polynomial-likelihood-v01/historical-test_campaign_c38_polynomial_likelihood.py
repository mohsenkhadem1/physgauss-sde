import numpy as np
import pytest
from scipy.special import ndtr
from experiments.campaign_c38_polynomial_likelihood import GaussianPolynomialLikelihood,PolynomialPosterior

def model(coeff,observed=(0.,),sigma=1.,**kwargs):
 return GaussianPolynomialLikelihood(np.asarray(coeff,dtype=float).reshape(3,-1),np.asarray(observed),np.eye(len(observed))*sigma*sigma,**kwargs)

def test_uniform_log_prior_not_uniform_inverse_modulus_and_target_cdf_moments():
 m=model([0,0,0]);p=PolynomialPosterior({'a':m,'b':m},category_prior=[.25,.75]);np.testing.assert_allclose(p.category_probabilities,[.25,.75],atol=2e-15)
 assert p.cdf(75000*2**.3)==pytest.approx(.3,abs=2e-14)
 c=p.coordinate_cdf({'a':[0.,1.,0.],'b':[0.,1.,0.]},.75);assert c['cdf']==pytest.approx(np.log(1.5)/np.log(2),abs=2e-13)
 moments=p.target_moments({'a':np.array([[1000.],[1e-3],[0.]]),'b':np.array([[1000.],[1e-3],[0.]])})
 meanx=.5/np.log(2);varx=.75/(2*np.log(2))-meanx**2
 assert moments['mean_m'][0]==pytest.approx(1000+1e-3*meanx,abs=2e-12)
 assert moments['variance_m2'][0]==pytest.approx(1e-6*varx,rel=2e-10,abs=1e-18)
 assert not c['observation_noise_added'] and not moments['observation_noise_added']


def test_narrow_linear_gaussian_recovered_by_partitions_and_independent_rules():
 center=.75;sigma=2**-30;m=model([0.,1.,0.],observed=(center,),sigma=sigma)
 assert len(m.stationary)==1 and m.stationary[0][2]==1 and len(m.points)>=11
 # The uniform logE prior in x has density1/(x*log2). Its change over this
 # narrow mode is bounded~1e-8; probability splitatcenter shouldbehalfto1e-8.
 adaptive=PolynomialPosterior({'a':m,'b':m});fixed=PolynomialPosterior({'a':m,'b':m},method='legendre',order=64)
 assert adaptive.cdf(75000/center)==pytest.approx(.5,abs=2e-8)
 assert abs(adaptive.cdf(75000/center)-fixed.cdf(75000/center))<1e-10
 assert abs(adaptive.log_normalizer-fixed.log_normalizer)<1e-9


def test_bimodal_near_repeated_and_exact_multiple_stationary_roots():
 a=.625;b=.875;m=model([a*b,-a-b,1.],sigma=2**-30)
 assert len(m.stationary)==3
 p=PolynomialPosterior({'a':m,'b':m});f=PolynomialPosterior({'a':m,'b':m},method='legendre',order=128)
 # Narrow-root masses scale1/(x*|a-b|): larger x root carries smaller mass.
 expected_highroot=(1/b)/(1/a+1/b)
 assert p.cdf(75000/.75)==pytest.approx(expected_highroot,abs=2e-7)
 assert abs(p.cdf(75000/.75)-f.cdf(75000/.75))<1e-9
 a=.75;b=a+2**-25;near=model([a*b,-a-b,1.],sigma=2**-50)
 assert len(near.stationary)==3 and sum(x[2] for x in near.stationary)==3
 repeated=model([a*a,-2*a,1.],sigma=2**-20)
 assert len(repeated.stationary)==1 and repeated.stationary[0][2]==3


def test_local_expansion_avoids_large_coefficient_cancellation_and_rejects_inputs():
 a=.625;b=.875;m=model([a*b,-a-b,1.],sigma=2**-30);x=m.stationary[0][0]
 r0,r1,r2=m.local(x);assert np.linalg.norm(r0)<1e-35
 dx=1e-10;expected=.5*float(np.linalg.norm(r0+dx*(r1+dx*r2))**2)-.5*float(r0@r0)
 assert m.local_D_change(x,dx)==pytest.approx(expected,rel=1e-13)
 with pytest.raises(np.linalg.LinAlgError):GaussianPolynomialLikelihood(np.zeros((3,1)),[0.],[[-1.]])
 with pytest.raises(ValueError):m.log_likelihood(74000.)
 with pytest.raises(ValueError):PolynomialPosterior({'a':m,'b':model([0,0,0],sigma=2.)})
