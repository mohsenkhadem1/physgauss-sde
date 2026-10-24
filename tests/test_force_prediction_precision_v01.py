import numpy as np
import pytest
from experiments import campaign_force_prediction_precision_v01 as h


def blocks(x,w,B=16):
 n=len(x)//B;w=w/w.sum();mass=[];mean=[];variance=[]
 for b in range(B):
  xb=x[b*n:(b+1)*n];wb=w[b*n:(b+1)*n];m=wb.sum();a=xb[0];mu=wb@(xb-a)/m
  mass.append(m);mean.append(a+mu);variance.append(wb@((xb-a-mu)**2)/m)
 return np.array(mass)[:,None],np.array(mean)[:,None],np.array(variance)[:,None]


def test_ratio_and_sd_independent_draw_influence_covariance():
 rng=np.random.default_rng(23);x=rng.normal(size=(3200,5));w=np.exp(1.5*x[:,0]);w/=w.sum()
 mass,mean,var=blocks(x,w);result=h.delta_mcse(mass,mean,var)
 mu=w@x;vv=w@((x-mu)**2);B=16;n=len(x)//B
 # Direct draw influence with global normalization, then contiguous averaging.
 im=(len(x)*w[:,None]*(x-mu)).reshape(B,n,5).mean(1)
 iv=(len(x)*w[:,None]*((x-mu)**2-vv)).reshape(B,n,5).mean(1)/(2*np.sqrt(vv))
 np.testing.assert_allclose(result['mean'][0],mu,atol=2e-15)
 np.testing.assert_allclose(result['variance'][0],vv,rtol=1e-13)
 np.testing.assert_allclose(result['mean_MCSE'][0],im.std(0,ddof=1)/np.sqrt(B),rtol=1e-13)
 np.testing.assert_allclose(result['SD_MCSE'][0],iv.std(0,ddof=1)/np.sqrt(B),rtol=1e-13)
 # Naive variation of independently normalized block means loses denominator covariance.
 naive=mean[:,0].std(0,ddof=1)/np.sqrt(B)
 assert np.max(abs(naive-result['mean_MCSE'][0]))>1e-4


def test_translation_and_chain_merge_matches_direct_blocks():
 rng=np.random.default_rng(19);x=rng.normal(size=(1600,3));w=np.exp(x[:,0])
 a=h.delta_mcse(*blocks(x,w));b=h.delta_mcse(*blocks(x+1e5,w))
 np.testing.assert_allclose(a['mean_MCSE'],b['mean_MCSE'],atol=2e-12)
 np.testing.assert_allclose(a['SD_MCSE'],b['SD_MCSE'],atol=2e-12)
 merged=h.combine(*blocks(x,w),[[2*i,2*i+1] for i in range(8)])
 direct=blocks(x,w,B=8)
 for left,right in zip(merged,direct):np.testing.assert_allclose(left,right,atol=2e-15)


def test_rare_mass_and_no_mass_do_not_pass():
 x=np.arange(1600,dtype=float)[:,None]*1e-9;w=np.ones(1600);w[:100]=1e8
 result=h.assess(*blocks(x,w));assert not result['block_weight_resolution_passed'][0]
 assert not result['passed'][0]
 w[:]=0;w[0]=1
 with pytest.raises(ValueError,match='Zero block mass'):
  mass=np.zeros((16,1));mass[0]=1
  h.delta_mcse(mass,np.zeros((16,1,1)),np.zeros((16,1,1)))


def test_constant_distribution_zero_sd_is_exact():
 r=h.assess(np.full((16,1),1/16),np.ones((16,1,2)),np.zeros((16,1,2)))
 assert np.all(r['passed']);assert not np.any(r['SD_RMS_MCSE_m'])


def test_mean_or_sd_failure_is_preserved():
 mass=np.full((16,1),1/16);mean=np.zeros((16,1,1));var=np.ones_like(mean)*1e-8
 mean[:8]=1e-3
 assert not h.assess(mass,mean,var)['mean_precision_passed'][0]
 mean[:]=0;var[:8]=1e-4
 assert not h.assess(mass,mean,var)['SD_precision_passed'][0]
