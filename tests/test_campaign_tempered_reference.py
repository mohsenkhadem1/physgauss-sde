from dataclasses import dataclass,replace
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from campaign_tempered_reference import tempered_smc,reflect_coordinates,systematic_resample

@dataclass(frozen=True)
class State:
    coordinates:np.ndarray
    category:np.ndarray
    def take(self,ids):return State(self.coordinates[ids].copy(),self.category[ids].copy())
    def with_coordinates(self,x):return replace(self,coordinates=x)

class Prior:
    lower=np.array([-5.])
    upper=np.array([5.])
    def sample(self,n,rng):return State(rng.uniform(-5,5,(n,1)),rng.integers(2,size=n))
    def log_prob(self,s):
        valid=(np.abs(s.coordinates[:,0])<=5)&np.isin(s.category,[0,1])
        return np.where(valid,-np.log(20.),-np.inf)


def test_flat_likelihood_preserves_joint_prior_and_exact_evidence():
    result=tempered_smc(Prior(),lambda s:np.full(len(s.category),2.),particles=4000,
        seed=11,periodic_indices=())
    assert result.log_evidence_estimate==pytest.approx(2.)
    assert len(result.stages)==1
    assert abs(result.state.coordinates.mean())<.12
    assert abs(np.mean(result.state.category)-.5)<.035
    assert result.stages[0]['acceptance']['local']==1.
    assert result.stages[0]['acceptance']['independence']==1.


def test_conjugate_continuous_and_discrete_joint_posterior():
    def ll(s):
        x=s.coordinates[:,0]
        return -.5*((x-.7)/.8)**2-np.log(.8*np.sqrt(2*np.pi))+np.log(np.where(s.category,.8,.2))
    result=tempered_smc(Prior(),ll,particles=8192,seed=23,periodic_indices=())
    assert abs(result.state.coordinates.mean()-.7)<.055
    assert abs(result.state.coordinates.std()-.8)<.055
    assert abs(result.state.category.mean()-.8)<.03
    assert abs(result.log_evidence_estimate-np.log(.05))<.07
    assert result.stages[-1]['beta']==1.
    assert len(result.stages)>1


def test_separated_modes_retained_across_independent_runs():
    def ll(s):
        x=s.coordinates[:,0]
        a=np.log(.3)-.5*((x+2)/.22)**2
        b=np.log(.7)-.5*((x-2)/.22)**2
        return np.logaddexp(a,b)-np.log(.22*np.sqrt(2*np.pi))
    for seed in [41,104]:
        result=tempered_smc(Prior(),ll,particles=4096,seed=seed,periodic_indices=(),rejuvenation_steps=8)
        assert abs(np.mean(result.state.coordinates[:,0]>0)-.7)<.065
        assert abs(result.log_evidence_estimate+np.log(10))<.12


def test_box_proposal_and_failed_likelihood_are_explicit():
    x=reflect_coordinates(np.array([[-1.,7.]]),[0.,0.],[2.,3.],periodic_indices=(1,))
    np.testing.assert_allclose(x,[[1.,1.]])
    with pytest.raises(ValueError,match='No likelihood support'):
        tempered_smc(Prior(),lambda s:np.full(len(s.category),-np.inf),particles=16,periodic_indices=())
    with pytest.raises(ValueError,match='NaN|Likelihood'):
        tempered_smc(Prior(),lambda s:np.full(len(s.category),np.nan),particles=16,periodic_indices=())
    with pytest.raises(ValueError):systematic_resample([.1,.1],np.random.default_rng(0))


def test_local_moves_keep_nonuniform_intrinsic_prior_jacobian():
    class PhysicalUniformDensityPrior(Prior):
        lower=np.array([np.log(400.)])
        upper=np.array([np.log(850.)])
        def sample(self,n,rng):
            return State(np.log(rng.uniform(400,850,(n,1))),rng.integers(2,size=n))
        def log_prob(self,s):
            x=s.coordinates[:,0]
            return np.where((x>=self.lower[0])&(x<=self.upper[0]),x-np.log(900.),-np.inf)
    result=tempered_smc(PhysicalUniformDensityPrior(),lambda s:np.zeros(len(s.category)),
        particles=8000,seed=19,periodic_indices=(),rejuvenation_steps=80,
        independence_probability=.01,proposal_scale=.2)
    # Dropping the log-density Jacobian would drift toward a log-uniform rho.
    assert abs(np.exp(result.state.coordinates[:,0]).mean()-625)<5


def test_unknown_category_normalizers_cancel_only_with_declared_moves_and_safe_decode():
    class KernelPrior:
        lower=np.array([0.])
        upper=np.array([1.])
        def sample(self,n,rng):
            bc=rng.integers(2,size=n)
            lo=np.where(bc,.75,.25)
            return State(rng.uniform(lo,1.)[:,None],bc)
        def log_kernel(self,s):
            # Normalizers differ by3x but BC prior must remain uniform.
            lo=np.where(s.category,.75,.25)
            return np.where((s.coordinates[:,0]>=lo)&(s.coordinates[:,0]<=1),0.,-np.inf)
        def normalization_stratum(self,s):return s.category
    prior=KernelPrior()
    def likelihood(s):
        assert np.isfinite(prior.log_kernel(s)).all(), 'Invalid geometry reached strict decoder'
        return np.log(np.where(s.coordinates[:,0]>.8,10.,1.))
    result=tempered_smc(prior,likelihood,particles=8192,seed=71,periodic_indices=(),
        proposal_scale=.15,independence_probability=.35,rejuvenation_steps=8)
    # E[L|BC0]=3.4; E[L|BC1]=8.2, hence exact BC1 mass8.2/11.6.
    assert abs(result.state.category.mean()-8.2/11.6)<.035
    assert abs(result.log_evidence_estimate-np.log(5.8))<.07
    assert 'stratum' in result.prior_density_contract


def test_full_geometry_rejection_prior_runs_without_invalid_decoder_calls():
    from isaac_dataset import geometry
    from campaign_rejection_prior import ConditionalRejectionPrior
    points,_,_,spec=geometry(dict(cells=[12,6,4],seed=531000,
        boundary_type='point_pin',force_type='push_pulse',load_patch_nodes=9,seconds=3.))
    prior=ConditionalRejectionPrior(points,points[spec['load_nodes']],np.array(spec['load_patches']))
    def likelihood(state):
        decoded=prior.decode(state)  # Raises on infeasible candidates.
        assert decoded['physical'].shape==(len(state),25)
        return np.zeros(len(state))
    result=tempered_smc(prior,likelihood,particles=128,seed=29,rejuvenation_steps=2,
        proposal_scale=.2,independence_probability=.25)
    assert np.isfinite(prior.log_kernel(result.state)).all()
    assert result.log_evidence_estimate==pytest.approx(0.)
    assert 0<result.prior_sampling_failure_probability_bound<1e-12
