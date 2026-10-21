"""Normalized prior and exact-likelihood oracle checks, without physics runs."""
import itertools
import numpy as np
import pytest

from experiments import campaign_hypothesis_inputs as inputs
from experiments.campaign_material_marginal import MaterialMarginalPrior, MaterialState, realize_known_support_material
from experiments.campaign_tempered_reference import tempered_smc


@pytest.fixture(scope='module')
def context():
    public = inputs.public_hardware_contexts()[0]
    parent = inputs.make_prior(public)
    return public,parent,MaterialMarginalPrior(parent)


def test_intrinsic_density_integrates_to_one(context):
    _,_,prior = context
    nodes,weights = np.polynomial.legendre.leggauss(12)
    widths = prior.upper-prior.lower
    coords = np.array(list(itertools.product(*[lo+(nodes+1)*width/2 for lo,width in zip(prior.lower,widths)])))
    w = np.prod(np.array(list(itertools.product(*[weights*width/2 for width in widths]))),axis=1)
    integral = np.dot(w,np.exp(prior.log_prob(MaterialState(coords))))
    np.testing.assert_allclose(integral,1.,atol=2e-14,rtol=0)


def test_sampling_uses_physical_uniform_density_and_drag(context):
    _,_,prior = context
    s = prior.sample(100000,np.random.default_rng(20261025))
    physical = prior.physical_coordinates(s)
    assert abs(physical[:,1].mean()-625)<2.0
    assert abs(physical[:,3].mean()-.6)<.0025
    assert abs(s.coordinates[:,0].mean()-(prior.lower[0]+prior.upper[0])/2)<.0125
    assert abs(physical[:,2].mean()-.35)<.001
    bad = s.take([0]);x=bad.coordinates.copy();x[0,2]=.5
    assert np.isneginf(prior.log_prob(bad.with_coordinates(x))[0])
    with pytest.raises(ValueError):prior.physical_coordinates(bad.with_coordinates(x))


def test_material_density_ratios_match_full_normalized_prior(context):
    _,parent,prior = context
    rng = np.random.default_rng(20261026)
    original = parent.sample(16,rng)
    material = prior.sample(16,rng)
    changed = original.coordinates.copy();changed[:,:4]=material.coordinates
    full_delta = parent.log_prob(original.with_coordinates(changed))-parent.log_prob(original)
    marginal_delta = prior.log_prob(material)-prior.log_prob(MaterialState(original.coordinates[:,:4]))
    np.testing.assert_allclose(full_delta,marginal_delta,atol=8e-15,rtol=0)


def test_realization_preserves_supports_and_nuisances(context):
    public,parent,prior = context
    rng = np.random.default_rng(20261027)
    template = inputs.realize_hypothesis(public,parent.sample(1,rng),parent)
    candidates = prior.sample(2,rng)
    specs = realize_known_support_material(public,template,candidates)
    expected = prior.physical_coordinates(candidates)
    for row,spec in zip(expected,specs):
        for key in ['fixture_nodes','support_node_groups','support_origins_world_m','boundary_kind',
                    'motion_amplitude_m','motion_direction_world','physical_hardware_sha256']:
            assert spec[key]==template[key]
        np.testing.assert_array_equal(spec['retained_prior_state']['coordinates'][0][4:],template['retained_prior_state']['coordinates'][0][4:])
        np.testing.assert_allclose([spec['material'][k] for k in ['youngs_modulus_Pa','density_kg_m3','poissons_ratio','body_linear_drag_s_inverse']],row,rtol=0,atol=0)


def test_existing_smc_recovers_scalar_analytic_oracle_with_four_dimensional_prior(context):
    _,_,prior = context
    # A simple exact likelihood tests the generic sampler/prior interface. This
    # is not a physical inference/calibration experiment.
    result = tempered_smc(prior,lambda s:-.5*((s.coordinates[:,2]-.33)/.01)**2,
        particles=2048,seed=20261028,rejuvenation_steps=6,periodic_indices=())
    assert result.stages[-1]['beta']==1.
    assert abs(np.dot(result.weights,result.state.coordinates[:,2])-.33)<.002
    assert abs(np.dot(result.weights,prior.physical_coordinates(result.state)[:,1])-625)<20
    assert all(stage['category_occupancy']=={} for stage in result.stages)
    assert 'not proof' in result.diagnostics()['target']
