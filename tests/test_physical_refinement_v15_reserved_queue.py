import numpy as np
import pytest
from scripts.physical_refinement_v15.reserved_worlds_queue import sample_cases,guard


def test_private_sampler_is_reproducible_continuous_and_unbalanced():
    # Synthetic seed/identities, not the reserved study draw.
    worlds=[dict(name=f'fixture-{i}',initial_camera='left') for i in range(2)]
    distribution=dict(gradient_uniform=[-.75,.75],common_scale_log_uniform=[.78,1.47],
                      layouts=['parallel_lower','crossed'],fixture_ratios=[.75,1.25],E_base=12000,K_base=6000)
    a=sample_cases(worlds,distribution,1);b=sample_cases(worlds,distribution,1)
    assert a==b and all(set(w)=={'name','initial_camera'} for w in worlds)
    rng=np.random.default_rng(1)
    for row in a:
        g=rng.uniform(-.75,.75);s=np.exp(rng.uniform(np.log(.78),np.log(1.47)))
        layout=distribution['layouts'][rng.integers(2)];ratio=distribution['fixture_ratios'][rng.integers(2)]
        assert row['case']==dict(E=12000*s,Kleft=6000*s*ratio,Kright=6000*s*ratio,layout=layout,material_gradient=g)
        assert g not in np.linspace(-.8,.8,17)


def test_sampler_rejects_public_latent_labels():
    with pytest.raises(ValueError,match='metadata only'):
        sample_cases([dict(name='dummy',initial_camera='left',case={})],{},1)


def test_freeze_guard_fails_before_generation(tmp_path):
    path=tmp_path/'p.json';path.write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):guard(path,'bad',path,'bad')
