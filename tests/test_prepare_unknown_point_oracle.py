"""Reproducible input/private-role controls, never finite mechanics or responses."""
import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
from experiments import prepare_unknown_point_oracle as builder
from experiments.campaign_hypothesis_inputs import make_prior,conditioned_BC_sample,realize_hypothesis,plain
from experiments.campaign_exciting_commands import make_exciting_commands,PROGRAM_IDS


@pytest.fixture(scope='module')
def generated():
    with tempfile.TemporaryDirectory(prefix='unknown-point-input-test-',dir=builder.ROOT/'runs/operator-dev') as temporary:
        root=Path(temporary);output=(root/'inputs').relative_to(builder.ROOT).as_posix()
        finite=(root/'no-responses').relative_to(builder.ROOT).as_posix()
        original_run,original_build=builder.worker.run,builder.worker.build_model
        def forbidden(*args,**kwargs):raise AssertionError('Input generation must never construct/run finite mechanics')
        builder.worker.run=forbidden;builder.worker.build_model=forbidden
        try:summary=builder.build(output,finite)
        finally:builder.worker.run=original_run;builder.worker.build_model=original_build
        yield root,root/'inputs',finite,summary


def test_full_seeded_conditional_prior_retains_every_truth_nuisance_without_selection(generated):
    _,directory,_,_=generated
    public=json.loads((directory/'public/H0.json').read_text());prior=make_prior(public)
    expected,sampling=conditioned_BC_sample(prior,0,np.random.default_rng(20261110))
    saved=json.loads((directory/'private-truth/retained-state.json').read_text())
    spec=json.loads((directory/'private-truth/realized-hypothesis.json').read_text())
    # Generic errors deliberately avoid displaying private truth in test logs.
    if saved['state']!=plain(vars(expected)) or saved['sampling']!=sampling:
        raise AssertionError('Retained full seeded conditional prior state differs')
    if spec!=realize_hypothesis(public,expected,prior):raise AssertionError('Full physical realization differs from retained state')
    assert np.asarray(saved['state']['coordinates']).shape==(1,12)
    assert np.isfinite(prior.log_prob(expected)).all()
    assert (directory/'private-truth/realized-hypothesis.json').stat().st_mode&0o077==0


def test_public_queries_prior_and_commands_contain_no_selected_truth(generated):
    _,directory,_,_=generated
    manifest=json.loads((directory/'public/manifest.json').read_text())
    assert not any(key in manifest for key in ['truth_spec','private_retained_state','seed','selected_node_id','realized_hypothesis'])
    assert manifest['latent_truth_fields_present'] is False
    assert builder.worker.sha(directory/'public/acquired-queries.json')==builder.ACQUIRED_SHA
    assert builder.worker.sha(directory/'public/held-queries.json')==builder.HELD_SHA
    acquired=json.loads((directory/'public/acquired-queries.json').read_text())
    assert [q['time_s'] for q in acquired]==[.5,1.] and sum(len(q['xs']) for q in acquired)==72
    for program in PROGRAM_IDS:
        expected=make_exciting_commands(program)
        with np.load(directory/'public'/(program+'.npz'),allow_pickle=False) as saved:
            for key in saved.files:np.testing.assert_array_equal(saved[key],getattr(expected,key))


def test_validated_worker_protocols_and_prospective_stratified_plan_no_launch(generated):
    _,directory,finite,summary=generated
    assert not builder.worker.resolve(finite).exists()
    assert summary['status']=='INPUTS_FROZEN_NO_RUNS_LAUNCHED'
    design=json.loads((directory/'protocol-design.json').read_text());plan=design['reference_plan']
    assert plan['counts_per_mask']==[256,1024] and plan['seeds']==[20261112,20261113,20261114]
    assert plan['posterior_predictive_particles']==plan['prior_predictive_particles']==2048
    assert plan['posterior_predictive_seed']==20261115 and plan['prior_predictive_seed']==20261118
    assert plan['held_observation_seeds']==dict(zip(PROGRAM_IDS,[20261116,20261117]))
    assert design['likelihood']['observation_seed']==20261111 and design['support_prior']['mask_count']==55
    for record in design['physics_cases']:
        protocol=builder.worker.validate_protocol(record['path'],record['sha256'])
        assert protocol['integration_hz']==120 and protocol['settings']['frames']==61
        assert protocol['authorization'].startswith('INPUTS_ONLY_NOT_LAUNCH_AUTHORIZATION')
    assert design['finite_runs_launched']==design['inference_runs_launched']==0


def test_reproduction_same_private_truth_and_public_arrays_without_response_access(generated):
    root,directory,_,_=generated
    new=(root/'reproduced').relative_to(builder.ROOT).as_posix()
    finite=(root/'also-no-responses').relative_to(builder.ROOT).as_posix()
    builder.build(new,finite);second=builder.worker.resolve(new)
    for relative in ['private-truth/retained-state.json','private-truth/realized-hypothesis.json',
                     'public/H0.json','public/acquired-queries.json','public/held-queries.json',
                     'public/point-mask-prior.json',*[f'public/{p}.npz' for p in PROGRAM_IDS]]:
        if (directory/relative).read_bytes()!=(second/relative).read_bytes():raise AssertionError('Reproduction artifact differs')
    with pytest.raises(ValueError,match='Fresh'):builder.build(new,finite)
