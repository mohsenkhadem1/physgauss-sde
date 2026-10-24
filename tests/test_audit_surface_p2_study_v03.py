import json
from pathlib import Path
import numpy as np
import pytest
from experiments import audit_surface_p2_study_v03 as a


def write(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x))


def test_all_seals_before_any_npz_read_and_tamper_refusal(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'require_absent',lambda pid:None)
    actions={'normal':['n0','n1','n2'],'paired':['a0','a1','a2']};write(tmp_path/'protocol.json',dict(source_SHA256={},design=dict(actions=actions)))
    digest=a.sha(tmp_path/'protocol.json');write(tmp_path/'execution.json',dict(completed=True,protocol_SHA256=digest,all_worker_groups_absent=True,pid=900,records=[dict(slot=i,pid=1000+i,reaped=True,exit_code=0)for i in range(16)]))
    for i in range(16):
        folder=tmp_path/f'object-{i:02d}-LOCAL';terminal=folder/'terminal-LOCAL.json'
        write(terminal,dict(index=i,protocol_SHA256=digest,retained=True,completed=True))
        write(folder/'checkpoint.json',dict(stage='terminal',index=i,protocol_SHA256=digest,terminal_SHA256=a.sha(terminal)))
        write(tmp_path/f'worker-{i}-terminal.json',dict(slot=i,pid=1000+i,protocol_SHA256=digest,completed=True,records=[dict(index=i,completed=True)]))
        for law in a.LAWS:
            prefix=folder/f'{law}-prefix-LOCAL.json';write(prefix,dict(qualified=False))
            write(folder/f'{law}-choices-seal-LOCAL.json',dict(protocol_SHA256=digest,prefix_checkpoint_SHA256=a.sha(prefix),post_probe_outcomes_read=False,truth_used_for_selection=False,choices=None))
        (folder/'truth-LOCAL.npz').write_bytes(b'Never open before valid seals')
    def forbidden(*args,**kwargs):raise AssertionError('Truth/target NPZ read in seal precheck')
    monkeypatch.setattr(a.np,'load',forbidden)
    assert len(a.preconditions(tmp_path,digest)[3])==32
    prefix=tmp_path/'object-07-LOCAL/matching_linear-prefix-LOCAL.json';write(prefix,dict(qualified=True))
    with pytest.raises(ValueError,match='Seal prefix binding'):a.preconditions(tmp_path,digest)


def test_independent_score_and_observation_model_recipe_archive_bindings(tmp_path):
    path=tmp_path/'posterior.json';target=np.zeros(96);unit=np.array([.5,.9]);observations=[('prefix',np.arange(24)*1e-6)]
    np.savez_compressed(path.with_suffix('.npz'),mean=np.full(96,3e-6),MAP_prediction=np.full(96,4e-6),SD=np.full(96,2e-6),unit_quantiles=np.array([[.1,.5,.8],[.2,.5,.8]]),quadrature_probability=np.array([.25,.75]),unit_grid=np.array([[.2,.3],[.6,.7]]))
    record=dict(qualified=True,numerical_engine='conditional_amplitude',recipe_SHA256='recipe',model_fingerprint='model',observations_SHA256=a.fingerprint(observations),raw_SHA256=a.sha(path.with_suffix('.npz')),diagnostics=dict(qualified=True,successive=dict(inner=[dict(passed=True)],outer=[dict(passed=True)]),independent_adaptive=dict(passed=True)))
    write(path,record)
    value,passed=a.posterior(path,observations,target,unit,engine='conditional_amplitude',recipe='recipe',model='model')
    assert passed and value['posterior_RMS_m']==pytest.approx(3e-6)and value['MAP_RMS_m']==pytest.approx(4e-6)
    assert value['E_90percent_covered']==1 and value['K_90percent_covered']==0
    assert value['mean_predictive_SD_m']==pytest.approx(2e-6)
    for change in ['model','recipe','observation']:
        kwargs=dict(engine='conditional_amplitude',recipe='recipe',model='model');obs=observations
        if change in kwargs:kwargs[change]='wrong'
        else:obs=[('prefix',np.zeros(24))]
        with pytest.raises(ValueError,match='posterior context'):a.posterior(path,obs,target,unit,**kwargs)
    path.with_suffix('.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='archive hash'):a.posterior(path,observations,target,unit,engine='conditional_amplitude',recipe='recipe',model='model')


def test_policy_noise_averaging_exact_random_and_object_bootstrap():
    rows=[[dict(resolved=True,posterior_RMS_m=float(10*action+rep))for rep in range(3)]for action in range(3)]
    choices={f:dict(proposed=2,information=0)for f in ['normal','axial']}
    output=a.policies(choices,{f:rows for f in choices},[1,0],list(choices))
    assert output['normal']['proposed']['posterior_RMS_m']==21
    assert output['normal']['uniform_random']['posterior_RMS_m']==11
    assert output['normal']['ancillary_random_realization']['posterior_RMS_m']==11
    assert output['axial']['ancillary_random_realization']['posterior_RMS_m']==1
    b=a.bootstrap(np.full(16,2e-6),123);np.testing.assert_allclose(b['CI95'],[2e-6,2e-6],rtol=0,atol=1e-20)
    assert b['objects']==16
    with pytest.raises(ValueError):a.bootstrap(np.full(48,2e-6),123)


def test_physical_gate_recomputed_instead_of_trusting_boolean():
    d=dict(minimum_J=.99,maximum_absolute_symmetric_strain_eigenvalue=.01,nonrigid_RMS_m=200e-6,nonrigid_fraction=.5,material_valid=True,excitation_passed=True,prefix_excitation_required=True,valid=True,equilibrium=dict(exact_clamp=True,free_residual_N=1e-12,tolerance_N=1e-10,force_balance_error_N=1e-12,force_balance_tolerance_N=1e-9,moment_balance_error_Nm=1e-12,moment_balance_tolerance_Nm=1e-9,passed=True),stability=dict(minimum_eigenvalue_estimate_N_m=1.,eigenpair_residual_N_m=1e-9,relative_eigenpair_residual=1e-9,passed=True))
    assert a.physical_valid(d,True)
    d['maximum_absolute_symmetric_strain_eigenvalue']=.051
    with pytest.raises(ValueError,match='material gate'):a.physical_valid(d,True)
    d.update(material_valid=False,valid=False)
    assert not a.physical_valid(d,True)


def test_unqualified_posterior_retained_without_fabricated_score(tmp_path):
    path=tmp_path/'failed.json';obs=[('a',np.zeros(24))]
    write(path,dict(qualified=False,numerical_engine='conditional_amplitude',recipe_SHA256='r',model_fingerprint='m',observations_SHA256=a.fingerprint(obs),diagnostics=dict(qualified=False,successive=dict(inner=[dict(passed=False)],outer=[dict(passed=True)]),independent_adaptive=dict(passed=True))))
    result,qualified=a.posterior(path,obs,np.zeros(96),np.ones(2)*.5,engine='conditional_amplitude',recipe='r',model='m')
    assert not qualified and result==dict(resolved=False)


def test_numerical_scalar_gate_boolean_cannot_hide_failure():
    d=dict(successive=dict(inner=[dict(passed=True,maximum_CDF_difference=.006)],outer=[dict(passed=True)]),independent_adaptive=dict(passed=True))
    with pytest.raises(ValueError,match='scalar gate fails'):a.verify_numerical_flags(d)
    d['successive']['inner'][0].update(passed=False)
    assert not a.verify_numerical_flags(d)
    d['successive']['inner'][0].update(passed=True,maximum_CDF_difference=0.)
    d['independent_adaptive'].update(log_evidence_difference=0.,maximum_CDF_error=.006,adaptive_refinement_log_evidence_difference=0.,adaptive_refinement_maximum_CDF_difference=0.,maximum_QUADPACK_relative_error_estimates={'1e-8':1e-7})
    with pytest.raises(ValueError,match='adaptive scalar gates'):a.verify_numerical_flags(d)
