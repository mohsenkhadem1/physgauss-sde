import json
from pathlib import Path
import numpy as np
import pytest
from experiments import audit_surface_p2_study_v04_v02 as a


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
    original_read=a.read;seen=set()
    def ordered_read(path):
        path=Path(path)
        if path.name.endswith('-choices-seal-LOCAL.json'):seen.add(str(path))
        if path.name=='terminal-LOCAL.json':assert len(seen)==32,'Target-derived terminal scores read before all seals'
        return original_read(path)
    monkeypatch.setattr(a,'read',ordered_read)
    assert len(a.preconditions(tmp_path,digest,require_all_attempts=False)[3])==32
    prefix=tmp_path/'object-07-LOCAL/matching_linear-prefix-LOCAL.json';write(prefix,dict(qualified=True))
    with pytest.raises(ValueError,match='Seal prefix binding'):a.preconditions(tmp_path,digest,require_all_attempts=False)


def test_independent_score_and_observation_model_recipe_archive_bindings(tmp_path):
    path=tmp_path/'matching_linear-normal_x090-r0-LOCAL.json';target=np.zeros(96);unit=np.array([.5,.9]);observations=[('prefix',np.arange(24)*1e-6)]
    np.savez_compressed(path.with_suffix('.npz'),mean=np.full(96,3e-6),MAP_prediction=np.full(96,4e-6),SD=np.full(96,2e-6),unit_quantiles=np.array([[.1,.5,.8],[.2,.5,.8]]),unit_mean=np.array([.5,.5]),unit_covariance=np.eye(2),unit_CDF=np.tile(np.linspace(0,1,33),(2,1)),MAP_unit=np.array([.5,.5]),MAP_log_likelihood=np.array(0.))
    record=dict(archive_kind='joint_sufficient_statistics',qualified=True,numerical_engine='conditional_amplitude_PWL_v02',recipe_SHA256='recipe',model_fingerprint='model',observations_SHA256=a.fingerprint(observations),raw_SHA256=a.sha(path.with_suffix('.npz')),diagnostics=numeric_fixture())
    write(path,record)
    value,passed=a.posterior(path,observations,target,unit,engine='conditional_amplitude_PWL_v02',recipe='recipe',model='model')
    assert passed and value['posterior_RMS_m']==pytest.approx(3e-6)and value['MAP_RMS_m']==pytest.approx(4e-6)
    assert value['E_90percent_covered']==1 and value['K_90percent_covered']==0
    assert value['mean_predictive_SD_m']==pytest.approx(2e-6)
    for change in ['model','recipe','observation']:
        kwargs=dict(engine='conditional_amplitude_PWL_v02',recipe='recipe',model='model');obs=observations
        if change in kwargs:kwargs[change]='wrong'
        else:obs=[('prefix',np.zeros(24))]
        with pytest.raises(ValueError,match='posterior context'):a.posterior(path,obs,target,unit,**kwargs)
    path.with_suffix('.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='archive hash'):a.posterior(path,observations,target,unit,engine='conditional_amplitude_PWL_v02',recipe='recipe',model='model')


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
    write(path,dict(qualified=False,numerical_engine='conditional_amplitude_PWL_v02',recipe_SHA256='r',model_fingerprint='m',observations_SHA256=a.fingerprint(obs),diagnostics=numeric_fixture(False)))
    result,qualified=a.posterior(path,obs,np.zeros(96),np.ones(2)*.5,engine='conditional_amplitude_PWL_v02',recipe='r',model='m')
    assert not qualified and result==dict(resolved=False)


def numeric_fixture(passed=True):
    def row(c,f):return dict(coarse=c,fine=f,passed=True,log_evidence_difference=0.,maximum_CDF_difference=0.,maximum_unit_quantile_difference=0.,maximum_unit_mean_difference=0.,maximum_predictive_mean_difference_m=1e-7,maximum_predictive_SD_difference_m=1e-7)
    d=dict(qualified=passed,gates=dict(logZ=.01,CDF=.005,quantile=.002,unit_mean=.001,target_absolute_m=2e-6,target_relative_SD=.01),independent_adaptive_per_posterior_required=True,successive={axis:[row(128,256),row(256,512)]for axis in ('inner','outer')},independent_adaptive=dict(passed=True,log_evidence_difference=0.,maximum_CDF_error=0.,adaptive_refinement_log_evidence_difference=0.,adaptive_refinement_maximum_CDF_difference=0.,maximum_QUADPACK_relative_error_estimates={'1e-06':1e-7,'1e-08':1e-7},epsrel=[1e-6,1e-8],CDF_checks_per_epsrel=6))
    if not passed:d['successive']['inner'][-1].update(passed=False,maximum_CDF_difference=.006)
    return d


def test_numerical_scalar_gate_boolean_cannot_hide_failure():
    d=numeric_fixture();d['successive']['inner'][0]['maximum_CDF_difference']=.006
    with pytest.raises(ValueError,match='scalar gate fails'):a.verify_numerical_flags(d)
    d['successive']['inner'][0]['passed']=False
    assert a.verify_numerical_flags(d)
    d['independent_adaptive']['maximum_CDF_error']=.006
    with pytest.raises(ValueError,match='adaptive scalar gates'):a.verify_numerical_flags(d)


def test_numerical_metadata_missing_nonfinite_or_gate_tamper_rejected():
    for mutate in [lambda d:d['successive']['inner'][0].pop('maximum_predictive_SD_difference_m'),lambda d:d['independent_adaptive'].pop('maximum_CDF_error')]:
        d=numeric_fixture();mutate(d)
        with pytest.raises(KeyError):a.verify_numerical_flags(d)
    for mutate in [lambda d:d['gates'].update(CDF=.006),lambda d:d['independent_adaptive'].update(CDF_checks_per_epsrel=5),lambda d:d['independent_adaptive'].update(epsrel=[1e-8]),lambda d:d['independent_adaptive'].update(maximum_QUADPACK_relative_error_estimates={'1e-08':1e-7}),lambda d:d['successive']['inner'][0].update(maximum_predictive_SD_difference_m=float('nan'))]:
        d=numeric_fixture();mutate(d)
        with pytest.raises(ValueError):a.verify_numerical_flags(d)


def test_relative_only_prediction_pass_explicitly_source_bound():
    d=numeric_fixture();d['successive']['inner'][0]['maximum_predictive_SD_difference_m']=3e-6
    assert a.verify_numerical_flags(d)
    assert a.NUMERICAL_AUDITS[-1]==dict(sufficient_absolute_prediction_checks=3,relative_prediction_checks_source_bound_only=1)
    d['successive']['inner'][0]['passed']=False
    assert a.verify_numerical_flags(d)


def grouped_fixture():
    ratio=np.log([.0015,.0025,.0045]);rho=np.array([.2,.8,0.]);conditional=np.array([.2,.3,.5]);amp=np.array([[.00019,.00020,.00021],[.00022,.00023,.00024],[.00025,.00026,.00027]])
    m=amp@conditional;v=((amp-m[:,None])**2)@conditional
    flat=amp.ravel();z=np.repeat(ratio,3);unit=np.column_stack(((-np.log(flat)-np.log(2000))/np.log(4),(z-np.log(flat)-np.log(5))/np.log(6)))
    return dict(quadrature_probability=(rho[:,None]*conditional).ravel(),unit_grid=unit,inverse_amplitude=flat,log_ratio=z,group_log_ratio=ratio,group_probability=rho,group_amplitude_mean=m,group_amplitude_variance=v)


def test_grouped_covariance_preserves_all_positive_hypotheses_including_zero_global_mass():
    z=grouped_fixture();result=a.verify_grouped_archive(z)
    assert result==dict(groups=3,atoms=9,zero_mass_groups=1)
    bad={k:v.copy()for k,v in z.items()};bad['group_amplitude_variance'][0]+=1e-10
    with pytest.raises(ValueError,match='centered second'):a.verify_grouped_archive(bad)
    bad={k:v.copy()for k,v in z.items()};bad['group_amplitude_mean'][0]+=1e-5
    with pytest.raises(ValueError,match='first moments'):a.verify_grouped_archive(bad)
    bad={k:v.copy()for k,v in z.items()};bad['unit_grid'][0,0]+=.01
    with pytest.raises(ValueError,match='material coordinates'):a.verify_grouped_archive(bad)


def test_original_truth_byte_reuse_and_origin_receipts_reject_copy_or_membership_tampering(tmp_path,monkeypatch):
    root=tmp_path;old=root/'old';new=root/'new';old.mkdir();new.mkdir()
    monkeypatch.setattr(a,'ROOT',root);monkeypatch.setattr(a,'ORIGINAL',old)
    write(old/'protocol.json',{});monkeypatch.setattr(a,'ORIGINAL_PROTOCOL',a.sha(old/'protocol.json'))
    write(old/'controlled-stop-ownership.json',{});monkeypatch.setattr(a,'OLD_STOP_SHA',a.sha(old/'controlled-stop-ownership.json'))
    entries=[];files={}
    for i in range(16):
        folder=old/f'object-{i:02d}-LOCAL';target=new/f'object-{i:02d}-LOCAL';folder.mkdir();target.mkdir()
        entry=dict(index=i)
        for name,pathkey,hashkey in [('truth-LOCAL.npz','truth_file','truth_SHA256'),('truth-checkpoint-LOCAL.json','metadata_file','metadata_SHA256')]:
            path=folder/name;path.write_bytes(b'opaque synthetic archive bytes');(target/name).write_bytes(path.read_bytes())
            entry[pathkey]=str(path.relative_to(root));entry[hashkey]=a.sha(path);files[str(path.relative_to(old))]=a.sha(path)
        entries.append(entry)
        write(target/'truth-origin-LOCAL.json',dict(original_source_protocol_SHA256=a.ORIGINAL_PROTOCOL,new_numerical_protocol_SHA256='new-protocol',original_data=entry,no_simulation_or_redraw=True))
    manifest=old/'stopped-completed-manifest-LOCAL.json';write(manifest,dict(files=files));monkeypatch.setattr(a,'OLD_MANIFEST_SHA',a.sha(manifest))
    p=dict(original_source_protocol_SHA256=a.ORIGINAL_PROTOCOL,old_data=dict(original_source_protocol_SHA256=a.ORIGINAL_PROTOCOL,original_data_root=str(old),original_data_manifest=dict(path=str(manifest),SHA256=a.OLD_MANIFEST_SHA),old_stop_ownership_SHA256=a.OLD_STOP_SHA,objects=entries,no_new_simulation_or_draw=True))
    assert a.verify_origins(new,p,'new-protocol')['objects_byte_identical']==16
    copied=new/'object-00-LOCAL/truth-LOCAL.npz';copied.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Byte-identical'):a.verify_origins(new,p,'new-protocol')
    copied.write_bytes(b'opaque synthetic archive bytes');entries[0]['truth_file']=entries[1]['truth_file']
    with pytest.raises(ValueError,match='file mapping'):a.verify_origins(new,p,'new-protocol')
