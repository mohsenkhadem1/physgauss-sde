"""Prospective synthetic controls; no production object/physics/posteriors."""
import json,os,subprocess
from pathlib import Path
import numpy as np
import pytest
from experiments import campaign_surface_p2_study_v02 as study
from experiments import run_surface_p2_study_v03 as runner
from experiments.campaign_surface_probe_utility_v01 import covariance_utility


def test_streamed_covariance_matches_independent_dense_reference():
    rng=np.random.default_rng(25);unit=rng.uniform(size=(251,2));weights=rng.dirichlet(np.ones(len(unit)))
    labels=('A','B','C','T0','T1');basis=rng.normal(size=(5,7,5))*1e-3
    def predict(u,selected):
        features=np.column_stack((u,u*u,u[:,0]*u[:,1]))
        return np.einsum('bi,lqi->blq',features,basis[[labels.index(l)for l in selected]])
    values=predict(unit,labels)
    dense=[covariance_utility(weights,values[:,j],values[:,3:],noise_SD_m=study.SIGMA_M)for j in range(3)]
    for chunk in (1,37,1000):
        stream=study.streamed_utilities(unit,weights,predict,labels[:3],labels[3:],chunk=chunk)
        for a,b in zip(stream,dense):
            for k in a:assert np.isclose(a[k],b[k],rtol=2e-12,atol=1e-20)
    # Translation invariance uses centered, two-pass covariances.
    shifted=lambda u,l:predict(u,l)+.25
    shifted_scores=study.streamed_utilities(unit,weights,shifted,labels[:3],labels[3:],chunk=19)
    assert np.allclose([x['expected_linear_target_MSE_reduction_m2']for x in shifted_scores],[x['expected_linear_target_MSE_reduction_m2']for x in dense],rtol=1e-11,atol=1e-20)


def test_v02_force_budget_resultant_moment_and_disjoint_catalog():
    from experiments.campaign_surface_p2_static_v02 import geometry
    X=geometry(1)[0];loads=study.load_catalog(1)
    assert len(loads)==10
    assert set(study.TARGETS).isdisjoint(set(sum((list(x)for x in study.ACTIONS.values()),[])))
    for label,force in loads.items():
        axial=label.startswith('paired_axial');expected=np.array([.36,0,0])if axial else np.array([0,0,-study.NORMAL_FORCE_N])
        assert np.linalg.norm(force.sum(0)-expected)<1e-12
        center=int(label[-3:])/1000
        assert np.linalg.norm(np.cross(X-[center,.04,.04],force).sum(0))<1e-13
    assert study.design()['paired_axial_force_N_per_face']==.18


def test_original_streams_replay_paired_noise_and_object_independence(monkeypatch):
    for name,seed in zip(('COHORT_SEED','PREFIX_NOISE_SEED','PROBE_NOISE_SEED','RANDOM_ACTION_SEED'),(3100001,3100002,3100003,3100004)):
        monkeypatch.setattr(study,name,seed)
    first=runner.private_streams(0);again=runner.private_streams(0);second=runner.private_streams(1)
    for key in first:assert np.array_equal(first[key],again[key])
    assert not np.array_equal(first['prefix_noise'],second['prefix_noise'])
    assert not np.array_equal(first['probe_noise'][0],first['probe_noise'][1])
    assert first['probe_noise'].shape==(3,24)
    # Only object/repetition seeds exist: no action or law specific noise.
    assert set(first)=={'unit','prefix_noise','probe_noise','random_realization'}


def test_admission_fail_closed_on_missing_cases_or_force_mismatch(tmp_path):
    import hashlib
    file=tmp_path/'admission.json'
    record=dict(qualified=True,target_mesh_n=4,all50_cases_retained=True,all50_load_cases_qualified=True,
        source_SHA256={'experiments/campaign_surface_p2_study_v02.py':runner.sha(runner.ROOT/'experiments/campaign_surface_p2_study_v02.py')},protocol_SHA256='a'*64,force_SHA256={'test':'b'*64})
    def write():file.write_text(json.dumps(record));return hashlib.sha256(file.read_bytes()).hexdigest()
    assert runner.catalog_gate(file,write(),force_hashes={'test':'b'*64})['qualified']
    record['all50_cases_retained']=False
    with pytest.raises(ValueError,match='Full50'):runner.catalog_gate(file,write(),force_hashes={'test':'b'*64})
    record['all50_cases_retained']=True
    with pytest.raises(ValueError,match='catalog mismatch'):runner.catalog_gate(file,write(),force_hashes={'test':'c'*64})


def test_actual_generated_worker_commands_enforce_resource_environment():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for mode in ('cpu',):
        limits=runner.resources_for(mode)
        for slot in range(limits['workers']):
            command=runner.worker_command(slot,'0'*64,guard_only=True,device_mode=mode)
            childenv=dict(env,CUDA_VISIBLE_DEVICES=''if mode=='cpu'else limits['worker_GPU_UUIDs'][slot],CUBLAS_WORKSPACE_CONFIG=':4096:8')
            child=subprocess.run(command,cwd=runner.ROOT,env=childenv,capture_output=True,text=True,timeout=20)
            assert child.returncode==0,child.stderr
            assert json.loads(child.stdout)['guard_passed']
    bad=dict(env,CUDA_VISIBLE_DEVICES='0')
    child=subprocess.run(runner.worker_command(0,'0'*64,guard_only=True),cwd=runner.ROOT,env=bad,capture_output=True,text=True,timeout=20)
    assert child.returncode!=0 and 'guard' in child.stderr


def test_cluster_bootstrap_and_no_missing_object_filter():
    summary=study.paired_cluster_summary(np.arange(16)*1e-6)
    assert summary['objects']==16 and summary['bootstrap_repetitions']==10000
    assert summary['mean']==pytest.approx(7.5e-6)
    assert summary['CI95'][0]<summary['mean']<summary['CI95'][1]
    with pytest.raises(ValueError):study.paired_cluster_summary(np.arange(15))
    with pytest.raises(ValueError):study.paired_cluster_summary([0]*15+[np.nan])


def test_policy_exact_random_average_and_unresolved_propagation():
    choices={family:dict(proposed=2,information=1)for family in study.ACTIONS}
    joints={family:[[dict(resolved=True,posterior_RMS_m=float(10*i+r))for r in range(3)]for i in range(3)]for family in study.ACTIONS}
    scores=runner.policy_scores(choices,joints,random_realization=[0,1])
    for family in study.ACTIONS:
        assert scores[family]['proposed']['posterior_RMS_m']==21
        assert scores[family]['uniform_random']['posterior_RMS_m']==11
    joints['normal'][0][1]={'resolved':False}
    scores=runner.policy_scores(choices,joints,random_realization=[0,1])
    assert not scores['normal']['uniform_random']['resolved']
    assert scores['normal']['proposed']['resolved']


def test_reviewed_cost_release_bound_to_source_resources_and_protocol(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'LOCAL',tmp_path)
    cost=tmp_path/'tiny-cost.json';cost.write_text(json.dumps(dict(schema='surface-amplitude-cpu-cost-v01',completed=True,all_numerical_controls_qualified=True,recipe_SHA256='c'*64)))
    ownership=tmp_path/'cost-ownership.json';ownership.write_text(json.dumps(dict(completed=True,all_reaped=True,all_worker_exit_codes_zero=True,controller_exit_code=0,result_SHA256=runner.sha(cost),recipe_SHA256='c'*64)))
    (tmp_path/'protocol.json').write_text(json.dumps(dict(device_mode='cpu')))
    release=dict(approved=True,protocol_SHA256='a'*64,source_SHA256=runner.sources(),resources=runner.resources_for(),
        cost_control=dict(path=str(cost),SHA256=runner.sha(cost)),cost_ownership=dict(path=str(ownership),SHA256=runner.sha(ownership)))
    file=tmp_path/'reviewed-release.json';file.write_text(json.dumps(release))
    assert runner.release_gate('a'*64)['approved']
    release['resources']=dict(runner.resources_for(),workers=3)
    file.write_text(json.dumps(release))
    with pytest.raises(ValueError,match='release required'):runner.release_gate('a'*64)
    release['resources']=runner.resources_for();cost.write_text('changed')
    file.write_text(json.dumps(release))
    with pytest.raises(ValueError,match='cost-control'):runner.release_gate('a'*64)


def test_toy_posterior_checkpoint_replay_binds_actual_joint_data(tmp_path):
    from experiments.campaign_surface_inverse_reference_v01 import ModalStaticReference
    from experiments.campaign_surface_p2_inference_v01 import LabelledModal,IntegrationPlan
    labels=[study.PREFIX]+list(study.TARGETS)
    modal=ModalStaticReference(np.diag([1.,2.]),np.diag([.2,.3]),np.ones((2,len(labels)))*1e-4,np.eye(2))
    model=LabelledModal(modal,labels);data=[(study.PREFIX,np.array([1e-6,2e-6]))]
    path=tmp_path/'toy-posterior-LOCAL.json';plan=IntegrationPlan((17,33),(16,32),128)
    first=runner.one_posterior(path,model,data,plan)
    second=runner.one_posterior(path,model,data,plan)
    assert first==second and 'observations_SHA256' in first
    assert first['qualified']
    altered=[(study.PREFIX,np.array([1e-6,3e-6]))]
    with pytest.raises(ValueError,match='model/data changed'):runner.one_posterior(path,model,altered,plan)




def test_irregular_positive_hypothesis_prediction_and_streamed_policy_parity():
    from experiments.campaign_surface_conditional_amplitude_v01 import BOUNDS,LOG_WIDTH
    labels=study.ACTIONS['normal']+study.TARGETS;rng=np.random.default_rng(77)
    zr=np.log(np.array([.002,.003,.005,.008]));z=np.repeat(zr,7)
    lo=np.maximum(1/8000,np.exp(zr)/30);hi=np.minimum(1/2000,np.exp(zr)/5)
    a=np.concatenate([np.geomspace(l*1.001,h*.999,7)for l,h in zip(lo,hi)])
    unit=np.column_stack(((-np.log(a)-np.log(2000))/LOG_WIDTH[0],(z-np.log(a)-np.log(5))/LOG_WIDTH[1]))
    coefficients=rng.normal(size=(len(labels),3,2))
    def ratio(z,selected):
        features=np.column_stack((np.ones(len(z)),np.exp(z)))
        return np.einsum('bi,lqi->blq',features,coefficients[[labels.index(l)for l in selected]])
    calls=[]
    def fallback(u,selected):
        calls.append(len(u));E=np.exp(np.log(2000)+u[:,0]*LOG_WIDTH[0]);K=np.exp(np.log(5)+u[:,1]*LOG_WIDTH[1])
        return ratio(np.log(K/E),selected)/E[:,None,None]
    fast=study.weighted_hypothesis_predictor(unit,a,z,ratio,fallback)
    assert np.allclose(fast(unit[3:15],labels),fallback(unit[3:15],labels),rtol=1e-13,atol=1e-17)
    calls.clear();weights=rng.dirichlet(np.ones(len(a)))
    streamed=study.streamed_utilities(unit,weights,fast,labels[:3],labels[3:],chunk=5)
    assert calls==[]
    dense=a[:,None,None]*ratio(z,labels)
    for i,row in enumerate(streamed):
        expected=covariance_utility(weights,dense[:,i],dense[:,3:],noise_SD_m=study.SIGMA_M)
        for key in row:assert np.isclose(row[key],expected[key],rtol=1e-11,atol=1e-20)
    fast(unit[2:4].copy(),labels);assert calls==[2]


def test_conditional_checkpoint_preserves_irregular_hypotheses_and_baseline_choice(tmp_path):
    from experiments.campaign_surface_inverse_reference_v01 import ModalStaticReference
    from experiments.campaign_surface_p2_inference_v01 import LabelledModal,IntegrationPlan
    from experiments.campaign_surface_conditional_amplitude_v01 import AmplitudePlan
    labels=[study.PREFIX]+list(study.TARGETS)
    modal=ModalStaticReference(np.diag([1.,2.]),np.diag([.2,.3]),np.ones((2,len(labels)))*1e-4,np.eye(2))
    model=LabelledModal(modal,labels);data=[(study.PREFIX,np.array([1e-6,2e-6]))]
    path=tmp_path/'amplitude-LOCAL.json';tensor=IntegrationPlan((17,33),(16,32),128);amp=AmplitudePlan((8,16),(8,16))
    record=runner.one_posterior(path,model,data,tensor,numerical_engine='conditional_amplitude',amplitude_plan=amp)
    assert record['qualified']and record['diagnostics']['independent_adaptive']['passed']
    with np.load(path.with_suffix('.npz'))as z:
        assert z['unit_grid'].shape==(3*16*16,2)
        assert z['inverse_amplitude'].shape==z['quadrature_probability'].shape
        assert np.isclose(z['quadrature_probability'].sum(),1,atol=1e-12)
    with pytest.raises(ValueError,match='recipe changed'):runner.one_posterior(path,model,data,tensor)




def test_partial_truth_commits_refuse_tamper_context_and_orphans(tmp_path):
    path=tmp_path/'field-LOCAL.npz';context=dict(protocol_SHA256='a'*64,forces='fixed')
    runner.checkpoint_npz(path,context,fields=np.arange(9).reshape(3,3))
    runner.verify_npz(path,context)
    with pytest.raises(ValueError,match='context/checksum'):runner.verify_npz(path,dict(context,forces='changed'))
    with path.open('ab')as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='context/checksum'):runner.verify_npz(path,context)
    orphan=tmp_path/'orphan.npz';np.savez(orphan,fields=np.zeros(3))
    with pytest.raises(ValueError,match='Unbound partial'):runner.verify_npz(orphan,context)
    with pytest.raises(ValueError,match='before any solve'):runner.checkpoint_present(orphan)
    missing=tmp_path/'missing.npz';missing.with_suffix('.commit.json').write_text('{}')
    with pytest.raises(ValueError,match='before any solve'):runner.checkpoint_present(missing)
    assert runner.checkpoint_present(path)
    assert not runner.checkpoint_present(tmp_path/'absent.npz')
    with pytest.raises(ValueError,match='overwritten'):runner.checkpoint_npz(path,context,fields=np.zeros(3))


def test_CPU_only_resources_and_host_reserve(monkeypatch):
    limits=runner.resources_for('cpu')
    assert limits['workers']==16 and limits['worker_CPU_sets']==[[i]for i in range(16)]
    assert limits['controller_CPU']==16 and limits['aggregate_RSS_cap_bytes']==64*1024**3
    assert not limits['GPU'] and not limits['worker_GPU_UUIDs']
    with pytest.raises(ValueError,match='CPU-only'):runner.resources_for('gpu')
    monkeypatch.setattr(runner,'available_memory',lambda:93*1024**3)
    with pytest.raises(MemoryError):runner.host_preflight('cpu')
    monkeypatch.setattr(runner,'available_memory',lambda:94*1024**3)
    runner.host_preflight('cpu')
