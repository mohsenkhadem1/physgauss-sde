import json,os,subprocess,sys
import numpy as np
import pytest
from experiments import run_action_fisher_screen_v01 as run
from experiments.campaign_action_fisher_screen_v01 import screen


def test_independent_fd_stencils_inside_prior_include_both_boundaries():
    q=np.array([[0.,1.],[.25,.75],[.5,.5],[1.,0.]])
    def response(x):
        assert np.all((x>=0)&(x<=1))
        return (x[:,0]**4+3*x[:,0]*x[:,1]**2)[:,None]
    J=np.stack((4*q[:,0]**3+3*q[:,1]**2,6*q[:,0]*q[:,1]),axis=-1)[:,None]
    for h in(1e-3,5e-4):np.testing.assert_allclose(run.finite_difference(response,q,h),J,atol=5e-12)


def test_run_rejects_before_atlas_or_model_or_any_rng(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'guard',lambda :None);monkeypatch.setattr(run,'LOCAL',tmp_path)
    monkeypatch.setattr(run,'validate',lambda _:dict())
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('No archive before root release'))
    monkeypatch.setattr(run.model_tools,'make_model',lambda *a,**k:pytest.fail('No model before release'))
    monkeypatch.setattr(run.np.random,'default_rng',lambda *a:pytest.fail('Never draw screen RNG'))
    with pytest.raises(FileNotFoundError):run.run('no-release')


def test_prepare_validate_actual_JSON_roundtrip_metadata_only(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'guard',lambda :None);monkeypatch.setattr(run,'LOCAL',tmp_path);monkeypatch.setattr(run,'sources',lambda :{'toy':'abc'})
    monkeypatch.setattr(run.cost,'completed_cost_gate',lambda _:dict(memo=dict(model_fingerprint='toy')))
    origin=dict(origin_v01=dict(full_atlas={}),full_protocol_SHA256='protocol',full_atlas_fingerprint='atlas',all_target_labels=['toy'])
    monkeypatch.setattr(run.cost,'validate',lambda _:(origin,{}));monkeypatch.setattr(run,'record',lambda p:dict(path=str(p),sha256=run.sha(p)))
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('No atlas or scientificdata in metadata prepare'))
    prepared=run.prepare();p=run.validate(prepared['sha256']);assert p['math']==run.MATH and p['resources']==run.RESOURCES


def test_all81_refinement_disagreements_and_ties_retained():
    P=np.zeros((81,24,2));A=np.zeros((81,6,24,2));T=np.ones((81,96,2))*1e-4
    coarse=screen(P,A,T);fine=screen(P,A,T)
    fine['winner_mask'][17,0]=False
    report=run.aggregate({5:coarse,9:fine})
    assert report['n5_n9_rank_disagreement_points']==1
    assert report['5']['tied_points']==81 and report['9']['tied_points']==81
    assert report['5']['winner_membership_counts'][run.ACTIONS[0]]==81
    assert report['9']['winner_membership_counts'][run.ACTIONS[0]]==80
    assert report['all81_refinement_comparisons_retained']


def test_generated_CPU17_guard_without_data_or_gpu_context():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    out=subprocess.run(['taskset','-c','17',sys.executable,'-m','experiments.run_action_fisher_screen_v01','guard'],cwd=run.ROOT,env=env,capture_output=True,text=True,timeout=20)
    assert out.returncode==0,out.stderr
