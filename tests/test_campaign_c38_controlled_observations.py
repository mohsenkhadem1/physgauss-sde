"""Manufactured responses/fake RNG/identity entropy ONLY; no production draws."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import pytest
from experiments import campaign_c38_controlled_observations as m
from experiments import prepare_campaign_c38_point_table as table
from test_prepare_campaign_c38_point_table import fixtures
ROOT=Path(__file__).resolve().parents[1]


def design():return m.load_frozen_design(ROOT)
class FakeFactory:
    def __init__(self):self.entropies=[]
    def __call__(self,seed):
        self.entropies.append(list(seed.entropy));value=seed.entropy[-1]+1
        class Fake:
            def standard_normal(self,shape):return np.full(shape,value,dtype=float)
        return Fake()


def fake_bound(index=0,future_offset=0.):
    q={F:table.POINTS+F*.0001 for F in (-2.,0.,1.,2.)};q[1.]=q[1.]+future_offset;q[-2.]=q[-2.]+future_offset
    return m.BoundPrivateResponses(index,q,{'manufactured':True})


def test_exact_frozen_seed_channels_and_public_private_separation_no_actual_RNG(monkeypatch):
    monkeypatch.setattr(np.random,'default_rng',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Production RNG called')))
    factory=FakeFactory();public,private=m.private_repeat(0,3,'a'*32,fake_bound(),design=design(),rng_factory=factory)
    assert factory.entropies==[[2026102301,0,3,k] for k in range(3)]
    assert set(public)==m.PUBLIC_KEYS and public['observed_positions_m'].shape==(1,9,3)
    np.testing.assert_allclose(public['observed_positions_m'][0],table.POINTS[m.POINT_IDS]+.0002)
    assert not {'young_Pa','support_category','private_configuration_index','repeat_index','noise_seed_recipe','source_records','canonical_target_positions_m','candidate_probe_observations_m'}&set(public)
    assert 'PRIVATE_' in private['information_role'] and private['candidate_probe_observations_m'].shape==(2,9,3)
    assert private['canonical_target_positions_m'].shape==(63,3)
    assert not private['independent_population_object'] and not public['is_RGBD_perception'] and not public['is_temporal_forecast']


def test_future_counterfactual_changes_leave_public_prefix_identical():
    p1,f1=m.private_repeat(1,5,'b'*32,fake_bound(1),design=design(),rng_factory=FakeFactory())
    p2,f2=m.private_repeat(1,5,'b'*32,fake_bound(1,10),design=design(),rng_factory=FakeFactory())
    for key in p1:
        if isinstance(p1[key],np.ndarray):assert np.array_equal(p1[key],p2[key])
        else:assert p1[key]==p2[key]
    assert not np.array_equal(f1['canonical_target_positions_m'],f2['canonical_target_positions_m'])
    assert not np.array_equal(f1['candidate_probe_observations_m'],f2['candidate_probe_observations_m'])


def test_public_allowlist_point_scope_future_extra_payloads_and_noise_retuning_rejected():
    p,_=m.private_repeat(0,0,'c'*32,fake_bound(),design=design(),rng_factory=FakeFactory())
    bad=dict(p,young_Pa=90000.)
    with pytest.raises(ValueError,match='allowlist'):m.validate_public_prefix(bad,design=design())
    bad=dict(p,observed_positions_m=np.zeros((2,9,3)))
    with pytest.raises(ValueError,match='covariance'):m.validate_public_prefix(bad,design=design())
    d=copy.deepcopy(design());d['sensor']['qualified_point_ids'][0]=0
    with pytest.raises(ValueError,match='nine reviewed'):m.public_prefix_payload('c'*32,np.zeros((9,3)),design=d)
    d=copy.deepcopy(design());d['sensor']['noise']['sigma_m']=.0001
    with pytest.raises(ValueError,match='Frozen200um'):m.public_prefix_payload('c'*32,np.zeros((9,3)),design=d)
    with pytest.raises(ValueError,match='bound private'):m.private_repeat(0,0,'c'*32,fake_bound().responses_by_Fz_N,design=design(),rng_factory=FakeFactory())


def test_identity_entropy_has_no_physics_arguments_no_redraw_and_public_order_is_opaque():
    counter=iter(range(1,65));ids=m.mint_opaque_ids(64,token_hex=lambda n:format(next(counter),'032x'))
    p=[m.public_prefix_payload(i,np.zeros((9,3)),design=design()) for i in ids[::-1]]
    ordered=m.order_public_prefixes(p,design=design());assert [r['object_id'] for r in ordered]==ids
    with pytest.raises(ValueError,match='no redraw'):m.mint_opaque_ids(2,token_hex=lambda n:'d'*32)
    with pytest.raises(ValueError,match='All64'):m.order_public_prefixes(p[:-1],design=design())
    with pytest.raises(ValueError,match='Opaque'):m.public_prefix_payload('Elo_bothends',np.zeros((9,3)),design=design())


def test_private_point_archive_and_posterior_control_receipt_binding_before_arrays(tmp_path,monkeypatch):
    summary,_,_=fixtures(tmp_path);rows=table.load_point_rows(table.plan_completed_rows(summary,root=tmp_path),root=tmp_path);archives=table.split_archives(rows)
    path=tmp_path/'direct.npz';np.savez(path,**archives['WITHHELD_INTERPOLATION_VALIDATION'])
    def rec(p):return dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    a=rec(path);vpath=tmp_path/'validation.json';vpath.write_text(json.dumps(dict(completed=True,validation=dict(finite_checks_passed=True,tolerance_m=1e-6),archives={'WITHHELD_INTERPOLATION_VALIDATION':a})));vr=rec(vpath)
    cpath=tmp_path/'posterior_controls.json';cpath.write_text(json.dumps(dict(completed=True,all_numerical_controls_passed=True,design_sha256=m.DESIGN_SHA256)));cr=rec(cpath)
    bound=m.load_qualified_private_responses(tmp_path,a,vr,cr,private_configuration_index=2)
    assert isinstance(bound,m.BoundPrivateResponses) and set(bound.responses_by_Fz_N)=={-2.,0.,1.,2.}
    monkeypatch.setattr(np,'load',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Private arrays read before controls')))
    cpath.write_text(json.dumps(dict(completed=True,all_numerical_controls_passed=False,design_sha256=m.DESIGN_SHA256)));cr=rec(cpath)
    with pytest.raises(ValueError,match='posterior controls'):m.load_qualified_private_responses(tmp_path,a,vr,cr,private_configuration_index=2)
