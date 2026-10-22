import json
import numpy as np
import pytest
from experiments import run_campaign_c38_beta_control as r


def test_test_release_blocks_partial_or_negative_seed_without_reading_data(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path);out=tmp_path/'study/validation';out.mkdir(parents=True)
    proto={'path':'protocol.json','sha256':'a'*64}
    roles={role:{'seeds':{str(seed):{'passed':True} for seed in r.FIT_SEEDS}} for role in ('DEV','CAL')}
    summary={'completed':True,'all_gates_passed':True,'protocol':proto,'roles':roles}
    r.write(out/'summary.json',summary);release={'status':'DEV_CAL_ALL3_PASS_TEST_PREFIX_RELEASED','protocol':proto,'validation':r.record(out/'summary.json')};r.write(out/'test-release.json',release)
    monkeypatch.setattr(r,'fitted',lambda *_:None)
    assert r.test_release({'output':'study'},proto)['path']=='study/validation/test-release.json'
    roles['CAL']['seeds'][str(r.FIT_SEEDS[0])]['passed']=False;r.write(out/'summary.json',summary);release['validation']=r.record(out/'summary.json');r.write(out/'test-release.json',release)
    with pytest.raises(ValueError,match='seeds/roles'):r.test_release({'output':'study'},proto)
    # An all-pass bit cannot rescue a missing seed or stale report digest.
    del roles['CAL']['seeds'][str(r.FIT_SEEDS[0])];r.write(out/'summary.json',summary)
    with pytest.raises(ValueError,match='Changed artifact'):r.test_release({'output':'study'},proto)


def test_prefix_choice_stage_never_materializes_test_outcomes_or_labels(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setitem(r.ROLE_COUNTS,'TEST',2)
    prefix={'features':np.zeros((2,10)),'camera':np.array([0,1])};seen=[]
    monkeypatch.setattr(r,'test_release',lambda *_:{'path':'release','sha256':'a'*64})
    monkeypatch.setattr(r,'dataset',lambda *_:{'roles':{'TEST':{'prefix_only':{'path':'prefix'},'public':{'path':'DO_NOT_OPEN_ACTIONS'},'private':{'path':'DO_NOT_OPEN_LABELS'}}}})
    monkeypatch.setattr(r,'fitted',lambda *_:{'seeds':{str(s):{'final_checkpoint':{}} for s in r.FIT_SEEDS}})
    def load(rec):
        seen.append(rec['path']);assert rec['path']=='prefix';return prefix
    monkeypatch.setattr(r,'load_arrays',load)
    monkeypatch.setattr(r,'infer',lambda *_:dict(logits=np.zeros((2,16)),alpha=np.ones((2,16,2)),beta=np.ones((2,16,2))))
    def choice(*_,**kwargs):
        # Simulated public model utility, no object truth or acquired outcome.
        return {'policies':{'fixed':0,'random':1,'information_gain':0,'predictive_risk':1}}
    monkeypatch.setattr(r,'select_reset_actions',choice)
    p={'output':'study','policy':{'quadrature_order':2,'fantasies':64,'choice_seed':3}}
    r.choose_test(p,{'path':'protocol','sha256':'a'*64})
    assert seen==['prefix']
    barrier=json.loads((tmp_path/'study/test-choices/barrier.json').read_text());assert barrier['completed'] and barrier['action_outcomes_opened'] is False and barrier['target_labels_opened'] is False
    for rec in barrier['seeds'].values():
        value=json.loads(r.checked(rec).read_text());assert [x['policies']['opposite_endpoint'] for x in value['choices']]==[1,0]


def test_posterior_freeze_requires_all_seed_choices_and_preserves_labels_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path);proto={'path':'protocol','sha256':'a'*64};choice={'path':'choice','sha256':'b'*64};out=tmp_path/'study/test-states';out.mkdir(parents=True)
    monkeypatch.setattr(r,'checked_choices',lambda *_:({},choice))
    states={'completed':True,'protocol':proto,'choices':choice,'target_labels_opened':False,'seeds':{}}
    r.write(out/'barrier.json',states)
    with pytest.raises(ValueError,match='All test posterior'):r.checked_test_states({'output':'study'},proto)
    for seed in r.FIT_SEEDS:
        path=out/f'{seed}.txt';path.write_text('synthetic');states['seeds'][str(seed)]=r.record(path)
    r.write(out/'barrier.json',states);assert r.checked_test_states({'output':'study'},proto)[0]['completed']
    states['target_labels_opened']=True;r.write(out/'barrier.json',states)
    with pytest.raises(ValueError):r.checked_test_states({'output':'study'},proto)


def test_changed_artifact_and_failed_output_never_reused(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path);path=tmp_path/'source.txt';path.write_text('original');record=r.record(path);assert r.checked(record)==path
    path.write_text('changed')
    with pytest.raises(ValueError):r.checked(record)
    r.fresh(tmp_path/'failed')
    with pytest.raises(FileExistsError):r.fresh(tmp_path/'failed')


def test_resource_limits_fail_before_data_or_gpu_calls(tmp_path,monkeypatch):
    from collections import namedtuple
    monkeypatch.setattr(r,'ROOT',tmp_path);p={'resources':{'process_RSS_cap_bytes':10**30,'disk_free_reserve_bytes':30*1024**3,'GPU_allocated_cap_bytes':8*1024**3}}
    Disk=namedtuple('Disk','total used free');monkeypatch.setattr(r.shutil,'disk_usage',lambda _:Disk(100,99,1))
    with pytest.raises(OSError,match='disk reserve'):r.guard_resources(p)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda _:Disk(10**20,0,10**20));p['resources']['process_RSS_cap_bytes']=1
    with pytest.raises(MemoryError,match='RSS cap'):r.guard_resources(p)
