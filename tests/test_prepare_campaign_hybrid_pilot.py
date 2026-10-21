"""Purecontrolaggregation/fixedselectiontests; no simulator or target read."""
import copy
import numpy as np
import pytest
from experiments.prepare_campaign_hybrid_pilot import bin_known_commands,select_sources,CASES,FAMILIES,HELD


def history(hz=120):
    times=np.arange(3*hz+1)/hz
    return np.stack([times[:-1],times[1:]],1),np.zeros((3*hz,2,3)),np.arange(61)/20


def test_constant_known_load_mean_and_centered_moment_have_declared_units():
    intervals,F,times=history();F[:]=[[.1,.2,.3],[-.1,-.2,-.3]]
    controls,error=bin_known_commands(intervals,F,times)
    assert error<1e-15 and not controls[0].any()
    np.testing.assert_allclose(controls[1:,:6],np.tile(F[0].reshape(6),(60,1)),atol=1e-15,rtol=0)
    np.testing.assert_allclose(controls[1:,6:],0,atol=2e-15,rtol=0)


def test_centered_moment_separates_order_with_identical_mean_and_is_causal():
    intervals,F,times=history();F[0:3,0,0]=1;F[3:6,0,0]=-1
    control,_=bin_known_commands(intervals,F,times)
    assert abs(control[1,0])<1e-15 and control[1,6]==pytest.approx(-.25,abs=1e-15)
    changed=F.copy();changed[180:]=100
    later,_=bin_known_commands(intervals,changed,times)
    np.testing.assert_array_equal(control[:31],later[:31])
    assert np.max(abs(control[31:]-later[31:]))>1


def test_missing_command_coverage_fails_without_interpolating():
    intervals,F,times=history();intervals[50,0]+=.001
    with pytest.raises(ValueError):bin_known_commands(intervals,F,times)


def test_fixed_highest_verified_sources_and_split_never_use_target_scores():
    rows=[]
    for case in CASES:
        for family in FAMILIES:
            if (case,family)==('t0000','twist_couple'):continue
            for hz in [120,240]:rows.append({'passed':True,'state':'verified','cell':{'case':case,'response':family,'hz':hz},'arbitrary_target_error':1e9 if hz==240 else 0})
    rows.append({'passed':False,'state':'failed_preserved','cell':{'case':'t0000','response':'pull_hold','hz':960}})
    d={'passed':True,'cell_audit':{'passed':True,'state':'verified','cell':{'case':'t0000','response':'twist_couple','hz':120}}}
    selected=select_sources({'cells':rows},d)
    assert len(selected)==16 and selected[('t0000','pull_hold')]['cell']['hz']==240
    assert sum(key not in HELD for key in selected)==12 and len(HELD)==4
    d['passed']=False
    with pytest.raises(ValueError):select_sources({'cells':rows},d)
