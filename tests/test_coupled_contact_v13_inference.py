"""Independent pre-freeze tests of calibration isolation and forecast accounting."""
import numpy as np
import pytest
from scripts.coupled_contact_v13.inference import choose, calibrate
from scripts.coupled_contact_v13.scenarios import shared_node_indices
from scripts.coupled_contact_v13.physics import make_block
from scripts.coupled_contact_v13.pipeline import metrics


def sample():
    rest=np.array([[0.,0.,0.],[.08,0.,0.],[0.,.06,0.],[0.,0.,.05]])
    times=np.arange(5)*.1
    tools=np.zeros((5,2,3));tools[:,0,0]=np.arange(5)*.005
    active=np.ones((5,2))
    observed=np.broadcast_to(rest,(5,4,3)).copy();observed[:,:,0]+=tools[:,0,0,None]
    mask=np.ones((5,4),bool);mask[:,3]=False
    return rest,observed,mask,tools,times,active


def candidate(frames,scale=4500.,friction=.4):
    return dict(frames=frames,scale=scale,friction=friction)


def test_unobserved_values_cannot_change_fit_or_empirical_forecast():
    rest,obs,mask,tools,times,active=sample()
    libs={'fem':[candidate(obs.copy()),candidate(obs+.01,9000.,.8)]}
    hidden_nan=obs.copy();hidden_nan[~mask]=np.nan
    hidden_poison=obs.copy();hidden_poison[~mask]=1e12
    a,pa=calibrate(rest,hidden_nan,mask,tools,times,active,libs)
    b,pb=calibrate(rest,hidden_poison,mask,tools,times,active,libs)
    assert a==b
    newtools=tools.copy();newtools[:,:,1]=np.arange(5)[:,None]*.003
    for name,array in pa.predict(rest,newtools,times,active).items():
        np.testing.assert_array_equal(array,pb.predict(rest,newtools,times,active)[name])
    np.testing.assert_array_equal(pa.observed_counts,[5,5,5,0])


def test_failed_hypotheses_do_not_win_or_hide_valid_candidate():
    _,obs,mask,*_=sample()
    bad=obs.copy();bad[-1,3,0]=np.nan
    # A nonfinite simulated trajectory is a failure even on an unobserved node.
    rows=[candidate(None,2000.,0.),candidate(bad,3000.,.2),
          candidate(obs[:-1],4500.,.4),candidate(obs.copy(),6500.,.6)]
    result=choose(obs,mask,rows)
    assert result['success'] and result['index']==3
    assert all(np.isinf(result['scores_m2'][:3]))
    assert result['calibration_rmse_m']==0.


def test_all_failed_is_explicit_without_nominal_parameter_fallback():
    _,obs,mask,*_=sample()
    r=choose(obs,mask,[candidate(None),candidate(np.full_like(obs,np.nan),9000.,.8)])
    assert not r['success']
    assert 'index' not in r and 'scale' not in r
    assert len(r['scores_m2'])==2 and all(np.isinf(r['scores_m2']))


def test_empty_observation_set_cannot_select_physical_parameters():
    _,obs,mask,*_=sample();mask[:]=False
    with np.errstate(all='ignore'):
        r=choose(obs,mask,[candidate(obs.copy())])
    assert not r['success']


def test_ties_use_declared_grid_order_and_report_ambiguity():
    _,obs,mask,*_=sample()
    rows=[candidate(obs.copy(),2000.,0.),candidate(obs.copy(),6500.,.4),
          candidate(obs+.1,18000.,.8)]
    r=choose(obs,mask,rows)
    assert r['index']==0 and r['second_best_gap_m2']==0.
    assert r['scale_at_grid_edge'] and r['friction_at_grid_edge']


def test_interior_optimum_is_not_mislabeled_grid_edge():
    _,obs,mask,*_=sample()
    rows=[candidate(obs+.01,2000.,0.),candidate(obs.copy(),6500.,.4),
          candidate(obs+.02,18000.,.8)]
    r=choose(obs,mask,rows)
    assert r['index']==1
    assert not r['scale_at_grid_edge'] and not r['friction_at_grid_edge']
    assert r['second_best_gap_m2']>0


def test_fine_coarse_mapping_is_exact_and_rejects_shifted_mesh():
    fine,_=make_block((.08,.06,.05),(4,4,4),(0,0,.027))
    coarse,_=make_block((.08,.06,.05),(2,2,2),(0,0,.027))
    ids=shared_node_indices(fine,coarse)
    assert len(ids)==len(coarse) and len(np.unique(ids))==len(coarse)
    np.testing.assert_allclose(fine[ids],coarse,atol=1e-12,rtol=0)
    with pytest.raises(ValueError,match='exact shared'):
        shared_node_indices(fine,coarse+np.array([.001,0,0]))


def test_rigid_translation_affects_world_metric_but_not_shape_metric():
    _,target,*_=sample();prediction=target.copy();prediction[1:]+=np.array([.03,0,0])
    result=metrics(prediction,target)
    assert result['complete']
    assert result['coordinate_rmse_m']==pytest.approx(.03/np.sqrt(3))
    assert result['shape_rmse_m']<1e-12


def test_invalid_forecast_retains_explicit_failure():
    _,target,*_=sample();prediction=target.copy();prediction[-1,0,0]=np.nan
    result=metrics(prediction,target)
    assert not result['complete']
    assert result['coordinate_rmse_m'] is None and result['shape_rmse_m'] is None


def test_nonfinite_reference_target_cannot_produce_complete_nan_score():
    _,target,*_=sample();prediction=target.copy();target[-1,0,1]=np.nan
    result=metrics(prediction,target)
    assert not result['complete']
    assert result['coordinate_rmse_m'] is None and result['shape_rmse_m'] is None
    assert 'reference' in result['reason'].lower()


@pytest.mark.parametrize('fault',[None,'missing','nan','times','penetration','volume'])
def test_reference_qualification_checks_both_levels(tmp_path,monkeypatch,fault):
    import json
    from scripts.coupled_contact_v13 import pipeline
    monkeypatch.setattr(pipeline,'BASE',tmp_path)
    monkeypatch.setattr(pipeline,'protocol',lambda:dict(gates=dict(penetration_m=.001,trajectory_refinement_rms_m=.00025)))
    for level in ['reference','reference-refinement']:
        refined=level=='reference-refinement'
        if refined and fault=='missing':continue
        path=tmp_path/level/'test-object'/'squeeze';path.mkdir(parents=True)
        diagnostic=dict(max_penetration_m=.0001,min_j=.8)
        frames=np.zeros((3,4,3));times=np.array([0.,.01,.02])
        if refined:
            frames+=1e-5
            if fault=='nan':frames[-1,0,0]=np.nan
            if fault=='times':times[-1]+=.001
            if fault=='penetration':diagnostic['max_penetration_m']=.002
            if fault=='volume':diagnostic['min_j']=-.1
        (path/'complete.json').write_text(json.dumps(dict(success=True,diagnostics=diagnostic)))
        np.savez(path/'trajectory.npz',frames=frames,times=times)
    quality=pipeline.reference_quality('test-object','squeeze')
    assert quality['qualified']==(fault is None)
    if fault is None:assert quality['rms_m']==pytest.approx(1e-5)
    else:assert quality['reasons']


def test_independent_saved_volume_verifier_detects_affine_inversion():
    from scripts.coupled_contact_v13.verify import minimum_j
    rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    tet=np.array([[0,1,2,3]])
    stretch=np.diag([2.,.5,.75])
    assert minimum_j(np.array([rest@stretch]),rest,tet)==pytest.approx(.75)
    assert minimum_j(np.array([rest@np.diag([-1.,1.,1.])]),rest,tet)==pytest.approx(-1.)
    with pytest.raises(ValueError,match='reference tetrahedra'):
        minimum_j(np.array([rest]),rest,np.array([[0,1,1,3]]))
