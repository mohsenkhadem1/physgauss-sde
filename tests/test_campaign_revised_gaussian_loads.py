from experiments.run_campaign_revised_gaussian_loads import select_screen,fine_comparison
import numpy as np


def row(T,strain,nonrigid,completed=True):return dict(duration_s=T,maximum_absolute_principal_stretch_strain=strain,peak_nonrigid_coordinate_RMS_m=nonrigid,completed=completed,minimum_J=1.)


def test_screen_selection_never_replaces_failed_excitation_or_searches_amplitude():
    assert select_screen([row(.4,.09,.003),row(.8,.06,.0009)]) is None
    assert select_screen([row(.4,.04,.0011),row(.8,.06,.0015)])==.4
    assert select_screen([row(.4,.04,.0011,False),row(.8,.06,.0015)])==.8
    assert select_screen([row(.4,.03,.0011),row(.8,.05,.0015)])==.8


def test_original_and_drivephase_noise_checks_do_not_replace_each_other():
    q=np.zeros((97,5,3));b={'positions_m':q.copy(),'rest_m':np.zeros((5,3))};a={'positions_m':q.copy(),'rest_m':np.zeros((5,3))}
    # Difference only at late peak/end: original early-time diagnostic must not
    # masquerade as whole-drive numerical accuracy.
    a['positions_m'][72,1]=1e-4
    meta={'completed':True,'tetrahedra':128};fine={'completed':True,'tetrahedra':1024}
    d=fine_comparison((meta,a),(fine,b),.4)
    assert d['noise_checks']['original_query_times']['displacement20um_0p1sigma_passed']
    assert not d['noise_checks']['drive_fraction_times']['displacement20um_0p1sigma_passed']
    assert not d['passed']


def test_timeout_preserves_partial_and_never_launches_second_screen(tmp_path,monkeypatch):
    import pytest
    import json
    import experiments.run_campaign_revised_gaussian_loads as runner
    class Dummy:
        rest=np.zeros((4,3));tets=np.array([[0,1,2,3]]);node_mass=np.ones(4)
        def known_force(self,F):return np.zeros((4,3))
        def step(self,*args):raise TimeoutError('test-budget')
    monkeypatch.setattr(runner,'make_model',lambda level:Dummy())
    monkeypatch.setattr(runner,'model_api',lambda:(None,None,lambda t,A,T:np.zeros((2,3))))
    with pytest.raises(TimeoutError):runner.capture(2,.4,tmp_path/'partial')
    partial=json.loads((tmp_path/'partial.json').read_text())
    assert not partial['summary']['completed'] and partial['summary']['steps_completed']==0
    assert (tmp_path/'partial.npz').exists()
    with pytest.raises(ValueError,match='nosuffix'):runner.capture(2,.4,tmp_path/'decimal-T0.4')
    calls=[]
    def timed_capture(*args):calls.append(args);raise TimeoutError('test-budget')
    monkeypatch.setattr(runner,'capture',timed_capture)
    monkeypatch.setattr(runner,'validate',lambda *a:{'screen':{'durations_s':[.4,.8]}})
    result=runner.execute('unused','unused',tmp_path/'execution','screen')
    assert len(calls)==1 and not result['completed']
    assert 'TimeoutError' in result['execution_failure']
