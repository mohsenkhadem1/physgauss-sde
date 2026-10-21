from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from campaign_free_deformable_control import volume_weights,mass_momentum_fit,cotranslation_control


def test_volume_lumping_and_COM_translation_are_exact_for_known_tet():
    points=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    tets=np.array([[0,1,2,3]])
    w,volume,v=volume_weights(points,tets)
    np.testing.assert_allclose(w,np.full(4,.25));assert volume==pytest.approx(1/6)
    shifted=points+[.2,-.7,.4];w2,v2,_=volume_weights(shifted,tets)
    np.testing.assert_allclose(w2,w);assert v2==pytest.approx(volume)
    np.testing.assert_allclose(w2@shifted-w@points,[.2,-.7,.4])
    with pytest.raises(ValueError):volume_weights(points,np.array([[0,2,1,3]]))


def test_scalar_mass_fit_recovers_known_mass_with_nontrivial_two_handle_motion():
    t=np.linspace(0,3,40);M=1.8;masses=np.array([.001,.002])
    vb=np.column_stack([t,t*t,np.sin(t)])*.001
    vr=np.stack([vb+np.column_stack([t*.1,t*.05,t*.02]),vb-np.column_stack([t*.03,t*.02,t*.04])],axis=1)
    impulse=M*vb+np.einsum('thc,h->tc',vr,masses)
    result=mass_momentum_fit(vb+[.2,.3,.4],vr+np.array([[[.1,.2,.3],[.4,.5,.6]]]),masses,impulse,M)
    assert result['fitted_body_mass_kg']==pytest.approx(M,rel=1e-12)
    assert result['maximum_declared_mass_momentum_residual_Ns']<1e-15
    wrong=mass_momentum_fit(vb,vr,masses,impulse,2*M)
    assert wrong['fitted_to_declared_mass_ratio']==pytest.approx(.5)


def test_no_COM_excitation_has_no_identifiable_mass_instead_of_fabricated_result():
    with pytest.raises(ValueError,match='No identifiable'):
        mass_momentum_fit(np.zeros((4,3)),np.zeros((4,2,3)),[.001,.001],np.zeros((4,3)),1.8)


def test_cotranslation_proxy_gate_rejects_background_momentum_loss():
    v=np.tile([.001,0,0],(40,1));rigid=np.broadcast_to(v[:,None,:],(40,2,3)).copy()
    assert cotranslation_control(v,rigid,[.001,.001],1.8)['passed']
    decay=np.exp(-.6*np.linspace(0,3,40))
    result=cotranslation_control(v*decay[:,None],rigid*decay[:,None,None],[.001,.001],1.8)
    assert not result['passed'] and result['maximum_relative_proxy_total_momentum_drift']>.8
