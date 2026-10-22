import numpy as np
import pytest
import torch
from experiments.campaign_sparse_nonlinear_gaussian import SparseNonlinearGaussian
from experiments.campaign_nonlinear_excitation import build_model,command

torch.set_num_threads(2)

@pytest.mark.parametrize('level',[0,1])
def test_analytic_energy_gradient_hessian_against_original_torch(level):
    sparse=SparseNonlinearGaussian.from_design(level);old=build_model(mesh_level=level)
    F=np.array([[1.07,.025,0],[.01,.96,.015],[0,0,1.02]])
    x=sparse.rest@F.T+np.array([.03,-.01,.02]);U,g,H,J=sparse.elastic(x,tangent=True)
    a=torch.tensor(x,dtype=torch.float64,requires_grad=True);energy,j=old.elastic(a)
    gradient=torch.autograd.grad(energy,a)[0]
    hessian=torch.autograd.functional.hessian(lambda z:old.elastic(z.reshape_as(a))[0],a.flatten())
    assert U==pytest.approx(float(energy.detach()),rel=1e-12,abs=1e-15)
    np.testing.assert_allclose(g,gradient.detach().numpy(),rtol=2e-11,atol=2e-12)
    np.testing.assert_allclose(H.toarray(),hessian.numpy(),rtol=1e-11,atol=3e-10)
    np.testing.assert_allclose(J,j.detach().numpy(),rtol=0,atol=2e-15)
    np.testing.assert_allclose(H.toarray(),H.toarray().T,rtol=0,atol=3e-12)

@pytest.mark.parametrize('level',[0,1])
def test_one_step_parity_at_rest_and_operating_strain(level):
    sparse=SparseNonlinearGaussian.from_design(level);old=build_model(mesh_level=level)
    target=np.repeat(sparse.rest[None],2,axis=0);force=sparse.known_force(command('balanced_axial',.1,1.))
    for F in [np.eye(3),np.array([[1.07,.025,0],[.01,.96,.015],[0,0,1.02]])]:
        x=sparse.rest@F.T;v=np.full_like(x,.001)
        q,vn,row=sparse.step(x,v,target,target,force,1/240)
        oldargs=[torch.tensor(a,dtype=torch.float64) for a in (x,v,target,target,force)]
        qt,vt,rt=old.step(*oldargs,1/240)
        np.testing.assert_allclose(q,qt.numpy(),rtol=0,atol=2e-12)
        np.testing.assert_allclose(vn,vt.numpy(),rtol=0,atol=5e-10)
        assert row['force_residual_N']<=row['threshold_N']
        assert abs(row['balance_error_J']-row['free_residual_work_J'])<2e-15


def test_constraints_work_and_positive_J_rejection():
    base=SparseNonlinearGaussian.from_design();mask=np.zeros_like(base.rest,dtype=bool);mask[0]=True
    model=SparseNonlinearGaussian.from_design(constraint_mask=mask)
    targets=np.repeat(model.rest[None],2,axis=0);new=targets+np.array([.0001,0,0]);prescribed=model.rest+np.array([.00005,0,0])
    q,v,row=model.step(model.rest,np.zeros_like(model.rest),targets,new,np.zeros_like(model.rest),.01,constraint_positions=prescribed)
    assert np.array_equal(q[mask],prescribed[mask]);assert row['constraint_max_error_m']==0
    assert row['target_work_J']!=0 and row['reaction_work_J']!=0
    assert row['damping_dissipation_J']>=0
    assert abs(row['balance_error_J']-row['free_residual_work_J'])<2e-15
    with pytest.raises(ValueError,match='no repair'):model.elastic(model.rest*np.array([-1,1,1]))


def test_noise_referenced_gate_is_distinct_from_development_gate(tmp_path):
    from experiments.run_campaign_nonlinear_resolution import noise_checks,compare_mesh,prepare
    q=np.zeros((49,5,3));fine={'positions_m':q.copy(),'rest_m':np.zeros((5,3))}
    coarse={'positions_m':q.copy(),'rest_m':np.zeros((5,3))};coarse['positions_m'][:,[1,4]]=5e-6
    checks=noise_checks(coarse,fine,240,240)
    assert not checks['displacement20um_0p1sigma']['passed']
    assert checks['depth200um_coordinateproxy_0p1sigma']['passed']
    p=prepare(tmp_path/'plan.json')
    assert p['wall_cap_s']==900 and p['mesh']['tet_counts']==[128,1024,8192]
    assert p['temporal']['conditional_final_rate_hz']==1920
    assert p['excitation_target']['no_amplitude_retuning']
