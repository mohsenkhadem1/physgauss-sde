import numpy as np
from scripts.position_ood.smooth_fixture import SmoothMaterialFixture,SmoothDensePositionFEM,footprint
from scripts.position_ood.fixture import controls

def test_smooth_footprint_vanishes_at_perimeter():
    z=np.linspace(-.04,.04,21)
    for kind in ['lower','upper','yminus','yplus']:
        assert np.all(footprint(np.full_like(z,.04),z,kind)==0)
        assert np.all(footprint(z,np.full_like(z,-.04),kind)==0)
        assert footprint(0,0,kind)>0

def test_independent_dense_and_sparse_smooth_boundary_assembly():
    r=SmoothMaterialFixture(2,20000.,5000.,8000.,'crossed',device='cpu',material_gradient=.2,field_amplitude_J=.08)
    d=SmoothDensePositionFEM(2,'cpu');pose=controls('bend_z');r.set_controls(pose);theta=d.tensor([*np.log([20000.,5000.,8000.]),0,0,-9.81,.08,.2]);u=np.random.default_rng(123).normal(size=r.rest.shape)*1e-5
    sparse=r.evaluate(u,True);energy,residual,tangent=d.evaluate(d.tensor(u),theta,d.tensor(pose),1,True)
    np.testing.assert_allclose(float(energy),sparse['potential_J'],atol=1e-12)
    np.testing.assert_allclose(residual.numpy(),sparse['residual'],atol=1e-10)
    np.testing.assert_allclose(tangent.numpy(),sparse['tangent'].toarray(),atol=1e-8,rtol=1e-10)
    for left in [True,False]:
        M,w,area=r.patch(left,'lower');np.testing.assert_allclose(np.asarray(M.sum(1)).ravel(),w,atol=1e-18);np.testing.assert_allclose(w.sum(),area,rtol=1e-12)

def test_smooth_implicit_gradient_matches_resolved_perturbation():
    m=SmoothDensePositionFEM(1,'cpu');theta=m.tensor([*np.log([20000.,5000.,8000.]),0,0,-9.81,.08,.2]).requires_grad_();pose=m.tensor(controls('translate_y'));direction=m.tensor(np.random.default_rng(12).normal(size=m.rest.shape));value=(m.predict(theta,pose,1)*direction).sum();value.backward();epsilon=1e-5
    a=theta.detach().clone();b=a.clone();a[0]+=epsilon;b[0]-=epsilon
    finite=float(((m.predict(a,pose,1)-m.predict(b,pose,1))*direction).sum())/(2*epsilon)
    np.testing.assert_allclose(float(theta.grad[0]),finite,atol=1e-7,rtol=1e-4)
