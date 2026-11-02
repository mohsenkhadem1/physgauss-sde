import numpy as np
from scripts.compositional_neural_mechanics.gauss_newton_inference import GaussNewtonMAP
from scripts.compositional_neural_mechanics.joint_inference import JointMAP,neutral_parameters
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.scenes import make_model
from scripts.compositional_neural_mechanics.inference import observation_matrix


def fixture(wrench=True):
    p=np.zeros((2,6));p[1,0]=.004;p[1,5]=.08
    points=np.array([[.2,.3,.4],[.5,.7,.6],[.8,.4,.2]])
    o=dict(pose=p,points=points,displacement=np.zeros_like(points))
    if wrench:o.update(forces=np.zeros((2,3)),torques=np.zeros((2,3)))
    e=JointMAP(GeometrySpec(taper=.1,arch=.003),bulk_family='mooney_rivlin',interface_family=0,heterogeneity=0,observations=[o],position_sigma=.001,force_sigma=.1,torque_sigma=.01,n=1,continuation=(.5,1.),gravity=(0,0,-9.81))
    return e


def test_exact_objective_gradient_and_residual_finite_differences():
    e=fixture();gn=GaussNewtonMAP(e);z=neutral_parameters();z[2]=.1;active=np.array([0,3,7])
    r,J,_=gn.residual_jacobian(z,active);value,gradient,_=e.value_gradient(z)
    np.testing.assert_allclose(.5*r@r,value,atol=1e-9,rtol=1e-11)
    np.testing.assert_allclose(J.T@r,gradient[active],atol=1e-8,rtol=1e-9)
    for j,index in enumerate(active):
        plus=z.copy();minus=z.copy();plus[index]+=1e-5;minus[index]-=1e-5
        rp=gn.residual_jacobian(plus,active)[0];rm=gn.residual_jacobian(minus,active)[0]
        np.testing.assert_allclose((rp-rm)/2e-5,J[:,j],atol=2e-6,rtol=2e-5)


def test_small_fit_and_budget_preservation():
    e=fixture(False);truth=neutral_parameters();truth[0]=.15
    m=make_model(e.scene,**e.kw);e.bind(m,__import__('torch').tensor(truth));m.set_controls(e.observations[0]['pose']);s=m.solve(force_tolerance_N=1e-10)
    e.observations[0]['displacement']=observation_matrix(m,e.observations[0]['points'])@s['displacement_local_m']
    gn=GaussNewtonMAP(e);initial=.5*np.sum(gn.residual_jacobian(neutral_parameters(),[0])[0]**2)
    fitted=gn.fit(active=[0],max_evaluations=12,seconds=30)
    assert fitted['loss']<initial and fitted['optimizer_success'] and abs(fitted['parameters'][0]-.15)<.02
    incomplete=gn.fit(active=[0],max_evaluations=1,seconds=30)
    assert incomplete['incomplete'] and len(incomplete['evaluations'])==1 and incomplete['loss'] is not None
