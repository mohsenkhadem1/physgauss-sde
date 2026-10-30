import numpy as np
import torch
from scripts.contact_benchmark_v12.scene import StaticScene
from scripts.contact_benchmark_v12.scene_fem import SceneFEM,rollout


def tiny_data(T=6):
    nodes=np.array([[0,0,.5],[.04,0,.5],[0,.04,.5],[0,0,.54]],float)
    return dict(nodes=nodes,controller_targets=np.broadcast_to(nodes,(T,1,4,3)).copy(),
                attachment_weights=np.ones((1,4)),attachment_active=np.zeros((T,1)),floor=0.,
                times=np.arange(T)*.02,train_end=2),np.array([-4.,2.,5.,0.,0.])


def test_scene_friction_energy_gradient_hessian_consistent():
    data,theta=tiny_data();scene=StaticScene(floor=.501)
    m=SceneFEM(data,scene,field=False,friction=.3);th=m.t(theta)
    previous=m.rest.clone();m.prepare_step(previous,th,.01);m.inertial_target=previous
    x=(previous+m.t([.002,-.001,0])).requires_grad_()
    energy,r,H=m.evaluate(x,th,1,True)
    gradient=torch.autograd.grad(energy,x,create_graph=True)[0]
    hessian=torch.autograd.functional.hessian(lambda y:m.evaluate(y.reshape_as(x),th,1)[0],x.flatten())
    torch.testing.assert_close(r,gradient,rtol=1e-8,atol=1e-9)
    torch.testing.assert_close(H,hessian,rtol=1e-7,atol=1e-8)


def test_support_above_previous_nodes_not_teleported_onto():
    data,theta=tiny_data();v=np.array([[-1,-1,.7],[1,-1,.7],[1,1,.7],[-1,1,.7]])
    scene=StaticScene([dict(vertices=v,triangles=np.array([[0,1,2],[0,2,3]]))],floor=0.,spacing=.1)
    m=SceneFEM(data,scene,field=False);q=m.prepare_step(m.rest,m.t(theta),.01)
    np.testing.assert_array_equal(q['height'],np.zeros(4))


def test_freefall_refinement_and_no_horizontal_restoring_force():
    data,theta=tiny_data();base=np.broadcast_to(data['nodes'],(6,4,3)).copy();scene=StaticScene(floor=0.)
    predictions=[]
    for n in [1,2,4]:
        p,meta=rollout(data,base,theta,scene,start=2,field=False,friction=0.,substeps=(n,),initial_velocity=np.tile([.1,0,0],(4,1)))
        assert not meta['failed'];predictions.append(p)
        np.testing.assert_allclose(p[-1,:,0],base[1,:,0]+.1*.08,atol=1e-6)
    d12=np.linalg.norm(predictions[0][-1]-predictions[1][-1]);d24=np.linalg.norm(predictions[1][-1]-predictions[2][-1])
    assert d24<d12*.6


def test_free_mask_does_not_mutate_input_attachment_states():
    data,theta=tiny_data();data['attachment_active'][:]=1
    base=np.broadcast_to(data['nodes'],(6,4,3)).copy()
    rollout(data,base,theta,StaticScene(floor=0.),start=2,field=False,substeps=(1,),free_mask=np.ones(6,bool))
    np.testing.assert_array_equal(data['attachment_active'],np.ones((6,1)))


def test_floor_impact_changes_elastic_shape_not_only_rigid_pose():
    data,theta=tiny_data(T=12);data['nodes'][:,2]-=.49
    data['controller_targets'][:]=data['nodes'];theta[0]=-1.;theta[2]=7.
    base=np.broadcast_to(data['nodes'],(12,4,3)).copy()
    pred,meta=rollout(data,base,theta,StaticScene(floor=0.),field=False,substeps=(10,),friction=0.)
    assert not meta['failed']
    original=np.linalg.norm(base[0,3]-base[0,0]);final=np.linalg.norm(pred[-1,3]-pred[-1,0])
    assert final<original*.9
    assert pred[-1,:,2].min()>-.001


def test_release_field_disable_preserves_static_field_and_parameters():
    data,theta=tiny_data();theta[-1]=2.;scene=StaticScene(floor=0.)
    model=SceneFEM(data,scene,field=True,disable_field_on_release=True)
    baseline=SceneFEM(data,scene,field=True);no_field=SceneFEM(data,scene,field=False)
    th=model.t(theta);x=model.rest.clone()
    for actual,expected in zip(model.evaluate(x,th,1,True),baseline.evaluate(x,th,1,True)):
        torch.testing.assert_close(actual,expected)
    for m in (model,no_field):
        m.prepare_step(x,th,.01);m.inertial_target=x
    for actual,expected in zip(model.evaluate(x,th,1,True),no_field.evaluate(x,th,1,True)):
        torch.testing.assert_close(actual,expected)
    assert model.field is True
