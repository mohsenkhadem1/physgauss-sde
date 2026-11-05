import numpy as np
import pytest
import torch
from scripts.external_benchmark.dynamic_fem import DynamicFEM
from scripts.gaussian_boundary_flow_v02.object_dynamics import falling_cube_payload
from scripts.gaussian_boundary_flow_v02.elastic_continuation import continue_elastic


def fixture():
    p=falling_cube_payload();m=DynamicFEM(p,device='cpu',field=False)
    t=torch.zeros(m.i_field+1,dtype=torch.float64);t[0]=np.log(.03);t[m.i_floor]=-torch.inf
    return p,m,t


def test_arbitrary_start_freefall_and_model_restoration():
    p,m,t=fixture();q=p['nodes']+[.2,-.1,.3];v=np.tile([.1,-.2,.3],(8,1));rest=m.rest.clone();targets=m.targets;active=m.active
    times=np.array([1.3,1.31,1.32]);a=continue_elastic(m,t,q,v,1.3,times,max_dt=.001)
    dt=times-1.3;truth=q[None]+dt[:,None,None]*v[None];truth[:,:,2]-=.5*9.81*dt[:,None]**2
    np.testing.assert_allclose(a['positions_m'],truth,atol=1e-10)
    assert torch.equal(m.rest,rest)and m.targets is targets and m.active is active and m.inertial_target is None


def test_output_sampling_not_steps_and_strain_is_not_reset():
    p,m,t=fixture();q=p['nodes'].copy();q[:,0]*=1.05;v=np.zeros_like(q)
    full=continue_elastic(m,t,q,v,0,np.arange(4)*.01,max_dt=.001)
    sparse=continue_elastic(m,t,q,v,0,np.array([0.,.03]),max_dt=.001)
    np.testing.assert_array_equal(sparse['positions_m'],full['positions_m'][[0,-1]])
    np.testing.assert_allclose(full['positions_m'][0],q,atol=1e-15)
    assert np.linalg.norm((full['positions_m'][-1]-full['positions_m'][-1].mean(0))-(q-q.mean(0)))>1e-5


def test_floor_event_fails_and_restores_model():
    p,m,t=fixture();q=p['nodes']-[0,0,.199];v=np.zeros_like(q);v[:,2]=-1.
    with pytest.raises(RuntimeError,match='Unsupported plane contact'):
        continue_elastic(m,t,q,v,0,[0.,.01],max_dt=.001)
    assert m.inertial_target is None


def test_mass_weighted_freefall_and_angular_momentum_with_elastic_relaxation():
    p,m,t=fixture();q=p['nodes'].copy();q[:,0]*=1.05;mass=m.mass.numpy();center=(mass[:,None]*q).sum(0)
    v=np.cross(np.array([0.,0.,2.]),q-center)+[.1,0,0]
    a=continue_elastic(m,t,q,v,0,[0.,.02],max_dt=.0005)
    qt=a['positions_m'][-1];vt=a['velocities_m_s'][-1];c=(mass[:,None]*qt).sum(0)
    np.testing.assert_allclose(c,center+np.array([.1*.02,0,-.5*9.81*.02**2]),atol=1e-10)
    L0=(mass[:,None]*np.cross(q-center,v-(mass[:,None]*v).sum(0))).sum(0)
    L1=(mass[:,None]*np.cross(qt-c,vt-(mass[:,None]*vt).sum(0))).sum(0)
    np.testing.assert_allclose(L1,L0,atol=1e-10)


def test_release_event_matches_explicit_state_handoff_without_early_switch():
    p=falling_cube_payload();p['attachment_weights']=np.ones((1,8));p['attachment_active']=np.ones((len(p['times']),1));p['controller_targets']=np.broadcast_to(p['nodes'],(len(p['times']),1,8,3)).copy()
    m=DynamicFEM(p,device='cpu',field=False);t=torch.zeros(m.i_field+1,dtype=m.dtype);t[0]=np.log(.03);t[m.i_floor]=-torch.inf
    commands=np.broadcast_to(p['nodes'],(3,1,8,3));clock=[0.,.01,.02];gates=[[1.],[0.],[0.]]
    a=continue_elastic(m,t,p['nodes'],np.zeros((8,3)),0,[0.,.01,.02],max_dt=.001,command_times_s=clock,targets=commands,activation=gates)
    held=continue_elastic(m,t,p['nodes'],np.zeros((8,3)),0,[0.,.01],max_dt=.001)
    np.testing.assert_allclose(a['positions_m'][1],held['positions_m'][-1],atol=1e-13)
    free=continue_elastic(m,t,held['positions_m'][-1],held['velocities_m_s'][-1],.01,[.01,.02],max_dt=.001,command_times_s=[.01,.02],targets=commands[:2],activation=[[0.],[0.]])
    np.testing.assert_allclose(a['positions_m'][-1],free['positions_m'][-1],atol=1e-13)
    np.testing.assert_allclose(a['velocities_m_s'][-1],free['velocities_m_s'][-1],atol=1e-11)


def test_existing_natural_metric_is_preserved_and_gravity_unchanged():
    from scripts.gaussian_boundary_flow_v02.object_prestrain import InitialPrestrainFEM
    class NaturalDynamic(InitialPrestrainFEM,DynamicFEM):pass
    p=falling_cube_payload();m=NaturalDynamic(p,device='cpu',field=False)
    t=torch.zeros(m.parameter_count,dtype=m.dtype);t[0]=np.log(.03);t[m.i_floor]=-torch.inf;t[m.i_prestrain]=.2
    before=m.natural_metric(t).clone();mass=m.mass.numpy();q=p['nodes'];c=(mass[:,None]*q).sum(0)
    a=continue_elastic(m,t,q,np.zeros_like(q),0,[0.,.005],max_dt=.001)
    np.testing.assert_allclose((mass[:,None]*a['positions_m'][-1]).sum(0),c+[0,0,-.5*9.81*.005**2],atol=1e-10)
    assert torch.equal(before,m.natural_metric(t))


def test_contact_safety_and_zero_mass_cannot_be_bypassed():
    p,m,t=fixture()
    with pytest.raises(ValueError,match='Contact safety'):continue_elastic(m,t,p['nodes'],np.zeros((8,3)),0,[0,.01],max_dt=.001,reject_floor=False)
    t[0]=-torch.inf
    with pytest.raises(ValueError,match='mass scale'):continue_elastic(m,t,p['nodes'],np.zeros((8,3)),0,[0,.01],max_dt=.001)


def test_existing_static_model_can_continue_without_rebuilding_reference():
    from scripts.external_benchmark.fem import ExternalFEM
    p,dynamic,t=fixture();static=ExternalFEM(p,device='cpu',field=False)
    q=p['nodes'].copy();q[:,0]*=1.03;v=np.zeros_like(q)
    a=continue_elastic(dynamic,t,q,v,0,[0.,.005],max_dt=.001)
    b=continue_elastic(static,t,q,v,0,[0.,.005],max_dt=.001)
    np.testing.assert_allclose(a['positions_m'],b['positions_m'],atol=1e-13)
    np.testing.assert_allclose(a['energies'],b['energies'],atol=1e-14)


def test_varying_existing_commands_cannot_silently_freeze_first_row():
    p=falling_cube_payload();p['attachment_weights']=np.ones((1,8));p['attachment_active']=np.ones((len(p['times']),1));p['attachment_active'][-1]=0
    p['controller_targets']=np.broadcast_to(p['nodes'],(len(p['times']),1,8,3)).copy()
    m=DynamicFEM(p,device='cpu',field=False);t=torch.zeros(m.i_field+1,dtype=m.dtype)
    with pytest.raises(ValueError,match='Explicit activation schedule'):
        continue_elastic(m,t,p['nodes'],np.zeros((8,3)),0,[0.,.01],max_dt=.001)
