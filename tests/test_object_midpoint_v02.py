import numpy as np
from scripts.gaussian_boundary_flow_v02.object_dynamics import falling_cube_payload
from scripts.gaussian_boundary_flow_v02.object_midpoint import MidpointObjectBoundaryBackend


def test_midpoint_freefall_exact_and_raw():
    d=falling_cube_payload(duration=.1);b=MidpointObjectBoundaryBackend(d,floor_mode='off',substeps=1)
    velocity=np.tile([.13,-.1,.2],(8,1))
    x,_=b.rollout(np.zeros(b.parameter_dim),np.arange(len(d['times'])),initial_velocity=velocity)
    t=d['times'];exact=d['nodes'][None]+t[:,None,None]*velocity[None]
    exact[:,:,2]-=.5*9.81*t[:,None]**2
    np.testing.assert_allclose(x,exact,atol=1e-10)
    np.testing.assert_allclose(b.last_velocity,velocity+np.array([0,0,-.981]),atol=1e-9)
    assert b.model.inertial_target is None
    assert np.max(abs(b.last_energies-b.last_energies[0]))<1e-10


def test_midpoint_sampling_invariant():
    d=falling_cube_payload(duration=.04);b=MidpointObjectBoundaryBackend(d,floor_mode='off')
    z=np.zeros(b.parameter_dim);full,_=b.rollout(z,np.arange(5));sparse,_=b.rollout(z,np.array([0,2,4]))
    np.testing.assert_array_equal(sparse,full[[0,2,4]])


def test_midpoint_smooth_spring_second_order():
    d=falling_cube_payload(duration=.12,step=.02)
    d['attachment_weights']=np.ones((1,8))/8
    d['attachment_active']=np.ones((len(d['times']),1))
    d['controller_targets']=np.broadcast_to(d['nodes'],(len(d['times']),1,8,3)).copy()
    errors=[]
    # Use mass-proportional springs for exact rigid oscillator mode.
    for sub in (1,2,4):
        b=MidpointObjectBoundaryBackend(d,floor_mode='off',substeps=sub)
        b.base_weights=b.model.mass[None].clone()
        z=np.zeros(b.parameter_dim);theta=b.bind(z)
        # Width transforms would spoil mass proportionality; fix only the test binding.
        original_bind=b.bind
        def bind(q):
            th=original_bind(q);th[0]=0.;th[1]=np.log(.1);b.model.weights=b.model.mass[None].clone();return th
        b.bind=bind
        theta=bind(z);gravity=float(theta[0].exp());spring=float(theta[1].exp())
        omega=np.sqrt(spring*9.81/(gravity*b.model.scale))
        x,_=b.rollout(z,np.arange(len(d['times'])))
        exact=d['nodes'].mean(0)[2]-9.81/omega**2*(1-np.cos(omega*d['times']))
        errors.append(np.max(abs(x.mean(1)[:,2]-exact)))
        displacement=x[-1].mean(0)[2]-d['nodes'].mean(0)[2]
        velocity=b.last_velocity.mean(0)[2]
        assert abs(.5*velocity**2+.5*omega**2*displacement**2+9.81*displacement)<1e-9
    assert 3.5<errors[0]/errors[1]<4.5
    assert 3.5<errors[1]/errors[2]<4.5
