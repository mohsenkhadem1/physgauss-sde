"""Mechanics invariants for the new arbitrary-surface bowl adapter."""
import numpy as np
from .bowl_physics import build,Bowl,rotation_y

def test_geometry_and_contact():
    a,b=build(n=3),build(n=6)
    assert np.isclose(a.volume.sum(),b.volume.sum(),rtol=1e-13)
    assert np.isclose(a.surface_area.sum(),b.surface_area.sum(),rtol=1e-13)
    assert np.all(a.volume>0) and np.all(b.volume>0)
    p,v=a.surface_state();rng=np.random.default_rng(405)
    f=rng.normal(size=p.shape);fn=a.distribute_surface_force(f)
    assert np.allclose(fn.sum(0),f.sum(0),atol=1e-12)
    assert np.allclose(np.cross(a.x,fn).sum(0),np.cross(p,f).sum(0),atol=1e-13)
    internal,J=a.internal_forces()
    assert np.max(abs(internal))<1e-10 and np.allclose(J,1)
    a.x+=np.array([.1,-.03,.07]);internal,J=a.internal_forces()
    assert np.max(abs(internal))<1e-10 and np.allclose(J,1)
    bowl=Bowl(center=(0,0,.09),radius=.09)
    points=np.array([[0,0,-.01],[.1,0,.08],[0,0,.19]])
    gap,n=bowl.query(points)
    assert gap[0]<0 and n[0,2]>0 and gap[1]<0 and n[1,0]<0 and np.isinf(gap[2])
    bowl.rotation=rotation_y(.4)
    local=np.array([[.1,0,-.01],[0,0,.1]])
    gap,n=bowl.query(local@bowl.rotation.T+bowl.center)
    assert gap[0]<0 and np.isinf(gap[1])
    # Finite-difference normal agrees with gradient of the allowed-space SDF.
    x=np.array([[.03,.01,.025]])
    _,normal=bowl.query(x);eps=1e-7
    gradient=np.array([(bowl.query(x+eps*np.eye(3)[j])[0]-bowl.query(x-eps*np.eye(3)[j])[0])/(2*eps) for j in range(3)]).ravel()
    assert np.allclose(gradient,normal[0],atol=1e-9)

def test_risk_and_action():
    from .bowl_campaign import risk,choose,update,SIGMA
    rng=np.random.default_rng(79)
    bank=dict(target=rng.normal(size=(4,6))*.008,action=rng.normal(size=(4,2,3))*.0005)
    w=np.array([.1,.2,.3,.4]);p,index=risk(w,bank['target'])
    assert index==3 and 0<=p<=1
    assert risk(np.array([0,0,0,1]),bank['target'])[0]==0
    action,fast=choose(w,bank)
    z=np.random.default_rng(110701).normal(size=(128,3));slow=[]
    for a in range(2):
        value=0.
        for h in range(4):
            for eps in z:
                wp=update(w,bank['action'][h,a]+SIGMA*eps,bank['action'][:,a])
                value+=w[h]*risk(wp,bank['target'])[0]/len(z)
        slow.append(value)
    assert np.allclose(fast,slow,atol=1e-14)
    assert action==int(np.argmin(slow))

def test_rotating_bowl_wall_velocity():
    # Rotation about the spherical center changes tangential wall velocity, not
    # the spherical normal field. Zero friction removes its effect below rim.
    s=build(friction=0,penalty=38400);s.x[:,2]-=.025
    bowl=s.colliders[0]
    still=s.contact_forces()[0]
    bowl.rotation=rotation_y(.2);bowl.omega=np.array([0,.5,0])
    moving=s.contact_forces()[0]
    assert np.allclose(still,moving,atol=1e-12)
    bowl.friction=.4
    points,_=s.surface_state();arm=points-bowl.center;wall=np.cross(bowl.omega,arm)
    force,reactions,torques,depth,power,count=s.contact_forces()
    # Forces are scattered with the same barycentric geometry; nodal wall
    # velocity is affine, so nodal work equals quadrature work exactly.
    nodal_wall=np.cross(bowl.omega,s.x-bowl.center)
    assert np.sum((force-still)*nodal_wall)>0
    assert power<0
    assert count>0 and depth>0

if __name__=='__main__':
    test_geometry_and_contact();test_risk_and_action();test_rotating_bowl_wall_velocity();print('All bowl mechanics and risk invariants passed')
