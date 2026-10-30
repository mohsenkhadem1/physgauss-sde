"""Development-only rounded rail diagnostic; preserves original sharp-edge study.

A fixed capsule replaces only the edge_drop rail. Capsule radius8mm, straight
segment124mm along local y, totalspan140mm; center(.01,0,.042), top50mm and
width16mm. This changes obstacle geometry and is NOT a qualification of the
sharp rail. Existing sparse material quadrature/culling handles its 1-Lipschitz
SDF; exact clipped plane contact remains in use for the floor. No production
binding is changed. Static capsules only; dynamic capsule inertia unsupported.
"""
from dataclasses import dataclass
import numpy as np
from .physics import Collider,FEMSystem,simulate
from . import engineered_scenarios as original


@dataclass
class CapsuleCollider(Collider):
    kind: str='capsule'
    segment_length: float=.124

    def __post_init__(self):
        if self.kind!='capsule':raise ValueError('CapsuleCollider requires capsule kind')
        if not np.isinf(self.mass) or self.mass<0:raise ValueError('Only static/prescribed capsules supported')
        # Reuse common field normalization/validation without changing base class.
        self.kind='sphere'
        super().__post_init__()
        self.kind='capsule'
        if not np.isfinite(self.segment_length) or self.segment_length<0:raise ValueError('Nonnegative finite segment length required')
        if not np.allclose(self.rotation.T@self.rotation,np.eye(3),rtol=0,atol=1e-10) or not np.isclose(np.linalg.det(self.rotation),1.,atol=1e-10):raise ValueError('Proper capsule rotation required')

    def query(self,x):
        local=(np.asarray(x)-self.center)@self.rotation
        offset=local.copy();offset[:,1]-=np.clip(local[:,1],-self.segment_length/2,self.segment_length/2)
        length=np.linalg.norm(offset,axis=1)
        normal=offset/np.maximum(length[:,None],1e-15)
        # SDF is nondifferentiable on the segment axis; choose local x there.
        normal[length<1e-15]=[1.,0.,0.]
        return length-self.radius,normal@self.rotation.T


def build(name,young=6000.,friction=.35,cells=(2,2,2),system_class=FEMSystem):
    system=original.build(name,young,friction,cells,system_class)
    if name=='edge_drop':
        for i,c in enumerate(system.colliders):
            if c.name=='edge':
                system.colliders[i]=CapsuleCollider(name='edge',center=(.01,0,.042),radius=.008,segment_length=.124,
                     friction=c.friction,stiffness=c.stiffness,damping_ratio=c.damping_ratio,friction_speed=c.friction_speed)
    return system


command=original.command
shared_node_indices=original.shared_node_indices


def run(name,young=6000.,friction=.35,cells=(2,2,2),dt=.0001,duration=original.DURATION,system_class=FEMSystem):
    system=build(name,young,friction,cells,system_class)
    result=simulate(system,duration,dt,original.FPS,lambda t:command(name,t))
    result.update(rest=system.rest,tets=system.tets,tool_positions=np.array([[command(name,t)[k]['center'] for k in ('left','right')] for t in result['times']]),
                  active=np.broadcast_to([1.,1.] if name=='squeeze' else ([0.,0.] if name=='edge_drop' else [1.,0.]),(len(result['times']),2)).copy(),
                  scenario=name,young=young,friction=friction,development_geometry='Rounded capsule rail replaces sharp edge; not production study')
    return result
