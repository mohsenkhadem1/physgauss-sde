"""Public geometry and position commands; no measured future state or force inputs."""
import numpy as np
from scripts.coupled_contact_v13.physics import Material,Collider,FEMSystem,make_block,simulate
SIZE=(.08,.06,.05)
FPS=100
DURATION=.6

def ramp(t,start,end):
    q=np.clip((t-start)/(end-start),0,1)
    return float(q*q*(3-2*q)),float(6*q*(1-q)/(end-start)) if start<t<end else 0.

def command(name,t):
    """Two public tool trajectories, including an inactive distant second tool."""
    if name=='squeeze':
        q,dq=ramp(t,.03,.15);r,dr=ramp(t,.35,.43);s,ds=ramp(t,.17,.30)
        gap=.043-.009*q+.012*r;vg=-.009*dq+.012*dr
        y=.012*s
        return {'left':{'center':np.array([-gap,y,.027]),'velocity':np.array([-vg,.012*ds,0])},'right':{'center':np.array([gap,-y,.027]),'velocity':np.array([vg,-.012*ds,0])}}
    if name in ('slide','moving_sphere','curved'):
        q,dq=ramp(t,.03,.4)
        y=.008*np.sin(np.pi*q) if name=='curved' else 0.
        vy=.008*np.pi*np.cos(np.pi*q)*dq if name=='curved' else 0.
        return {'left':{'center':np.array([-.06+.10*q,y,.028]),'velocity':np.array([.10*dq,vy,0.])},'right':{'center':np.array([.5,0,.1]),'velocity':np.zeros(3)}}
    if name=='edge_drop':
        return {'left':{'center':np.array([-.5,0,.1]),'velocity':np.zeros(3)},'right':{'center':np.array([.5,0,.1]),'velocity':np.zeros(3)}}
    raise ValueError(name)

def build(name,young=6000.,friction=.35,cells=(2,2,2),system_class=FEMSystem):
    """Same initial geometry is known to every method; material/friction are hypotheses."""
    center=(0,0,.027) if name!='edge_drop' else (.008,0,.09)
    rest,tets=make_block(SIZE,cells,center)
    common=dict(friction=friction,stiffness=4800.,damping_ratio=.5)
    cols=[Collider(kind='plane',name='floor',center=(0,0,0),**common)]
    if name=='squeeze':
        cols.extend([Collider(kind='plane',name='left',normal=(1,0,0),**common),Collider(kind='plane',name='right',normal=(-1,0,0),**common)])
    elif name in ('slide','moving_sphere','curved'):
        cols.append(Collider(kind='sphere',name='left',radius=.018,**common))
        if name=='moving_sphere':cols.append(Collider(kind='sphere',name='payload',center=(.065,0,.025),radius=.024,mass=.04,**common))
        if name=='curved':cols.append(Collider(kind='sphere',name='obstacle',center=(.085,.055,.033),radius=.035,**common))
    elif name=='edge_drop':
        cols.append(Collider(kind='box',name='edge',center=(.01,0,.025),half_extents=(.008,.07,.025),**common))
    else:raise ValueError(name)
    for c in cols:
        if c.name in command(name,0):
            for key,value in command(name,0)[c.name].items():setattr(c,key,value.copy())
    return system_class(rest,tets,Material(young=young,viscosity=1.),cols)

def run(name,young=6000.,friction=.35,cells=(2,2,2),dt=.0001,duration=DURATION,system_class=FEMSystem):
    system=build(name,young,friction,cells,system_class)
    result=simulate(system,duration,dt,FPS,lambda t:command(name,t))
    result.update(rest=system.rest,tets=system.tets,tool_positions=np.array([[command(name,t)[k]['center'] for k in ('left','right')] for t in result['times']]),
                  active=np.broadcast_to([1.,1.] if name=='squeeze' else ([0.,0.] if name=='edge_drop' else [1.,0.]),(len(result['times']),2)).copy(),
                  scenario=name,young=young,friction=friction)
    return result

def shared_node_indices(fine_rest,coarse_rest):
    distance=np.linalg.norm(fine_rest[:,None]-coarse_rest[None,:],axis=2)
    ids=distance.argmin(axis=0)
    if not np.all(distance[ids,np.arange(len(coarse_rest))]<1e-10):raise ValueError('Meshes lack exact shared reference vertices')
    return ids
