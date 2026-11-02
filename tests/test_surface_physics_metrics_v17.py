import numpy as np
import pytest
from scripts.compositional_dynamics_v17.calibration_observations import queries
from scripts.compositional_dynamics_v17.surface_physics_metrics import diagnostics,closed_triangles

def test_closed_oriented_mesh_and_affine_volume():
    q,_=queries();tri=closed_triangles(q);edges={}
    for a,b,c in tri:
        for i,j in ((a,b),(b,c),(c,a)):edges[i,j]=edges.get((i,j),0)+1
    assert all(count==1 and edges.get((j,i))==1 for (i,j),count in edges.items())
    dims=np.array([.12,.06,.06]);rest=q*dims
    F=np.array([[.7,.2,0],[0,1.1,.1],[0,0,1.05]])
    x=np.stack([rest,rest@F.T+[10,20,30]])
    d=diagnostics([0,.01],x,q,dims,-1)
    np.testing.assert_allclose(d['volume_over_rest'],[1,np.linalg.det(F)],rtol=1e-12)

def test_contact_rebound_and_censored_endpoint():
    q,_=queries();dims=np.array([.12,.06,.06]);rest=q*dims
    z=np.array([.02,0,-.001,.01,.02,.01]);x=rest[None]+np.array([0,0,1])[None,None]*z[:,None,None]
    d=diagnostics(np.arange(6)*.01,x,q,dims,0)
    assert d['first_contact_s']==.01 and d['first_separation_s']==.03 and d['first_rebound_apex_s']==.04
    assert d['maximum_sampled_penetration_m']==pytest.approx(.001)
    assert diagnostics(np.arange(5)*.01,x[:5],q,dims,0)['first_rebound_apex_s'] is None
