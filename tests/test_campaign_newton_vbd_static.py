import numpy as np
import pytest
from experiments.campaign_newton_vbd_static import assemble,query,static_gates
from experiments.campaign_newton_vbd_probe_v02 import block_inputs,element_reference


def test_original_force_not_backward_euler_or_visual_stopping():
 X,T,_,_,_,force=block_inputs();r=element_reference(X,T,X);internal=assemble(r['local_force_N'],T,len(X))
 assert np.linalg.norm(internal+force)>.1
 assert not static_gates(np.linalg.norm(internal+force),0,0,0,1,0)['passed']
 assert static_gates(1e-4,1e-7,1e-5,0,.99,1e-7)['passed']
 assert not static_gates(1e-4,1e-7,1e-5,0,.99,1e-5)['passed']
 assert not static_gates(1e-4,1e-7,1e-5,0,.99,np.nan)['passed']


def test_fixed_query_reproduces_affine_material_field_and_edges():
 X,T,_,_,_,_=block_inputs();A=np.array([[1.03,.01,0],[0,.99,.02],[0,0,1.]]);q=X@A.T+[.01,-.02,.03]
 P=np.array([[.02,0,0],[.12,-.03,.02],[.24,.04,.04],[0,-.04,-.04]])
 np.testing.assert_allclose(query(X,T,q,P),P@A.T+[.01,-.02,.03],atol=1e-16)
 with pytest.raises(ValueError):query(X,T,q,[[.3,0,0]])


def test_assembled_forces_conserve_and_euclidean_querygate():
 X,T,_,_,_,_=block_inputs();q=X@np.diag([1.03,.99,.99]);r=element_reference(X,T,q);f=assemble(r['local_force_N'],T,len(X))
 np.testing.assert_allclose(f.sum(0),0,atol=1e-15)
 np.testing.assert_allclose(np.cross(q,f).sum(0),0,atol=1e-15)
 assert not static_gates(0,0,np.linalg.norm([15e-6,15e-6,15e-6]),0,1,0)['passed']
