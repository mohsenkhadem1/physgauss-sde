import inspect
import numpy as np
from experiments import isaac_block_dense_drop as d
from experiments import isaac_block_gravity_suite as old


def test_dense_mechanics_rigidmotion_and_affine_stretch():
 X,T,_,_=old.geometry_and_fixed(old.cases()[-1]);_,w,_=old.mass_and_reference(X,T)
 angle=.3;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
 q=np.stack([X,X@Q.T+np.array([.3,.2,-.01])]);r=d.dense_mechanics(X,T,q,w)
 assert r['maximum_massweighted_rigid_removed_coordinate_RMS_m']<1e-15
 assert r['maximum_abs_principal_stretch_minus1']<1e-13
 q=np.stack([X,X@np.diag([1.03,.98,1.01])]);r=d.dense_mechanics(X,T,q,w)
 assert abs(r['maximum_abs_principal_stretch_minus1']-.03)<1e-13


def test_capture_cadence_is_only_worker_body_change():
 from pathlib import Path
 original=Path('experiments/isaac_block_gravity_worker.py').read_text();new=Path('experiments/isaac_block_dense_drop_worker.py').read_text()
 assert original[original.index('def worker('):]==new[new.index('def worker('):]
 assert len(np.arange(145)/240)==145
 assert np.all((np.arange(145)/240)[::8]==np.arange(19)/30)
