import numpy as np
from experiments.diagnose_fixture_gaussian_strain_location_v02 import locate,stage


def test_affine_symmetric_gradient_max_and_rigid_translation_location_guard():
    X=stage.bank.core.geometry(1)[0];A=np.array([[.012,.003,0],[.003,-.02,.001],[0,.001,.007]])
    d=locate(1,X@A.T+np.array([.1,.2,.3]))
    np.testing.assert_allclose(d['maximum_absolute_symmetric_strain'],np.max(abs(np.linalg.eigvalsh(A))),atol=2e-14,rtol=0)
    assert 0<=d['distance_to_nearest_endclamp_m']<=.18
    p=np.array(d['material_point_m']);assert np.all(p>=-1e-16) and np.all(p<=np.array([.36,.08,.08])+1e-16)
