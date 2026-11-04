import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.checkpoint_contract import relative_observation_noise
from scripts.gaussian_boundary_flow_v02.affine_inputs import export_affine
from scripts.gaussian_boundary_flow_v02.export_inputs import write_package,load_package,sha
from scripts.gaussian_boundary_flow_v02.object_dynamics import falling_cube_payload


@pytest.mark.parametrize('sigma',[float('nan'),float('inf'),-float('inf'),-.1])
def test_invalid_sigma_rejected_before_sampling(sigma):
    with pytest.raises(ValueError,match='sigma'):relative_observation_noise(2,3,4,sigma)


@pytest.mark.parametrize('dimension',[0,-1,1.5,True])
def test_invalid_dimension_rejected_before_sampling(dimension):
    with pytest.raises(ValueError,match='integer'):relative_observation_noise(dimension,3,4,.001)


def test_affine_export_preserves_source_and_nonmap_arrays(tmp_path):
    d=falling_cube_payload();d.update(name='test-cube',dataset='synthetic')
    source=tmp_path/'source';destination=tmp_path/'affine'
    write_package(d,source);before={p.name:sha(p)for p in source.iterdir()}
    export_affine(source,destination)
    assert before=={p.name:sha(p)for p in source.iterdir()}
    a=load_package(destination);b=load_package(source)
    for k,v in b.items():
        if isinstance(v,np.ndarray) and k!='observation_matrix':np.testing.assert_array_equal(a[k],v)
    np.testing.assert_allclose(a['observation_matrix']@a['nodes'],a['observed_initial'],atol=1e-12)
    export_affine(source,destination) # Exact idempotent export allowed.
    with pytest.raises(ValueError,match='different predictor'):export_affine(source,source)
