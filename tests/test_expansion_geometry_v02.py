import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02.expansion_geometry import increase_phystwin_resolution


def fixture():
    nodes=np.stack(np.meshgrid([0.,.01,.02],[0.,.01,.02],[0.,.01,.02],indexing='ij'),-1).reshape(-1,3)
    controls=np.array([[[0.,0.,0.],[.01,0,0],[0,.01,0]]])+np.arange(4)[:,None,None]*np.array([.002,0,0])
    return dict(dataset='PhysTwin',name='single_test',support_geometry=nodes,controls=controls,observed=nodes[:3][None],nodes=nodes[:4])


def test_expansion_preserves_inputs_and_uses_prescribed_position_only():
    data=fixture();copies={k:v.copy()for k,v in data.items()if isinstance(v,np.ndarray)}
    a=increase_phystwin_resolution(data,12);b=increase_phystwin_resolution(data,12)
    assert len(a['nodes'])==12
    for k,v in copies.items():np.testing.assert_array_equal(data[k],v)
    np.testing.assert_array_equal(a['nodes'],b['nodes'])
    np.testing.assert_allclose(a['controller_targets'][0,0],a['nodes'])
    shift=np.broadcast_to(np.array([.006,0,0]),(12,3))
    np.testing.assert_allclose(a['controller_targets'][-1,0]-a['nodes'],shift,atol=1e-15)
    np.testing.assert_array_equal(a['observed'],data['observed'])


def test_expansion_rejects_wrong_dataset():
    data=fixture();data['dataset']='SoftVTBench'
    with pytest.raises(ValueError):increase_phystwin_resolution(data,12)


@pytest.mark.parametrize('count',[0,True,3,4.5])
def test_expansion_rejects_invalid_resolution(count):
    with pytest.raises(ValueError):increase_phystwin_resolution(fixture(),count)
