import numpy as np
from scripts.gaussian_boundary_flow_v02 import original_phystwin_controls as wrapper
from scripts.gaussian_boundary_flow_v02 import phystwin_expansion_baseline as original
from scripts.gaussian_boundary_flow_v02.export_inputs import write_package
from scripts.external_benchmark.data import fps


def test_numeric_recipe_matches_frozen_expansion_baseline(tmp_path):
    nodes=np.array([[0.,0,0],[.04,0,0],[0,.04,0],[0,0,.04]])
    data={'name':'single_lift_dinosor','dataset':'PhysTwin','nodes':nodes,'observed_initial':nodes.copy(),
          'observed':np.stack([nodes+[.001*t,0,0]for t in range(4)]),'observed_mask':np.ones((4,4),bool),'observation_matrix':np.eye(4),
          'controller_targets':np.stack([nodes+[.001*t,0,0]for t in range(6)])[:,None],
          'attachment_weights':np.ones((1,4)),'attachment_active':np.ones((6,1)),'times':np.arange(6)*.1,'train_end':4,'floor':0.}
    package=tmp_path/'inputs';write_package(data,package);original.prepare(package,tmp_path/'reference')
    frames=np.unique(np.linspace(0,3,12,dtype=int));ids=fps(nodes,48)
    initial,rigid,controls,_=wrapper.numeric_controls(data,frames,ids)
    with np.load(tmp_path/'reference/prediction.npz',allow_pickle=False)as arrays:
        assert np.array_equal(arrays['initial_geometry'],initial)
        assert np.array_equal(arrays['rigid_contact'],rigid)
        for key,value in controls.items():assert np.array_equal(arrays['sealed_'+key],value)
