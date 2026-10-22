import numpy as np
from experiments.run_campaign_quality_mesh_control import prepare
from experiments.campaign_quality_tet_refinement import geometry
from experiments.campaign_sparse_nonlinear_gaussian import SparseNonlinearGaussian
from experiments.campaign_nonlinear_excitation import command


def test_matched_control_retains_physical_law_and_budget(tmp_path):
    p=prepare(tmp_path/'protocol.json')
    assert p['wall_cap_s']==300 and p['no_temporal_inverse_or_training']
    assert [r['tets'] for r in p['cases']]==[128,1024,8192]
    old=SparseNonlinearGaussian.from_design();new=SparseNonlinearGaussian(*geometry(0))
    np.testing.assert_array_equal(old.rest,new.rest)
    for name in ['node_mass','weights','load_weights','Knode','Dnode']:
        np.testing.assert_array_equal(getattr(old,name),getattr(new,name))
    f=command('balanced_axial',.1,1.)
    np.testing.assert_array_equal(old.known_force(f),new.known_force(f))
