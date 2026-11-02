import torch
from scripts.compositional_dynamics_v17.native_phystwin_graph_recovery import replace_initial_vertices

def test_fixed_graph_reset_preserves_edge_order_rest_lengths_and_masses():
    old=torch.tensor([[0.,0,.04],[.03,0,.04],[.06,0,.04]])
    edges=torch.tensor([[1,2],[0,1]],dtype=torch.int32);lengths=torch.tensor([.03,.03]);masses=torch.ones(3)*.00216
    new=old+torch.tensor([0.,0,.03]);out=replace_initial_vertices((old,edges,lengths,masses,2),new,torch.empty((0,3)))
    assert torch.equal(out[0],new) and out[1]is edges and out[2]is lengths and out[3]is masses and out[4]==2
    assert torch.equal(old[:,2],torch.full((3,),.04))
