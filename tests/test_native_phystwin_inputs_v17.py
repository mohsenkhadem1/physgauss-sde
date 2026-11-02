import numpy as np
from scripts.compositional_dynamics_v17.native_phystwin_inputs import export_prefix,apply_known_object_masses

def test_export_no_future_tracks_and_invisible_prefix_invariance():
    initial=np.array([[0.,0,0],[.02,0,0]]);prefix=np.stack([initial,initial+.001,initial+.002]);visibility=np.ones((3,2),bool);visibility[1,1]=False
    kw=dict(initial_tracked_m=initial,observed_prefix_m=prefix,visible_prefix=visibility,controller_commands_m=np.zeros((8,1,3)),other_surface_m=np.zeros((0,3)),interior_m=np.zeros((0,3)),object_masses_kg=[.01,.02],frame_dt_s=.01)
    a,meta=export_prefix(**kw);prefix[1,1]=np.nan;b,_=export_prefix(**kw)
    for key in a:assert np.array_equal(a[key],b[key])
    assert not a['object_visibilities'][3:].any()and not a['object_motions_valid'][2:].any()
    assert np.allclose(a['object_points'][3:],initial[None])and meta['gravity_m_s2']==9.8

def test_mass_adapter_leaves_native_graph_and_kinematic_mass_exact():
    import torch
    x=torch.zeros((3,3));edges=torch.tensor([[0,1],[1,2]]);length=torch.ones(2);mass=torch.ones(3)
    old=(x,edges,length,mass,1);new=apply_known_object_masses(old,[.01,.02])
    assert new[0]is x and new[1]is edges and new[2]is length and new[4]==1
    assert torch.equal(new[3],torch.tensor([.01,.02,1.]))and torch.equal(mass,torch.ones(3))
