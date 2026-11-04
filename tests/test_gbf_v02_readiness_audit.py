"""Readiness instrumentation must expose failures and raw equilibrium mismatch."""
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.benchmark_readiness import prefix_metrics,draw_screen,mesh_diagnostic


def payload():
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    return dict(nodes=nodes,observed_initial=nodes.copy(),observed=np.broadcast_to(nodes,(5,4,3)).copy(),observed_mask=np.ones((5,4),bool),observation_matrix=np.eye(4))


def test_prefix_metrics_do_not_hide_initial_equilibrium_offset():
    data=payload();positions=np.broadcast_to(data['nodes']+np.array([.02,0.,0.]),(3,4,3))
    result=prefix_metrics(data,positions,np.array([0,2,4]))
    assert np.isclose(result['coordinate_rmse_mm'],20/np.sqrt(3))
    assert np.isclose(result['raw_initial_node_coordinate_mismatch_mm'],20/np.sqrt(3))
    assert np.isclose(result['raw_initial_node_max_distance_mm'],20)
    assert result['aligned_shape_rmse_mm']<1e-10


class FakeModel:
    scale=1.
    origin=np.zeros(3)
    def __init__(self,nodes,fail=None):self.nodes=nodes;self.fail=fail
    def solve(self,theta,frame,initial=None,**kwargs):
        if frame==self.fail:raise RuntimeError('Deliberate solver failure')
        state=torch.tensor(self.nodes+np.array([.001*frame,0.,0.]))
        return state,None,dict(iterations=frame+1,residual=1e-8,minJ=.9)


class FakeBackend:
    def __init__(self,fail=None):
        self.data=payload();self.model=FakeModel(self.data['nodes'],fail)
    def rollout(self,z,frames,maxiter=60):
        positions=[];diags=[];state=None
        for t in range(int(frames[-1])+1):
            state,_,d=self.model.solve(None,t,state,maxiter=maxiter)
            if t in frames:positions.append(state.numpy());diags.append(d)
        return np.stack(positions),diags


def test_canonical_diagnostics_include_unrequested_frames():
    backend=FakeBackend();r=draw_screen(backend,np.zeros(2),np.array([0,2,4]),60,5.)
    assert r['completed'] and r['numerical_acceptance_only']
    assert r['canonical_frames_completed']==5
    assert r['returned_frames']==[0,2,4]
    assert r['total_iterations']==15
    assert backend.model.solve.__func__ is FakeModel.solve


def test_failed_draw_is_retained_with_partial_metrics_and_failure_frame():
    backend=FakeBackend(fail=2);r=draw_screen(backend,np.zeros(2),np.array([0,2,4]),60,5.)
    assert not r['completed'] and not r['numerical_acceptance_only']
    assert r['failure_frame']==2 and r['canonical_frames_completed']==2
    assert r['partial_metrics_only'] and r['returned_frames']==[0]
    assert r['prefix_metrics']['frames_scored']==1
    assert r['error']['type']=='RuntimeError'
    assert backend.model.solve.__func__ is FakeModel.solve


def test_mesh_diagnostic_exposes_isolated_nodes():
    class Model:pass
    class Backend:pass
    b=Backend();b.model=Model();m=b.model
    rest=np.r_[payload()['nodes'],[[.2,.2,.2]]]
    m.rest=torch.tensor(rest);m.elements=torch.tensor([[0,1,2,3]])
    m.mass=torch.tensor([.25,.25,.25,.25,0.]);m.vol=torch.tensor([1/6]);m.scale=1.;m.mesh_filter='test';b.data={'nodes':rest}
    r=mesh_diagnostic(b)
    assert r['disconnected_node_ids']==[4] and r['zero_mass_nodes']==[4]
    assert r['connected_components']==2 and r['retained_tetrahedra']==1
