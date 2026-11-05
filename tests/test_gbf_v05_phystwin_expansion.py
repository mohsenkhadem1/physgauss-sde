import json
from types import SimpleNamespace
import pytest
from scripts.gaussian_boundary_flow_v02 import phystwin_expansion as module


def test_failure_preserves_independent_baseline_and_never_scores(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(module,'launch',lambda command,*a:(calls.append(command ) or 0))
    monkeypatch.setattr(module,'load_sealed',lambda path:None)
    monkeypatch.setattr(module,'load_package',lambda path:{})
    def bad(*a):raise ValueError('Geometry invalid')
    monkeypatch.setattr(module,'GeometryAwareGaussianBoundaryBackend',bad)
    args=SimpleNamespace(run_root=str(tmp_path),plan_sha256='a'*64)
    job={'case':'double_stretch_sloth','package':'package','nodes':64,'device':'cuda:0'}
    module.execute(job,args);p=tmp_path/job['case']/'prediction-status.json';s=json.loads(p.read_text())
    assert s['terminal']and s['stages']['baseline']['state']=='complete'
    assert s['stages']['physics']['state']=='blocked'and len(calls)==1
    module.execute(job,args);assert len(calls)==1
    with pytest.raises(PermissionError):module.terminal_gate(tmp_path,args.plan_sha256)


def test_live_orphan_prevents_status_mutation(tmp_path):
    folder=tmp_path/'single_lift_dinosor';folder.mkdir();path=folder/'prediction-status.json'
    old={'case':folder.name,'plan_sha256':'a'*64,'terminal':False,'coordinator':module.process_identity(),'stages':{'physics':{'state':'running','child':module.process_identity(),'command':['running']}}}
    path.write_text(json.dumps(old));before=path.read_bytes()
    with pytest.raises(RuntimeError):module.execute({'case':folder.name},SimpleNamespace(run_root=str(tmp_path),plan_sha256='a'*64))
    assert path.read_bytes()==before


def test_shared_cohort_lock_refuses_duplicate_worker(tmp_path):
    with module.cohort_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with module.cohort_lock(tmp_path):pass


def test_rigid_failure_does_not_remove_prefix_baselines(tmp_path,monkeypatch):
    import numpy as np
    from scripts.gaussian_boundary_flow_v02 import phystwin_expansion_baseline as baseline
    from scripts.gaussian_boundary_flow_v02.export_inputs import write_package
    nodes=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
    data={'name':'single_lift_dinosor','dataset':'PhysTwin','nodes':nodes,'observed_initial':nodes.copy(),
          'observed':np.broadcast_to(nodes,(3,4,3)).copy(),'observed_mask':np.ones((3,4),bool),'observation_matrix':np.eye(4),
          'controller_targets':np.broadcast_to(nodes,(5,1,4,3)).copy(),'attachment_weights':np.ones((1,4)),'times':np.arange(5)*.1,'train_end':3}
    package=tmp_path/'inputs';write_package(data,package)
    def fail(*a,**kw):raise ValueError('Rigid pose unidentifiable')
    monkeypatch.setattr(baseline,'predict',fail)
    baseline.prepare(package,tmp_path/'baseline')
    with np.load(tmp_path/'baseline/prediction.npz',allow_pickle=False)as arrays:
        assert np.isfinite(arrays['sealed_prefix_persistence']).all()
        assert np.isfinite(arrays['sealed_prefix_velocity']).all()
        assert np.isnan(arrays['rigid_contact']).all()
    assert json.loads((tmp_path/'baseline/complete.json').read_text())['rigid_contact_failed']
