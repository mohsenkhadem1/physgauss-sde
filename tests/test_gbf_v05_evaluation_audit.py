import json
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import phystwin_expansion_evaluate as module
from scripts.gaussian_boundary_flow_v02.phystwin_expansion import STAGES


@pytest.fixture
def setup(tmp_path,monkeypatch):
    root=tmp_path/'cohort';root.mkdir();plan={'cases':[]};digest='a'*64
    for case in module.CASES:
        folder=root/case;folder.mkdir();job={'case':case,'input_digest':case,'manifest_sha256':'manifest'};plan['cases'].append(job)
        status={'case':case,'plan_sha256':digest,'terminal':True,'stages':{name:{'state':'complete'if name=='baseline'else'failed'}for name in STAGES}}
        (folder/'prediction-status.json').write_text(json.dumps(status))
    monkeypatch.setattr(module,'check_plan',lambda *a:plan)
    def load(path):
        case=path.parent.name
        p={'case':case,'input_digest':case,'input_package_manifest_sha256':'manifest','prefix_frames':[0,2],'point_ids':[0,1,2],'args':{'nodes':4}}
        return p,{'train_end':3},{'frames':5},{'nodes':np.zeros((4,3)),'observed_initial':np.zeros((3,3)),'observation_matrix':np.zeros((3,4))}
    monkeypatch.setattr(module,'load_sealed',load)
    calls=[];monkeypatch.setattr(module,'launch',lambda *a:(calls.append(a[0])or 0));monkeypatch.setattr(module,'verify_score',lambda *a:{})
    return root,digest,plan,calls


def test_missing_fourth_case_blocks_all_evaluation(setup):
    root,digest,plan,calls=setup
    (root/module.CASES[-1]/'prediction-status.json').unlink()
    with pytest.raises(PermissionError):module.evaluate(root,'unused',digest)
    assert calls==[]and not(root/'evaluation-gate.json').exists()


def test_live_later_orphan_blocks_all_writes(setup):
    root,digest,plan,calls=setup
    path=root/module.CASES[-1]/'baseline-evaluation-status.json'
    path.write_text(json.dumps({'state':'running','coordinator':module.process_identity(),'child':module.process_identity(),'command':['live']}))
    before=path.read_bytes()
    with pytest.raises(RuntimeError):module.evaluate(root,'unused',digest)
    assert path.read_bytes()==before and not calls and not(root/'evaluation-gate.json').exists()


def test_independent_baseline_survives_all_physics_failures_without_retries(setup):
    root,digest,plan,calls=setup
    module.evaluate(root,'unused',digest)
    assert len(calls)==4
    for case in module.CASES:
        assert json.loads((root/case/'baseline-evaluation-status.json').read_text())['state']=='complete'
        assert json.loads((root/case/'physics-evaluation-status.json').read_text())['state']=='blocked'
    module.evaluate(root,'unused',digest);assert len(calls)==4


def test_changed_input_identity_rejected_before_any_target_process(setup):
    root,digest,plan,calls=setup;plan['cases'][-1]['input_digest']='wrong'
    with pytest.raises(ValueError,match='frozen case inputs'):module.evaluate(root,'unused',digest)
    assert not calls and not(root/'evaluation-gate.json').exists()


def test_summary_keeps_all_four_failure_denominators(setup,monkeypatch,tmp_path):
    root,digest,plan,calls=setup;module.evaluate(root,'unused',digest)
    metric={'coordinate_rmse_mm':1.,'aligned_shape_rmse_mm':.5,'total_frames':2,'valid_frames':2,'failed_frames':0,'no_reference_frames':0}
    document={'methods':{name:{'incoming_v22':{'summary':metric}}for name in ('prefix_persistence','prefix_velocity','rigid_contact')}}
    monkeypatch.setattr(module,'verify_score',lambda *a:document)
    for case in module.CASES:(root/case/'baseline-scores.json').write_text(json.dumps(document))
    result=module.summarize(root,'unused',digest,tmp_path/'summary.json')
    assert result['aggregates']['persistence']['complete_cases']==4
    assert result['aggregates']['flow_contact']['failed_or_missing_cases']==4
    pairs=result['paired_differences']['flow_contact']['rigid_contact']
    assert pairs['expected_pairs']==4 and pairs['complete_pairs']==0 and len(pairs['differences'])==4
    assert not module.qualified(dict(metric,total_frames=0,valid_frames=0))
    assert not module.qualified(dict(metric,coordinate_rmse_mm=float('nan')))
