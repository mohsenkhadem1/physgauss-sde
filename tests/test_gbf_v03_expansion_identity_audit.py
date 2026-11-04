import copy
import numpy as np
import pytest
from scripts.gaussian_boundary_flow_v02 import expansion_summary as module


def test_empty_or_nonfinite_metrics_cannot_complete():
    good={'total_frames':2,'valid_frames':2,'failed_frames':0,'no_reference_frames':0,'coordinate_rmse_mm':1.,'aligned_shape_rmse_mm':.5}
    assert module.complete_metric(good)
    assert not module.complete_metric(dict(good,total_frames=0,valid_frames=0))
    for value in [None,float('nan'),float('inf')]:
        assert not module.complete_metric(dict(good,coordinate_rmse_mm=value))
    assert not module.complete_metric(dict(good,failed_frames=1))


def test_narrow_gate_pass_does_not_hide_velocity_deficit():
    def metric(x):return {'complete':True,'coordinate_rmse_mm':x,'aligned_shape_rmse_mm':x}
    rows=[{'methods':{'flow_pose_contact':metric(2),'mixture_pose_contact':metric(2),'nearest_bank_pose_contact':metric(3),
                     'prefix_velocity':metric(1),'prefix_persistence':metric(4)}}for _ in range(4)]
    result=module.aggregate(rows,4)
    assert result['candidate_gates']['flow_pose_contact']['dataset_gate_pass']
    velocity=result['strong_baseline_comparisons']['flow_pose_contact']['prefix_velocity']
    assert velocity['mean_coordinate_deficit_mm']==1 and velocity['coordinate_worse_cases']==4
    rows[0]['methods']['flow_pose_contact']['coordinate_rmse_mm']=float('nan')
    assert not module.aggregate(rows,4)['candidate_gates']['flow_pose_contact']['gate_evaluable']


@pytest.mark.parametrize('change',['none','hash','interval','nodes','budget','digest','matrix','ids'])
def test_cross_channel_identity_guard(tmp_path,monkeypatch,change):
    runs={k:tmp_path/k for k in ['learned-raw','retrieval-raw']}
    documents={};saved={}
    for category,directory in runs.items():
        directory.mkdir();(directory/'prediction.npz').write_bytes(category.encode());(directory/'plan.json').write_text('{}')
        plan={'case':'case','input_digest':'same','prefix_frames':[0,2],'point_ids':[0,1,2]}
        arrays={'nodes':np.zeros((4,3)),'observation_matrix':np.ones((3,4))/4,'observed_initial':np.zeros((3,3))}
        saved[directory]=(plan,{'train_end':3},{'frames':5},arrays)
        documents[category]={'case':'case','interval':[3,5],'nodes':4,'baseline_budget':{'frames':[0,2],'point_ids':[0,1,2]},
                             'source_sha256':{'prediction':module.sha(directory/'prediction.npz')},'observed_node_ids':[7,3,1]}
    document=documents['retrieval-raw'];plan,_,_,arrays=saved[runs['retrieval-raw']]
    if change=='hash':document['source_sha256']['prediction']='bad'
    elif change=='interval':document['interval']=[2,5]
    elif change=='nodes':document['nodes']=5
    elif change=='budget':document['baseline_budget']['frames']=[1,2]
    elif change=='digest':plan['input_digest']='different'
    elif change=='matrix':arrays['observation_matrix'][0,0]=2
    elif change=='ids':document['observed_node_ids']=[1,3,7]
    monkeypatch.setattr(module,'load_sealed',lambda path:saved[path])
    if change=='none':assert len(module.validate_channel_identity('case',documents,runs))==2
    else:
        with pytest.raises(ValueError):module.validate_channel_identity('case',documents,runs)
