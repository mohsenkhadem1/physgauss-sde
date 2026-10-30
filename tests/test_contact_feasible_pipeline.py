"""Independent ordering and failure audit of the separate guarded runner."""
import numpy as np
import pytest
from scripts.contact_benchmark_v12 import feasible_experiment as pipeline
from scripts.contact_benchmark_v12.feasibility import inherited_elements,jacobians
from scripts.contact_benchmark_v12.scene import StaticScene


def fixture():
    nodes=np.array([[0,0,.20],[.04,0,.20],[0,.04,.20],[0,0,.24],[.04,.04,.24]],float)
    T=8;end=4;base=np.broadcast_to(nodes,(T,5,3)).copy();proposal=base.copy();proposal[1:,3,2]=.18
    active=np.ones((T,1));active[end:]=0
    data=dict(dataset='SoftVTBench',nodes=nodes,controller_targets=base[:,None],
              attachment_weights=np.ones((1,5)),attachment_active=active,floor=0.,
              observed=base[:end].copy(),observed_initial=nodes.copy(),observation_matrix=np.eye(5),
              observed_mask=np.ones((end,5),bool),times=np.arange(T)*.02,train_end=end,
              grasp=0,gripper_command=np.r_[-np.ones(end),np.ones(T-end)])
    return data,base,proposal


def test_repair_precedes_contact_and_final_tetrahedra_stay_feasible(monkeypatch):
    data,base,proposal=fixture();elements=inherited_elements(data)
    assert (jacobians(data['nodes'],proposal[3],elements)<=0).any()
    original=pipeline.rollout;seen=[]
    def audited(payload,nodes,scene,**kwargs):
        seen.append(nodes.copy())
        for frame in nodes:assert (jacobians(payload['nodes'],frame,elements)>.03+1e-8).all()
        return original(payload,nodes,scene,**kwargs)
    monkeypatch.setattr(pipeline,'rollout',audited)
    predictions,metadata=pipeline.predict(data,base,proposal,StaticScene(floor=0.))
    assert len(seen)==1
    np.testing.assert_array_equal(seen[0],predictions['feasible_object_frame'])
    for frame in predictions['feasible_contact_scene_state']:
        assert (jacobians(data['nodes'],frame,elements)>.03+1e-8).all()
    assert metadata['topology']['feasible_contact_scene_state']['frames_with_inversions']==0
    assert metadata['guard']['modified_frames']
    # An unguarded call keeps the inverted separation shape. Thus this tests
    # actual order, not merely two individually successful functions.
    unguarded,_=original(data,proposal,StaticScene(floor=0.))
    assert (jacobians(data['nodes'],unguarded[-1],elements)<=0).any()


def test_bad_prefix_anchor_has_explicit_failure_and_no_truth_repair():
    data,base,proposal=fixture();base[3,3,2]=.18
    predictions,metadata=pipeline.predict(data,base,proposal,StaticScene(floor=0.))
    assert metadata['guard']['failed_frames']==[3]
    assert metadata['contact']['inherited_failure'] is True
    assert metadata['topology']['feasible_contact_scene_state']['failed_frames']==1
    assert np.isnan(predictions['feasible_contact_scene_state'][3]).all()


def test_pipeline_rejects_evaluation_targets_and_ignores_unused_labels():
    data,base,proposal=fixture();first,meta=pipeline.predict(data,base,proposal,StaticScene(floor=0.))
    data['true_material']=np.nan;data['future_forces']=np.full((100,3),1e99)
    second,other=pipeline.predict(data,base,proposal,StaticScene(floor=0.))
    for key in first:np.testing.assert_array_equal(first[key],second[key])
    data['evaluation']={'points':np.zeros_like(base)}
    with pytest.raises(ValueError,match='Future targets'):
        pipeline.predict(data,base,proposal,StaticScene(floor=0.))


def test_guard_failure_in_future_free_segment_can_be_replaced_by_safe_contact():
    data,base,proposal=fixture();base[6]=np.nan
    predictions,metadata=pipeline.predict(data,base,proposal,StaticScene(floor=0.))
    assert metadata['guard']['failed_frames']==[6]
    assert np.isnan(predictions['feasible_object_frame'][6]).all()
    # Independent contact evolution starts at valid prefix and does not read
    # this failed future frame. The original failure remains in guard metadata.
    assert np.isfinite(predictions['feasible_contact_scene_state']).all()
    assert metadata['topology']['feasible_contact_scene_state']['failed_frames']==0
    assert metadata['topology']['feasible_contact_scene_state']['frames_with_inversions']==0
