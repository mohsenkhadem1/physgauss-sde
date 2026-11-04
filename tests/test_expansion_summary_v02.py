from scripts.gaussian_boundary_flow_v02.expansion_summary import aggregate


def row(a,b,complete=True):
    return {'methods':{k:{'complete':complete,'coordinate_rmse_mm':v,'aligned_shape_rmse_mm':v/2}for k,v in [('flow_pose_contact',a),('mixture_pose_contact',a),('nearest_bank_pose_contact',b)]}}


def test_missing_case_cannot_promote():
    r=aggregate([row(1,2)]*3,4)
    assert not r['candidate_gates']['flow_pose_contact']['gate_evaluable']
    assert not r['candidate_gates']['flow_pose_contact']['dataset_gate_pass']


def test_failed_case_not_dropped_for_promotion():
    r=aggregate([row(1,2)]*3+[row(1,2,False)],4)
    assert r['methods']['flow_pose_contact']['completed_cases']==3
    assert not r['candidate_gates']['flow_pose_contact']['dataset_gate_pass']


def test_equal_case_mean_and_two_win_requirement():
    r=aggregate([row(1,3),row(2,4),row(4,3),row(3,3)],4)
    assert r['methods']['flow_pose_contact']['conditional_coordinate_rmse_mm']==2.5
    assert r['candidate_gates']['flow_pose_contact']['wins']==2
    assert r['candidate_gates']['flow_pose_contact']['dataset_gate_pass']
    r=aggregate([row(0,20),row(3,2),row(3,2),row(3,2)],4)
    assert not r['candidate_gates']['flow_pose_contact']['dataset_gate_pass']
