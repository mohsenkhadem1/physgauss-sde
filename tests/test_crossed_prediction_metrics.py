"""Closed-form score checks, including reporting rather than repairing folds."""
import numpy as np

from experiments.train_campaign_crossed_prediction import metrics


def test_role_scores_have_known_translation_error_and_preserve_inversions():
    rest = np.tile(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])[None],(96,1,1))
    times = np.arange(61)/20
    truth = np.repeat(rest[:,None],61,axis=1)
    prediction = truth.copy()
    roles = np.repeat(['fit','development_force_transfer','unseen_middle_material'],32)
    for lo,offset in zip([0,32,64],[.001,.002,.003]):
        prediction[lo:lo+32,1:,:,0] += offset
    # An actual fold must be visible in the report, never repaired for scoring.
    prediction[64,1,1] = prediction[64,1,0]-np.array([1.,0.,0.])
    report = metrics(prediction,truth,rest,times,roles,np.arange(96),np.array([[0,1,2,3]]))
    assert report['role_summaries']['fit']['cases']==32
    np.testing.assert_allclose(report['role_summaries']['fit']['mean_position_rmse_m'],.001/np.sqrt(3))
    np.testing.assert_allclose(report['role_summaries']['development_force_transfer']['mean_position_rmse_m'],.002/np.sqrt(3))
    assert report['role_summaries']['unseen_middle_material']['inverted_tetrahedron_frames']==1
    assert report['cases'][64]['minimum_J']<0
    assert report['observed_motion_frames']==0
    assert not report['calibrated_uncertainty'] and not report['unseen_layout_generalization']
