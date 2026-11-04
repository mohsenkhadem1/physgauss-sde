import numpy as np
from .metrics import manual_response_metrics,tetra_sanity

def test_rigid_motion_does_not_count_as_deformation():
    initial=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    truth=np.stack([initial,initial@R+2]);pred=np.stack([initial,initial])
    result=manual_response_metrics(pred,truth,1,2)['summary']
    assert result['tracking_mm']>1000
    assert result['pair_distance_change_mm']<1e-10

def test_true_stretch_is_detected_and_matching_stretch_is_zero():
    initial=np.array([[0,0,0],[.1,0,0],[0,.1,0]],float)
    truth=np.stack([initial,initial*1.5]);pred=np.stack([initial,initial])
    assert manual_response_metrics(pred,truth,1,2)['summary']['pair_distance_change_mm']>40
    assert manual_response_metrics(truth,truth,1,2)['summary']['pair_distance_change_mm']==0

def test_missing_prediction_is_failure_not_silent_exclusion():
    truth=np.zeros((2,3,3));pred=truth.copy();pred[1,0]=np.nan
    result=manual_response_metrics(pred,truth,1,2)['summary']
    assert result['failed_frames']==1 and not result['complete_prediction']

def test_inversion_and_unused_nodes():
    x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[2,2,2]],float)
    p=x.copy();p[3,2]=-1
    result=tetra_sanity(x,np.stack([x,p]),np.array([[0,1,2,3]]))
    assert result['unused_nodes']==1 and result['inverted_element_frame_fraction']==.5

def test_target_loader_cannot_export_future_object_motion(tmp_path,monkeypatch):
    import pickle
    from . import registration
    monkeypatch.setattr(registration,'ROOT',tmp_path)
    path=tmp_path/'runs/third_party/PhysTwin/data/different_types/test/final_data.pkl'
    path.parent.mkdir(parents=True)
    rng=np.random.default_rng(3)
    payload=dict(object_points=rng.normal(size=(8,20,3)),surface_points=rng.normal(size=(10,3)),
                 interior_points=rng.normal(size=(10,3)),controller_points=rng.normal(size=(8,4,3)))
    path.write_bytes(pickle.dumps(payload));a,c,_=registration.initial_bundle('test')
    payload['object_points'][1:]+=100
    path.write_bytes(pickle.dumps(payload));b,d,_=registration.initial_bundle('test')
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(c,d)

def test_render_metrics_identical_and_missing_object():
    from .metrics import foreground_render_metrics
    image=np.zeros((48,48,3));image[12:36,12:36]=.8;mask=image[:,:,0]>0
    identical=foreground_render_metrics(image,image,mask)
    missing=foreground_render_metrics(image,np.zeros_like(image),mask)
    assert identical['ssim']==1 and identical['psnr_db']>100
    assert missing['ssim']<.8 and missing['psnr_db']<20
