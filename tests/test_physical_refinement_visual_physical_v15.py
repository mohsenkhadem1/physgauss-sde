import numpy as np
from scripts.physical_refinement_v15_reporting.visual_physical_audit import image_metrics

def test_union_rgb_and_missing_coverage_not_hidden_by_background():
 a=dict(rgb=np.zeros((10,10,3)),depth=np.full((10,10),np.inf))
 b=dict(rgb=np.zeros((10,10,3)),depth=np.full((10,10),np.inf))
 a['depth'][0,0]=.5;a['depth'][0,1]=.5;b['depth'][0,0]=.6
 a['rgb'][0,1]=1
 m=image_metrics(a,b)
 assert np.isclose(m['rgb_rmse'],.1)
 assert np.isclose(m['foreground_union_rgb_rmse'],np.sqrt(.5))
 assert m['missing_foreground_pixels']==1 and m['extra_foreground_pixels']==0
 assert m['silhouette_iou']==.5
 assert np.isclose(m['joint_foreground_depth_rmse_m'],.1)
 b['depth'][:]=np.inf
 assert image_metrics(a,b)['joint_foreground_depth_rmse_m']is None
