import numpy as np
from scripts.physical_refinement_v15_reporting.independent_results_audit import proper,point_metrics,canonical,geometry_names

def test_independent_cdf_crps_and_weight_normalization():
 v=np.array([[0.,2.],[1.,5.],[4.,8.]]);w=np.array([.2,.3,.5]);y=np.array([2.,4.])
 reference=np.mean(w@abs(v-y)-.5*np.einsum('i,j,ijd->d',w,w,abs(v[:,None]-v[None])))
 assert np.isclose(proper(v,w,y)['crps_mean'],reference)
 assert proper(np.array([[1000.]]),np.array([1.000001]),np.array([1000.]))['crps_mean']==0

def test_raw_metrics_rigid_separation_and_weighted_blocks():
 f=np.zeros((4,81,3));f[:,:,0]=.003;zero=np.zeros_like(f);w=np.ones((4,6));w[:,:3]=.1;w[:,3:]=.01
 target=(zero,canonical(zero),np.zeros((4,6)))
 metrics=point_metrics(f,w,canonical(f),target)
 assert np.isclose(metrics['coordinate_rmse_mm'],np.sqrt(3))
 assert metrics['aligned_shape_rmse_mm']<1e-10
 assert np.isclose(metrics['normalized_design_mse'],.5)
 assert len(geometry_names())==68 and len(set(geometry_names()))==68
