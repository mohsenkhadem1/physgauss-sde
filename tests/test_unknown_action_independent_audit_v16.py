import numpy as np
import pytest
from scripts.compositional_neural_mechanics_reporting.unknown_action_independent_audit import metrics,Z

def test_independent_metrics_known_values_and_coordinate_coverage():
 x=np.array([3.,4.,0.]);s=np.array([3./Z,0.,0.]);m=metrics(x,s,np.zeros(3))
 assert m['position_coordinate_rmse_m']==pytest.approx(5/np.sqrt(3))
 assert m['nominal90_coordinate_coverage']==pytest.approx(2/3)
 assert m['mean_band_halfwidth_m']==pytest.approx(1)

def test_nonfinite_and_negative_band_reject():
 for mean,sd in [(np.array([np.nan]),np.ones(1)),(np.ones(1),np.array([-1.]))]:
  with pytest.raises(ValueError):metrics(mean,sd,np.zeros(1))
