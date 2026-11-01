"""Input-only development fixtures; no evaluator or target suffix read."""
import unittest
import numpy as np
from scripts.benchmark_improvement_v14.common import ROOT
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.benchmark_improvement_v14.phystwin_gaussian_residual import predict
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.d=load_payload(ROOT/'runs/prefix-ridge-switch-v13/inputs/single_lift_sloth/full.npz')
  cls.base=np.load(ROOT/'runs/contact-benchmark-v12/base/full/single_lift_sloth/prediction.npz')['predicted'];cls.p,cls.m=predict(cls.d,cls.base)
 def test_fixed_shape_initial_and_finite(self):
  self.assertEqual(self.p.shape,self.base.shape);self.assertTrue(np.isfinite(self.p).all());np.testing.assert_array_equal(self.p[0],self.base[0]);self.assertEqual(len(self.m['centers']),12)
 def test_future_commands_do_not_change_prefix_fit(self):
  d=dict(self.d);end=d['train_end']
  for key in ('controls','controller_targets'):
   d[key]=d[key].copy();d[key][end:]+=.01
  p,m=predict(d,self.base);np.testing.assert_array_equal(p[:end],self.p[:end]);self.assertEqual(m['coefficient_norm'],self.m['coefficient_norm'])
 def test_no_extra_observations_or_evaluator(self):
  with self.assertRaises(ValueError):predict(dict(self.d,evaluation={}),self.base)
  with self.assertRaises(ValueError):predict(dict(self.d,observed=np.r_[self.d['observed'],self.d['observed'][-1:]]),self.base)
if __name__=='__main__':unittest.main()
