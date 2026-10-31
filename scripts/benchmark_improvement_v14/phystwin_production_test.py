"""Development cached forecast fixtures; no evaluator/manual/suffix target reads."""
import unittest
import numpy as np
from scripts.benchmark_improvement_v14.common import ROOT,read
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.benchmark_improvement_v14.phystwin_production import select,strict
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  name='single_lift_sloth';d=ROOT/'runs/prefix-ridge-switch-v13/inputs'/name;cls.full=load_payload(d/'full.npz');cls.short=load_payload(d/'short.npz');p=ROOT/'runs/benchmark-improvement-v14/phystwin-attachment-development'
  cls.fa=np.load(p/'full-prediction.npz')['predicted'];cls.sa=np.load(p/'short-prediction.npz')['predicted'];cls.expected=np.load(p/'selected-prediction.npz')['predicted'];o=ROOT/'runs/contact-benchmark-v12/evaluation/development'/name
  cls.old=np.load(o/'predictions.npz')['object_frame'];cls.os=np.load(ROOT/'runs/contact-benchmark-v12/base/validation'/name/'prediction.npz')['predicted'];cls.config=read(o/'prediction-metadata.json')['object_frame']
 def call(self,fa=None,sa=None,os=None):return select(self.full,self.short,self.fa if fa is None else fa,self.sa if sa is None else sa,self.old,self.os if os is None else os,self.config)
 def test_independent_selection(self):
  x,m=self.call();self.assertTrue(m['switched']);self.assertAlmostEqual(m['candidate']['validation_rmse_m']*1000,12.30152631092337,places=8);self.assertAlmostEqual(m['original_validation_rmse_m']*1000,13.938729085992833,places=8);np.testing.assert_array_equal(x,self.expected)
 def test_failed_short_exact_fallback(self):
  x,m=self.call(sa=np.full_like(self.sa,np.nan));self.assertFalse(m['switched']);np.testing.assert_array_equal(x,self.old)
 def test_selected_full_failure_original_fallback(self):
  x,m=self.call(fa=np.full_like(self.fa,np.nan));self.assertFalse(m['switched']);self.assertTrue(m['validation_passed']);np.testing.assert_array_equal(x,self.old)
 def test_no_future_failure_selection(self):
  p=self.os.copy();p[self.full['train_end']:]=np.nan;x,m=self.call(os=p);self.assertTrue(m['switched']);np.testing.assert_array_equal(x,self.expected)
 def test_strict_prefix(self):
  with self.assertRaises(ValueError):strict(dict(self.full,observed=np.r_[self.full['observed'],self.full['observed'][-1:]]))
  with self.assertRaises(ValueError):strict(dict(self.full,evaluation={}))
  with self.assertRaises(ValueError):strict(dict(self.full,name='not_declared'))
 def test_unavailable_original_retained(self):
  config=dict(method='none',selection_unavailable='Inherited failure',inherited_failure=True)
  x,m=select(self.full,self.short,self.fa,self.sa,self.old,self.os,config)
  self.assertFalse(m['switched']);np.testing.assert_array_equal(x,self.old)
 def test_changed_original_score_rejected(self):
  import copy
  config=copy.deepcopy(self.config)
  for r in config['validation']:
   if (r['method'],r['strength'])==(config['method'],config['strength']):r['validation_rmse_m']*=.9
  with self.assertRaises(ValueError):select(self.full,self.short,self.fa,self.sa,self.old,self.os,config)
if __name__=='__main__':unittest.main()
