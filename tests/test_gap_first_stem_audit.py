import unittest
import numpy as np
from numpy.testing import assert_allclose
from scripts.gap_first_v01.stem_audit import green,qualify_risk
from scripts.gap_first_v01.stem_mechanics import beam_fem,response_row
from scripts.gap_first_v01.stem_inference import Posterior,expected_risk

class StemAuditTests(unittest.TestCase):
 def test_reciprocity_and_support(self):
  L,EI,k=.12,.0014,5.;x=np.linspace(0,L,17);a=.07
  assert_allclose(green(x,a,L,EI,k),green(a,x,L,EI,k),atol=1e-15)
  self.assertEqual(float(green(0,a,L,EI,k)),1/k)
 def test_correct_two_half_experiments(self):
  F,L=.02,.12
  for theta,look,push in [([.4,.1],.45,.3),([.1,.4],.8625,.825)]:
   tip=response_row(L,L,L,F)@theta
   self.assertAlmostEqual(response_row(L/2,L,L,F)@theta/tip,look)
   self.assertAlmostEqual(response_row(L/2,L/2,L,F)@theta/tip,push)
 def test_rank(self):
  F,L=.02,.12;tip=response_row(L,L,L,F);base=response_row(0,L,L,F)
  self.assertEqual(np.linalg.matrix_rank(np.stack([tip,tip])),1)
  self.assertEqual(np.linalg.matrix_rank(np.stack([tip,base])),2)
 def test_fem_candidate_fraction(self):
  L,F,b,s=.1,.02,.48,.04;EI=L**3/(3*b)
  x,u,_=beam_fem(L,EI,1/s,F,.35,40)
  assert_allclose(u,F*green(x,.35*L,L,EI,1/s),atol=1e-9)
 def test_expected_failure_identity(self):
  self.assertTrue(qualify_risk()['passed'])
 def test_expected_risk_does_not_mutate_posterior(self):
  p=Posterior(np.array([[.1,.4],[.4,.1]]),np.array([.4,.6]),np.log([.4,.6]));before=p.weights.copy()
  expected_risk(p,np.array([.1,.2]),dict(row=np.array([.2,.3]),sigma=.01,cost=.03),.01,161)
  assert_allclose(p.weights,before)

if __name__=='__main__':unittest.main()
