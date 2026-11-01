"""Geometric guarantee checks; no benchmark evaluation data."""
import unittest
import numpy as np
from scripts.benchmark_improvement_v14.phystwin_local_guard import local_repair,CUTOFF
from scripts.contact_benchmark_v12.feasibility import jacobians
class Checks(unittest.TestCase):
 def setUp(self):
  self.x=np.array([[0.,0,0],[1.,0,0],[0.,1,0],[0.,0,1],[2.,0,0],[3.,0,0],[2.,1,0],[2.,0,1]])
  self.e=np.array([[0,1,2,3],[4,5,6,7]]);self.d=dict(nodes=self.x,times=np.array([0.]))
 def test_valid_exact(self):
  p=self.x[None]*1.1;out,m=local_repair(self.d,self.x[None],p,self.e);np.testing.assert_array_equal(out,p)
 def test_inverted_repaired(self):
  p=self.x.copy();p[3,2]=-1
  out,m=local_repair(self.d,self.x[None],p[None],self.e)
  self.assertTrue((jacobians(self.x,out[0],self.e)>CUTOFF).all());self.assertFalse(m['failed_frames']);np.testing.assert_array_equal(out[0,4:],p[4:])
 def test_invalid_anchor_not_hidden(self):
  p=self.x.copy();p[3,2]=-1
  out,m=local_repair(self.d,p[None],self.x[None],self.e);self.assertTrue(np.isnan(out).all());self.assertEqual(m['failed_frames'],[0])
 def test_objectivity(self):
  p=self.x.copy();p[3,2]=-1
  out,_=local_repair(self.d,self.x[None],p[None],self.e)
  R=np.array([[0.,-1,0],[1.,0,0],[0.,0,1]]);b=np.array([.1,.2,.3])
  other,_=local_repair(dict(self.d,nodes=self.x@R+b),(self.x@R+b)[None],(p@R+b)[None],self.e)
  np.testing.assert_allclose(other,out@R+b,atol=1e-12)
if __name__=='__main__':unittest.main()
