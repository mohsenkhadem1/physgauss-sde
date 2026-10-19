from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_pacnerf_conditioned_reference import full_curvature_reference


class ReferenceCurvatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_known_correlated_gaussian_precision_and_whitening(self):
        expected=torch.tensor([[2.,.4,.2],[.4,1.5,.1],[.2,.1,.8]],dtype=torch.float64)
        def objective(z):return .5*torch.einsum('bi,ij,bj->b',z,expected,z)
        precision,info=full_curvature_reference(objective,torch.zeros(3,dtype=expected.dtype))
        torch.testing.assert_close(precision,expected,atol=1e-14,rtol=1e-14)
        root=torch.linalg.cholesky(precision)
        transform=torch.linalg.solve_triangular(root.T,torch.eye(3,dtype=expected.dtype),upper=True)
        torch.testing.assert_close(transform.T@expected@transform,torch.eye(3,dtype=expected.dtype),atol=1e-14,rtol=1e-14)
        self.assertEqual(info['adjusted_eigenvalues'],0)

    def test_negative_proposal_curvature_is_recorded_without_changing_objective(self):
        original=torch.diag(torch.tensor([-.2,1.,2.],dtype=torch.float64))
        def objective(z):return .5*torch.einsum('bi,ij,bj->b',z,original,z)
        before=objective(torch.ones(1,3,dtype=original.dtype))
        precision,info=full_curvature_reference(objective,torch.zeros(3,dtype=original.dtype))
        self.assertEqual(info['adjusted_eigenvalues'],1)
        self.assertAlmostEqual(float(precision[0,0]),.05)
        torch.testing.assert_close(objective(torch.ones(1,3,dtype=original.dtype)),before,atol=0,rtol=0)
