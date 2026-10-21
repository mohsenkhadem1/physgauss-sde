import unittest
import numpy as np
from experiments.campaign_numerical_force_polynomial import (
    force_features,fit_response_map,predict_response,finite_strain,require_train_ids)


def forces():
    axes=.02*np.eye(3)
    directions=np.array([[1,1,0],[1,0,1],[0,1,1],[-1,1,0]])/np.sqrt(2)
    return np.concatenate((axes,np.array([m*d for d in directions for m in [.01,.03]])))


class ForcePolynomialTests(unittest.TestCase):
    def test_quadratic_identifies_cross_terms_on_unseen_diagonal(self):
        f=forces();rng=np.random.default_rng(19);truth=rng.normal(size=(9,2,5,3))
        y=np.tensordot(force_features(f,2),truth,axes=([-1],[0]))
        coef,info=fit_response_map(f,y,2);query=np.array([.01,.01,.005])
        np.testing.assert_allclose(predict_response(coef,query,2),predict_response(truth,query,2),atol=1e-12)
        self.assertEqual(info['rank'],9)
        linear,_=fit_response_map(f,y,1)
        self.assertGreater(np.linalg.norm(predict_response(linear,query,1)-predict_response(truth,query,2)),.1)

    def test_axes_cannot_identify_quadratic_cross_terms(self):
        with self.assertRaises(ValueError):fit_response_map(.02*np.eye(3),np.ones((3,4)),2)

    def test_measured_moving_support_offset_does_not_bias_force_fit(self):
        f=forces();rng=np.random.default_rng(11);offset=rng.normal(size=(4,3));coef=rng.normal(size=(3,4,3))
        observed=offset+np.tensordot(force_features(f,1),coef,axes=([-1],[0]))
        fitted,_=fit_response_map(f,observed-offset,1)
        np.testing.assert_allclose(fitted,coef,atol=1e-12)
        np.testing.assert_array_equal(predict_response(fitted,np.zeros(3),1),np.zeros_like(offset))

    def test_finite_strain_removes_rigid_rotation(self):
        rest=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
        rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1.]])
        result=finite_strain(rest,np.array([[0,1,2,3]]),(rest@rotation.T+2)[None])
        self.assertAlmostEqual(result['green_strain_frobenius_max'],0)
        self.assertAlmostEqual(result['minimum_volume_ratio'],1)

    def test_reserved_and_calibration_ids_rejected_before_loading(self):
        for ids in [['v0008'],['h0000'],['t0112'],['t0000','t0000']]:
            with self.assertRaises(ValueError):require_train_ids(ids,['t0000','t0001','t0002','t0003'])


if __name__=='__main__':unittest.main()
