import unittest
import numpy as np
from experiments.gaussian_silhouette_sensor import query,predict_support,measure_support,covariance


class SilhouetteSensorTests(unittest.TestCase):
    def test_projection_and_pixel_centers_match_known_rectangle(self):
        # Camera at origin, -Z forward; rectangle corners align to pixel centers.
        points=np.array([[-.5,-.3,-1.],[-.5,.3,-1.],[.5,-.3,-1.],[.5,.3,-1.]])
        k=np.array([[10.,0,10.5],[0,10.,10.5],[0,0,1.]])
        sensor=query({'rest_points_m':points},np.eye(4),k,angles=16)
        mask=np.zeros((21,21));mask[7:14,5:16]=1
        np.testing.assert_allclose(predict_support(points,sensor),measure_support(mask,sensor),atol=1e-12)
        batch=predict_support(np.stack([points,points]),sensor)
        self.assertEqual(batch.shape,(2,16))

    def test_covariance_is_positive_and_angular_errors_correlated(self):
        noise=covariance()
        self.assertGreater(np.linalg.eigvalsh(noise).min(),0)
        self.assertGreater(noise[0,1],noise[0,8])


if __name__=='__main__':unittest.main()
