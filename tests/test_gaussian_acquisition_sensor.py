import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
from experiments.gaussian_acquisition_sensor import public_ray_queries,predict_ranges,measure_ranges,range_features,discrepancy_covariance
from experiments.isaac_acquisition_views import usd_look_at


class AcquisitionSensorTests(unittest.TestCase):
    def test_rest_prediction_and_measurement_agree_at_pixel_centers(self):
        # Closed cube: public rest geometry alone determines valid query pixels.
        points=np.array([[x,y,z] for x in [-.1,.1] for y in [-.1,.1] for z in [-.1,.1]])
        faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                        [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
        reference={'rest_points_m':points,'surface_faces':faces}
        pose=usd_look_at(np.zeros(3),45,25,1.)
        k=np.array([[600.,0,320],[0,600.,240],[0,0,1]])
        query=public_ray_queries(reference,pose,k,count=12)
        predicted=predict_ranges(points[None],reference,query)[0]
        np.testing.assert_allclose(predicted,0,atol=1e-12)
        image=np.zeros((480,640));image[query['ys'],query['xs']]=query['rest_depth_m']
        np.testing.assert_allclose(measure_ranges(image,query),predicted,atol=1e-12)
        again=public_ray_queries(reference,pose,k,count=12)
        np.testing.assert_array_equal(query['xs'],again['xs'])

    def test_missing_returns_are_censored_and_noise_correlated(self):
        np.testing.assert_allclose(range_features([0,np.inf,1.1,.9],[1,1,1,1]),[.05,.05,.05,-.05])
        covariance=discrepancy_covariance(5,.001,.002)
        self.assertTrue(np.linalg.eigvalsh(covariance).min()>0)
        self.assertGreater(covariance[0,1],0)


if __name__=='__main__':unittest.main()
