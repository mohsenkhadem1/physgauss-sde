import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.ridge_gaussian_inverse import Standardizer,RidgeInverse,append_texture_features


class RidgeInverseTests(unittest.TestCase):
    def test_standardization_is_train_only_and_constant_features_finite(self):
        training=np.array([[1.,5.],[3.,5.],[5.,5.]])
        normalizer=Standardizer.fit(training)
        np.testing.assert_allclose(normalizer.mean,[3.,5.])
        before=normalizer.mean.copy()
        transformed=normalizer.transform([[10000.,-20.]])
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_array_equal(normalizer.mean,before)

    def test_dual_ridge_recovers_linear_map_with_intercept(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(40,2,3))
        true=rng.normal(size=(6,33));intercept=rng.normal(size=33)
        y=x.reshape(40,-1)@true+intercept
        model=RidgeInverse.fit(x,y,1e-9,'rgb',np.full(25,-100.),np.full(25,100.))
        test=rng.normal(size=(5,2,3));expected=test.reshape(5,-1)@true+intercept
        found=model.predict_features(test)
        np.testing.assert_allclose(found['physical'],expected[:,:25],atol=1e-6)
        np.testing.assert_array_equal(found['boundary_type'],expected[:,25:29].argmax(-1))
        np.testing.assert_array_equal(found['force_type'],expected[:,29:33].argmax(-1))

    def test_checkpoint_round_trip_and_runtime_shape_guard(self):
        rng=np.random.default_rng(22);x=rng.normal(size=(10,3,4));y=rng.normal(size=(10,33))
        model=RidgeInverse.fit(x,y,.1,'rgbd',np.full(25,-5.),np.full(25,5.))
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model.npz';model.save(path,{'training_only':True})
            restored,metadata=RidgeInverse.load(path)
        self.assertEqual(metadata,{'training_only':True});self.assertEqual(restored.observation_mode,'rgbd')
        np.testing.assert_array_equal(model.predict_features(x)['physical'],restored.predict_features(x)['physical'])
        with self.assertRaises(ValueError):restored.predict_features(np.zeros((2,4)))

    def test_physical_clipping_is_declared_and_class_scores_are_not_probabilities(self):
        rng=np.random.default_rng(82);x=rng.normal(size=(12,2));y=np.tile(np.arange(33.),(12,1))
        model=RidgeInverse.fit(x,y,1.,'rgbd',np.full(25,-1.),np.full(25,1.))
        prediction=model.predict_features(x)
        self.assertTrue((prediction['physical']<1).all())
        self.assertGreater(float(prediction['physical_clipped_fraction'].mean()),.9)
        self.assertGreater(float(prediction['boundary_scores'].max()),1.)

    def test_texture_observation_schedule_and_finiteness_are_required(self):
        base=np.ones((3,2));texture={'frame_ids':np.array([0,5,10]),'features':np.full((3,4),2.)}
        combined=append_texture_features(base,np.array([0,5,10]),texture)
        np.testing.assert_array_equal(combined[:,:2],base)
        self.assertEqual(combined.shape,(3,6))
        with self.assertRaises(ValueError):append_texture_features(base,np.array([0,5,15]),texture)
        texture['features'][1,0]=np.nan
        with self.assertRaises(ValueError):append_texture_features(base,np.array([0,5,10]),texture)

    def test_frozen_texture_calibration_survives_checkpoint_loading(self):
        rng=np.random.default_rng(52);x=rng.normal(size=(10,2,5));y=rng.normal(size=(10,33))
        model=RidgeInverse.fit(x,y,1.,'rgb',np.full(25,-5.),np.full(25,5.))
        calibration={'training_episode_ids':['train0'],'color_matrix':np.eye(3).tolist()}
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model.npz';model.save(path,{'texture_calibration':calibration})
            restored,_=RidgeInverse.load(path)
        self.assertEqual(restored.texture_calibration,calibration)

    def test_prefix_future_values_never_enter_normalization_or_prediction(self):
        rng=np.random.default_rng(92);x=rng.normal(size=(30,4,2));y=rng.normal(size=(30,33))
        times=np.array([0.,.5,1.,2.]);x[:,3,:]=1000*y[:,:2]
        model=RidgeInverse.fit(x,y,.1,'rgb',np.full(25,-100.),np.full(25,100.),
                               observation_cutoff_s=1.,observation_times_s=times)
        np.testing.assert_array_equal(model.feature_standardizer.mean[-2:],0)
        before=model.predict_features(x)['physical'];changed=x.copy();changed[:,3,:]+=1e8
        np.testing.assert_array_equal(model.predict_features(changed)['physical'],before)
        changed[:,3,:]=np.nan
        np.testing.assert_array_equal(model.predict_features(changed)['physical'],before)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'prefix.npz';model.save(path);restored,metadata=RidgeInverse.load(path)
        self.assertEqual(metadata['observation_cutoff_s'],1.)
        np.testing.assert_array_equal(restored.predict_features(changed)['physical'],before)


if __name__=='__main__':unittest.main()
