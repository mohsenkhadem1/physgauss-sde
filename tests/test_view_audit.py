"""Check new multi-camera audit information against independent full AD."""
import importlib.util
from pathlib import Path
import unittest
import torch
from physgauss.benchmark import Config, Observations
from physgauss.marginal_forces import TrajectoryForceModel

path = Path(__file__).resolve().parents[1]/"scripts"/"audit_simulation_views.py"
spec = importlib.util.spec_from_file_location("simulation_view_audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ViewAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, views):
        config = Config.quick(); config.frames = 5; config.substeps = 1; config.observed_frames = (0, 4)
        model = audit.ViewAuditModel(config).set_views(views)
        gen = torch.Generator().manual_seed(8831)
        theta = .13*torch.randn(1, 8, dtype=torch.float64, generator=gen)
        force = .11*torch.randn(1, 15, dtype=torch.float64, generator=gen)
        state = model.rollout(theta, force)[0]
        observations = Observations(model.view_frames, model.predict(state[None], model.view_frames)[0].detach(), .015)
        return model, state, observations

    def test_single_camera_information_matches_existing_solver(self):
        model, state, obs = self.model([(0, 0), (4, 0)])
        original = TrajectoryForceModel(model.config)
        torch.testing.assert_close(model.predict(state[None], obs.indices), original.predict(state[None], obs.indices), rtol=0, atol=0)
        for actual, expected in zip(model.information(state, obs), original.information(state, obs)):
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-7)

    def test_multicamera_pose_blocks_match_full_image_autograd(self):
        model, state, obs = self.model([(0, 0), (4, 0), (0, 100), (4, 100)])
        image_h, record = model.pose_information(state, obs)
        jacobian = torch.autograd.functional.jacobian(lambda q: model.image_residual(q[None], obs)[0], state,
                                                     vectorize=True, strategy="forward-mode")
        torch.testing.assert_close(image_h, jacobian.T@jacobian, rtol=1e-9, atol=1e-7)
        self.assertLessEqual(record["image_only_local_rank"], 15)
        with self.assertRaises(ValueError):
            model.predict(state[None], torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
