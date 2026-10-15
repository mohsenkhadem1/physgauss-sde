import math
import unittest
import torch

from physgauss.camera import orbit
from physgauss.physics import (InvalidDeformation, cube_mesh, edge_impulses,
                               material_response, simulate)
from physgauss.render import GaussianRenderer
from physgauss.stochastic import calibration_law, spatial_basis


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.mesh = cube_mesh()
        cls.young = torch.tensor([20.], dtype=torch.float64)

    def test_reference_volume_mass_and_zero_stress(self):
        m = self.mesh
        self.assertAlmostEqual(float(m.volumes.sum()), .65 ** 3, places=12)
        self.assertAlmostEqual(float(m.mass.sum()), .65 ** 3, places=12)
        f, e, j = material_response(m.rest[None], m, self.young)
        self.assertLess(float(f.abs().max()), 1e-12)
        self.assertLess(float(e.abs().max()), 1e-12)
        torch.testing.assert_close(j, torch.ones_like(j))

    def test_force_is_negative_energy_gradient(self):
        x = self.mesh.rest[None].clone()
        x[:, ~self.mesh.pinned, 0] += .07
        x.requires_grad_()
        f, energy, _ = material_response(x, self.mesh, self.young)
        derivative, = torch.autograd.grad(energy.sum(), x)
        torch.testing.assert_close(f, -derivative, atol=1e-10, rtol=1e-9)

    def test_objectivity_internal_force_and_torque(self):
        m = self.mesh
        x = m.rest[None].clone()
        x[:, ~m.pinned, 0] += .08
        f, energy, _ = material_response(x, m, self.young)
        angle = .73
        r = x.new_tensor([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                           [-math.sin(angle), 0, math.cos(angle)]])
        xr = x @ r.T + x.new_tensor([.8, -.3, .2])
        fr, er, _ = material_response(xr, m, self.young)
        torch.testing.assert_close(energy, er, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(fr, f @ r.T, atol=1e-11, rtol=1e-10)
        self.assertLess(float(f.sum(-2).abs().max()), 1e-12)
        self.assertLess(float(torch.linalg.cross(x, f).sum(-2).abs().max()), 1e-12)

    def test_inverted_elements_fail_explicitly(self):
        x = self.mesh.rest[None].clone()
        x[..., 0] *= -1
        with self.assertRaises(InvalidDeformation):
            material_response(x, self.mesh, self.young)

    def test_stochastic_impulses_conserve_internal_momentum_and_torque(self):
        m = self.mesh
        x = m.rest[None].clone()
        x[:, ~m.pinned, 0] += .04
        impulse = edge_impulses(x, torch.zeros_like(x), m,
                                x.new_tensor([[.1, -.2, .3]]), calibration_law,
                                spatial_basis(m))
        self.assertLess(float(impulse.sum(-2).abs().max()), 1e-12)
        self.assertLess(float(torch.linalg.cross(x, impulse).sum(-2).abs().max()), 1e-12)

    def test_time_step_refinement_and_clamp(self):
        m = self.mesh
        times = torch.linspace(0, .30, 7, dtype=torch.float64)
        args = (m, times, self.young, times.new_tensor([.6]),
                times.new_tensor([[.06, -.02, .04]]), times.new_zeros(1, 3))
        coarse, fine, reference = [simulate(*args, substeps=s).x for s in [2, 4, 16]]
        coarse_error = torch.linalg.vector_norm(coarse - reference)
        fine_error = torch.linalg.vector_norm(fine - reference)
        self.assertLess(float(fine_error), float(coarse_error))
        torch.testing.assert_close(fine[:, :, m.pinned],
                                   m.rest[m.pinned][None, None].expand(1, len(times), -1, -1))

    def test_rendering_gradients_match_finite_difference(self):
        m = self.mesh
        times = torch.linspace(0, .1, 3, dtype=torch.float64)
        cameras = orbit(times)
        renderer = GaussianRenderer(m, size=12)
        z = torch.tensor(.03, dtype=torch.float64, requires_grad=True)
        direction = (~m.pinned).to(z.dtype)[None, None, :, None] * z.new_tensor([1, 0, 0])
        x = m.rest[None, None].expand(1, 3, -1, -1) + z * direction
        value = renderer(x, *cameras).square().sum()
        analytic, = torch.autograd.grad(value, z)
        h = 1e-5
        with torch.no_grad():
            xp = m.rest[None, None].expand(1, 3, -1, -1) + (z + h) * direction
            xm = m.rest[None, None].expand(1, 3, -1, -1) + (z - h) * direction
            numeric = (renderer(xp, *cameras).square().sum()
                       - renderer(xm, *cameras).square().sum()) / (2 * h)
        torch.testing.assert_close(analytic, numeric, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
