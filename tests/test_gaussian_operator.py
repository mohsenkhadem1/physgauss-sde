import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from gaussian_operator import (
    PHYSICAL_DIM, PhysicalParameterTransform, GaussianKernelFields,
    NeuralTrajectoryOperator, NeuralGreenOperator, ConditionalRealNVP, PhysicalPosterior,
    physical_vector, category_labels, force_envelope, canonicalize_physical,
    tetrahedral_deformation, deformation_diagnostics,
)
from isaac_dataset import make_plan, geometry, forces_at


class GaussianOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.case = make_plan()['episodes'][15]
        cls.points, cls.tets, _, cls.spec = geometry(cls.case)
        cls.vector = torch.tensor(physical_vector(cls.case, cls.spec, cls.points))[None]
        cls.boundary, cls.force = [torch.tensor([x]) for x in category_labels(cls.case)]

    def test_physical_label_force_replay_and_bounded_transform(self):
        transform = PhysicalParameterTransform().double()
        times = np.linspace(0, 3, 97)
        for case in make_plan()['episodes'][:16]:
            points, _, _, spec = geometry(case)
            vector = torch.tensor(physical_vector(case, spec, points), dtype=torch.float64)[None]
            boundary, force = category_labels(case)
            self.assertTrue(transform.contains(vector).all())
            reconstructed = transform.unconstrained_to_physical(transform.physical_to_unconstrained(vector))
            torch.testing.assert_close(reconstructed, vector, atol=1e-12, rtol=1e-12)
            actual = (force_envelope(torch.tensor([force]), torch.tensor(times))[0, :, None, None]
                      * vector[:, 16:22].reshape(2, 3))
            np.testing.assert_allclose(actual.numpy(), np.array([forces_at(spec, t) for t in times]), atol=4e-8)
        xi = torch.randn(2, PHYSICAL_DIM, dtype=torch.float64, requires_grad=True)
        jac = torch.autograd.functional.jacobian(lambda x: transform.unconstrained_to_physical(x), xi[0])
        expected = torch.linalg.slogdet(jac)[1]
        torch.testing.assert_close(expected, transform.log_abs_det_jacobian(xi[0]))
        properties = transform.material_properties(self.vector.double())
        self.assertGreater(float(properties['youngs_modulus_Pa']), 0)
        self.assertGreater(float(properties['bulk_modulus_Pa']), 0)
        invalid = vector.clone(); invalid[0, 2] = .7
        with self.assertRaises(ValueError):
            transform.physical_to_unconstrained(invalid)

    def test_normalized_surface_load_preserves_resultant_and_couple(self):
        fields = GaussianKernelFields().double()
        points = torch.tensor(self.points, dtype=torch.float64)
        vector = self.vector.double()
        area = torch.linspace(.0001, .0004, len(points), dtype=torch.float64)
        traction = fields.traction(points, vector, self.boundary, self.force,
                                   torch.tensor([.975], dtype=torch.float64), area)
        resultant = (traction*area[None, None, :, None]).sum(2)
        expected = (vector[:, 16:22].reshape(1, 2, 3).sum(1)
                    * force_envelope(self.force, torch.tensor([.975], dtype=torch.float64)))
        torch.testing.assert_close(resultant[:, 0], expected, atol=1e-12, rtol=1e-12)
        self.assertGreater(float(traction.abs().max()), 1.)
        features = fields(points, vector, torch.tensor([0]))
        torch.testing.assert_close(features['support'][..., 1], torch.zeros_like(features['support'][..., 1]))
        with self.assertRaises(ValueError):
            fields.traction(points, vector, self.boundary, self.force, torch.tensor([1.]), -area)

    def test_flow_roundtrip_and_nontrivial_log_jacobian(self):
        torch.manual_seed(42)
        flow = ConditionalRealNVP(context_dim=5, dimension=6, layers=4, hidden=16).double()
        for layer in flow.layers:
            torch.nn.init.normal_(layer.conditioner[-1].weight, std=.04)
            torch.nn.init.normal_(layer.conditioner[-1].bias, std=.04)
        value = torch.randn(4, 6, dtype=torch.float64)
        context = torch.randn(4, 5, dtype=torch.float64)
        noise, forward_det = flow(value, context)
        recovered, inverse_det = flow.inverse(noise, context)
        torch.testing.assert_close(value, recovered, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(forward_det, -inverse_det, atol=1e-11, rtol=1e-11)
        jacobian = torch.autograd.functional.jacobian(lambda v: flow(v[None], context[:1])[0][0], value[0])
        torch.testing.assert_close(torch.linalg.slogdet(jacobian)[1], forward_det[0], atol=1e-11, rtol=1e-11)
        self.assertGreater(float((value-noise).abs().max().detach()), .01)

    def test_posterior_density_gradient_and_sampling_without_private_inputs(self):
        torch.manual_seed(7)
        posterior = PhysicalPosterior(context_dim=9, hidden=24, layers=4)
        context = torch.randn(2, 9)
        samples = posterior.sample(context, samples=5, generator=torch.Generator().manual_seed(18))
        self.assertEqual(samples['physical'].shape, (2, 5, PHYSICAL_DIM))
        self.assertTrue(torch.isfinite(samples['log_prob']).all())
        self.assertTrue(posterior.transform.contains(samples['physical']).all())
        expanded = context[:, None].expand(-1, 5, -1)
        recomputed = posterior.log_prob(samples['physical'], expanded,
                                       samples['boundary_type'], samples['force_type'])
        torch.testing.assert_close(recomputed, samples['log_prob'])
        loss = -posterior.log_prob(self.vector.expand(2, -1), context,
                                   self.boundary.expand(2), self.force.expand(2)).mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in posterior.parameters()))
        invalid = self.vector.clone(); invalid[0, 2] = 1.
        self.assertTrue(torch.isneginf(posterior.log_prob(invalid, context[:1], self.boundary, self.force)).all())
        known = posterior.sample(context, samples=3, boundary_type=torch.tensor([0, 2]), force_type=torch.tensor([3, 1]))
        torch.testing.assert_close(known['boundary_type'], torch.tensor([[0, 0, 0], [2, 2, 2]]))

    def test_direct_operator_initial_conditions_derivatives_and_physical_dependence(self):
        torch.manual_seed(2)
        operator = NeuralTrajectoryOperator(hidden=24, rank=5, depth=2,
                                            reference_points=self.points).double()
        points = torch.tensor(self.points[:7], dtype=torch.float64)
        physical = self.vector.double().clone().requires_grad_(True)
        times = torch.tensor([0., .4, 1.5, 3.], dtype=torch.float64, requires_grad=True)
        prediction = operator(points, times, physical, self.boundary, self.force)
        self.assertEqual(prediction.shape, (1, 4, 7, 3))
        torch.testing.assert_close(prediction[0, 0], points, atol=0, rtol=0)
        initial_velocity = torch.autograd.grad(prediction[:, 0].sum(), times, retain_graph=True)[0][0]
        self.assertEqual(float(initial_velocity), 0.)
        physical_gradient = torch.autograd.grad(prediction[:, -1].sum(), physical, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(physical_gradient).all())
        self.assertGreater(float(physical_gradient.abs().sum()), 1e-9)
        acceleration = torch.autograd.grad(torch.autograd.grad(prediction.sum(), times, create_graph=True)[0].sum(), times)[0]
        self.assertTrue(torch.isfinite(acceleration).all())
        jacobian, determinant = operator.spatial_jacobian(points, torch.tensor(1.5), physical, self.boundary, self.force)
        eps = 1e-6
        for axis in range(3):
            shift = torch.zeros_like(points); shift[:, axis] = eps
            plus = operator(points+shift, torch.tensor([1.5]), physical, self.boundary, self.force)[0, 0]
            minus = operator(points-shift, torch.tensor([1.5]), physical, self.boundary, self.force)[0, 0]
            torch.testing.assert_close(jacobian[0, :, :, axis], (plus-minus)/(2*eps), atol=1e-8, rtol=1e-7)
        self.assertTrue(torch.isfinite(determinant).all())
        # Querying a subset cannot change the shape context or individual answers.
        full = operator(torch.tensor(self.points, dtype=torch.float64), times.detach(), physical,
                        self.boundary, self.force)
        torch.testing.assert_close(full[:, :, :7], prediction, atol=1e-12, rtol=1e-12)

    def test_inactive_coefficients_cannot_steer_the_physical_decoder(self):
        operator = NeuralTrajectoryOperator(hidden=16, rank=4, depth=1)
        points = torch.tensor(self.points[:5], dtype=torch.float32)
        physical = self.vector.clone()
        boundary, force = torch.tensor([1]), torch.tensor([0])
        perturbed = physical.clone()
        perturbed[:, 7:10] += .02
        perturbed[:, 19:22] += .3
        perturbed[:, 22:25] += .01
        first = operator(points, torch.tensor([1., 2.]), physical, boundary, force)
        second = operator(points, torch.tensor([1., 2.]), perturbed, boundary, force)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        effective = canonicalize_physical(perturbed, boundary, torch.tensor([3]))
        torch.testing.assert_close(effective[:, 16:19]+effective[:, 19:22], torch.zeros((1, 3)))
        torch.testing.assert_close(effective[:, 22:25], torch.zeros((1, 3)))

    def test_green_operator_zero_load_superposition_and_known_controls(self):
        torch.manual_seed(73)
        operator = NeuralGreenOperator(hidden=20, rank=4, depth=2,
                                       reference_points=self.points).double()
        points = torch.tensor(self.points[:11], dtype=torch.float64)
        times = torch.tensor([0., .2, 1., 2., 3.], dtype=torch.float64)
        boundary, force = torch.tensor([1]), torch.tensor([0])
        zero = self.vector.double().clone(); zero[:, 16:25] = 0.
        at_rest = operator(points, times, zero, boundary, force)
        torch.testing.assert_close(at_rest, points[None, None].expand(1, 5, -1, -1), atol=0, rtol=0)
        a, b = zero.clone(), zero.clone()
        a[:, 16:19] = torch.tensor([.1, -.03, .01])
        b[:, 16:19] = torch.tensor([-.02, .04, .06])
        together = zero.clone(); together[:, 16:19] = a[:, 16:19]+b[:, 16:19]
        prediction_a = operator(points, times, a, boundary, force)
        prediction_b = operator(points, times, b, boundary, force)
        combined = operator(points, times, together, boundary, force)
        torch.testing.assert_close(combined, prediction_a+prediction_b-at_rest, atol=1e-14, rtol=1e-14)
        controlled = operator(points, times, a, boundary, force, controls=[{
            'center_m': a[:, 10:13], 'force_N': b[:, 16:19], 'force_type': 0}])
        torch.testing.assert_close(controlled, combined, atol=1e-14, rtol=1e-14)
        delayed = operator(points, times[:2], zero, boundary, force, controls=[{
            'center_m': a[:, 10:13], 'force_N': b[:, 16:19], 'force_type': 0, 'start_s': 1.}])
        torch.testing.assert_close(delayed, at_rest[:, :2], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            operator(points, times, zero, boundary, force, controls=[{
                'center_m': a[:, 10:13], 'force_N': b[:, 16:19], 'force_type': 0, 'seconds': 1.}])
        moving = torch.tensor([3]); motion_a, motion_b = zero.clone(), zero.clone()
        motion_a[:, 22] = .004; motion_b[:, 23] = -.003
        motion_together = zero.clone(); motion_together[:, 22:25] = motion_a[:, 22:25]+motion_b[:, 22:25]
        pa = operator(points, times, motion_a, moving, force)
        pb = operator(points, times, motion_b, moving, force)
        pab = operator(points, times, motion_together, moving, force)
        torch.testing.assert_close(pab, pa+pb-at_rest, atol=1e-14, rtol=1e-14)
        # Every exact-neural basis column reconstructs the same nonlinear-context
        # forward output, and therefore supports conditional Gaussian loading.
        offset, design = operator.load_response_basis(points, times, a, boundary, force)
        reconstruction = offset+torch.einsum('btnck,bk->btnc', design, a[:, 16:22])
        torch.testing.assert_close(reconstruction, prediction_a, atol=1e-14, rtol=1e-14)
        twist_offset, twist_design = operator.load_response_basis(points, times, a, boundary, torch.tensor([3]))
        torch.testing.assert_close(twist_design[..., 3:], torch.zeros_like(twist_design[..., 3:]), atol=0, rtol=0)

    def test_geometry_audit_detects_inversion_and_reports_affine_gradient(self):
        rest = torch.tensor([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]], dtype=torch.float64)
        tets = torch.tensor([[0, 1, 2, 3]])
        affine = torch.tensor([[1.1, .2, 0], [0., .9, .1], [0., 0, 1.2]], dtype=torch.float64)
        positions = rest@affine.T+torch.tensor([.3, -.1, .4])
        gradient, determinant = tetrahedral_deformation(positions, rest, tets)
        torch.testing.assert_close(gradient[0], affine)
        torch.testing.assert_close(determinant[0], torch.linalg.det(affine))
        inverted = rest.clone(); inverted[[1, 2]] = inverted[[2, 1]]
        audit = deformation_diagnostics(inverted, rest, tets)
        self.assertFalse(audit['positive_volume']); self.assertEqual(audit['inverted_tetrahedron_count'], 1)
        invalid = rest.clone(); invalid[0, 0] = float('nan')
        self.assertFalse(deformation_diagnostics(invalid, rest, tets)['finite'])


if __name__ == '__main__':
    unittest.main()
