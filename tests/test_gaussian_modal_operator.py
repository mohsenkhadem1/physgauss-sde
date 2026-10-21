import io
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from gaussian_modal_operator import fit_modal_basis, RBFModalBasis, NeuralModalGreenOperator, NeuralModalOperator
from gaussian_operator import physical_vector, category_labels
from isaac_dataset import make_plan, geometry


class ModalOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(2)
        cls.rest = np.array([[x, y, z] for x in np.linspace(-.12, .12, 4)
                            for y in np.linspace(-.06, .06, 3) for z in [.21, .29]])
        x = cls.rest-np.array([0., 0., .25]); shape = np.zeros((3, len(x), 3))
        shape[0, :, 0] = 1.; shape[1, :, 1] = x[:, 0]/.12; shape[2, :, 2] = (x[:, 0]/.12)**2
        rng = np.random.default_rng(31)
        cls.records = [{'split': 'train', 'episode_id': f't{i}',
            'positions_m': cls.rest+np.einsum('tr,rnc->tnc', rng.normal(0, .001, (9, 3)), shape)} for i in range(3)]
        cls.basis = fit_modal_basis(cls.rest, cls.records, rank=3)
        case = make_plan()['episodes'][1]; p, _, _, spec = geometry(case)
        cls.physical = torch.tensor(physical_vector(case, spec, p), dtype=torch.float64)[None]
        cls.bc, cls.family = [torch.tensor([c]) for c in category_labels(case)]

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_train_only_basis_captures_known_modes_and_rejects_validation(self):
        self.assertGreater(self.basis['metadata']['weighted_captured_energy_fraction'], .99999999)
        self.assertEqual([x['episode_id'] for x in self.basis['metadata']['training_records']], ['t0', 't1', 't2'])
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            fit_modal_basis(self.rest, self.records+[{'split': 'validation', 'episode_id': 'v'}])
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            fit_modal_basis(self.rest, [{'positions_m': self.records[0]['positions_m'], 'episode_id': 'unknown'}])
        basis = RBFModalBasis(self.basis).double()
        d = torch.tensor(self.records[0]['positions_m']-self.rest)
        coefficients = basis.project_displacements(d)
        reconstruction = torch.einsum('tr,rnc->tnc', coefficients, basis.modes)
        torch.testing.assert_close(reconstruction, d, atol=3e-10, rtol=1e-6)

    def test_arbitrary_rbf_queries_and_exact_known_node_modes(self):
        basis = RBFModalBasis(self.basis).double()
        points = basis.rest_points_m.clone()
        torch.testing.assert_close(basis(points), basis.modes.permute(1, 0, 2), atol=0, rtol=0)
        queries = (points[:7]+.003).requires_grad_(True)
        modes = basis(queries)
        self.assertEqual(modes.shape, (7, 3, 3))
        gradient = torch.autograd.grad(modes.square().sum(), queries)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().max()), 0.)
        batch = basis(queries[None].expand(2, -1, -1))
        torch.testing.assert_close(batch[0], modes)

    def test_modal_green_initial_rest_force_superposition_controls_and_design(self):
        torch.manual_seed(18)
        model = NeuralModalGreenOperator(self.basis, hidden=24, depth=2).double()
        points = model.basis.rest_points_m.clone(); times = torch.tensor([0., .3, 1.2, 3.], dtype=torch.float64)
        zero = self.physical.clone(); zero[:, 16:25] = 0.
        rest = model(points, times, zero, self.bc, self.family)
        torch.testing.assert_close(rest, points[None, None].expand(1, 4, -1, -1), atol=0, rtol=0)
        a, b = zero.clone(), zero.clone(); a[:, 16:19] = torch.tensor([.08, -.01, .02]); b[:, 16:19] = torch.tensor([.01, .02, -.03])
        both = zero.clone(); both[:, 16:19] = a[:, 16:19]+b[:, 16:19]
        pa = model(points, times, a, self.bc, self.family); pb = model(points, times, b, self.bc, self.family)
        combined = model(points, times, both, self.bc, self.family)
        torch.testing.assert_close(combined, pa+pb-rest, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(pa[:, 0], points[None], atol=0, rtol=0)
        control = {'center_m': a[:, 10:13], 'force_N': b[:, 16:19], 'force_type': 0}
        controlled = model(points, times, a, self.bc, self.family, controls=[control])
        torch.testing.assert_close(controlled, combined, atol=1e-14, rtol=1e-14)
        offset, design = model.load_response_basis(points, times, a, self.bc, self.family)
        torch.testing.assert_close(offset+torch.einsum('btnci,bi->btnc', design, a[:, 16:22]), pa, atol=1e-14, rtol=1e-14)
        query = torch.tensor([[.02, .01, .251], [.07, -.01, .27]], dtype=torch.float64)
        self.assertEqual(model(query, times, a, self.bc, self.family).shape, (1, 4, 2, 3))
        t = torch.tensor([0., 1.], dtype=torch.float64, requires_grad=True)
        out = model(points, t, a, self.bc, self.family)
        derivative = torch.autograd.grad(out[:, 0].sum(), t)[0]
        self.assertEqual(float(derivative[0]), 0.)

    def test_modal_checkpoint_roundtrip_preserves_basis_metadata_and_predictions(self):
        model = NeuralModalGreenOperator(self.basis, hidden=20, depth=2).double()
        data = io.BytesIO(); torch.save(model.checkpoint(), data); data.seek(0)
        restored = NeuralModalGreenOperator.from_checkpoint(torch.load(data, weights_only=True))
        p = model.basis.rest_points_m[:4]+.001; times = torch.tensor([.7, 2.3], dtype=torch.float64)
        torch.testing.assert_close(model(p, times, self.physical, self.bc, self.family),
                                   restored(p, times, self.physical, self.bc, self.family), atol=0, rtol=0)
        self.assertEqual(restored.basis.metadata, model.basis.metadata)

    def test_nonlinear_modal_reset_control_rest_and_sampling_requirement(self):
        model = NeuralModalOperator(self.basis, hidden=20, depth=2).double()
        points = model.basis.rest_points_m; times = torch.tensor([0., 1., 3.], dtype=torch.float64)
        zero = self.physical.clone(); zero[:, 16:25] = 0.
        torch.testing.assert_close(model(points, times, zero, self.bc, self.family),
                                   points[None, None].expand(1, 3, -1, -1), atol=0, rtol=0)
        controlled = model(points, times, zero, self.bc, self.family, controls=[{
            'center_m': zero[:, 10:13], 'force_N': [.02, 0., 0.], 'force_type': 0}])
        direct = zero.clone(); direct[:, 16] = .02
        torch.testing.assert_close(controlled, model(points, times, direct, self.bc, self.family))
        with self.assertRaises(TypeError): model.load_response_basis(points, times, direct, self.bc, self.family)
        with self.assertRaises(ValueError): model(points, times, direct, self.bc, self.family, controls=[{
            'center_m': zero[:, 10:13], 'force_N': [.02, 0., 0.], 'force_type': 0}])
        checkpoint = model.checkpoint(); restored = NeuralModalOperator.from_checkpoint(checkpoint)
        torch.testing.assert_close(restored(points, times, direct, self.bc, self.family), controlled)


if __name__ == '__main__': unittest.main()
