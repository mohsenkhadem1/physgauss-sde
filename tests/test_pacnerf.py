import json
from pathlib import Path
import tempfile
import unittest
import torch
from physgauss.pacnerf_data import read_particles
from physgauss.pacnerf_model import PACModel, graph_modes
from physgauss.calibrated_diffusion import CalibratedDiffusion
from physgauss.neural_marginal import BlockLoadElimination


def fixture(noise='neural', force='marginal'):
    rng = torch.Generator().manual_seed(317)
    xyz = torch.rand(32,3,dtype=torch.float64,generator=rng)*torch.tensor([.6,.6,.4])+torch.tensor([-.3,.3,-.2])
    data = {'rest': xyz, 'gaussians': xyz.repeat(4, 1), 'weights': torch.eye(32, dtype=torch.float64).repeat(4, 1),
        'size': 8, 'times': torch.arange(14, dtype=torch.float64)/24}
    k = torch.tensor([[10.,0,4],[0,10.,4],[0,0,1]], dtype=torch.float64)
    eye = torch.tensor([0., .6, 3.], dtype=torch.float64)
    rot = torch.diag(torch.tensor([1., -1., -1.], dtype=torch.float64))
    data['cameras'] = {c:(eye, rot, k) for c in range(11)}
    data['backgrounds'] = {c:torch.ones(8,8,3,dtype=torch.float64)*.8 for c in range(11)}
    data['images'] = {(c,f):data['backgrounds'][c] for c in range(11) for f in range(14)}
    with torch.random.fork_rng():
        torch.manual_seed(19); network = CalibratedDiffusion()
    calibration = {'state_dict':{k:v.tolist() for k,v in network.state_dict().items()}, 'constant_amplitude':.0018}
    return PACModel(data, calibration, noise, force)


class PACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_initial_geometry_graph_has_six_rigid_modes(self):
        model = fixture()
        b = model.basis.flatten(0,1)
        torch.testing.assert_close(b.T@b, torch.eye(12,dtype=b.dtype), atol=1e-12,rtol=1e-12)
        self.assertEqual(int((model.eigen == 0).sum()), 6)
        net = torch.einsum('ne,btec->btnc', model.incidence,
            torch.randn(1,2,len(model.edges),3,dtype=b.dtype))
        torch.testing.assert_close(net.sum(2), torch.zeros(1,2,3,dtype=b.dtype), atol=1e-12,rtol=0)

    def test_joint_profile_marginal_identity_and_conditional(self):
        model = fixture(); z = torch.zeros(1, model.dimension,dtype=torch.float64)
        r, cov, g, _ = model.terms(z)
        e = BlockLoadElimination(g[0], cov); mean = e.conditional_mean(r.flatten(1))
        actual = e.marginal_nll(r.flatten(1))-e.joint_nll(r.flatten(1), mean)
        expected = .5*(e.logdet-e.logdet_q)-.5*model.force_dimension*torch.log(torch.tensor(2*torch.pi))
        torch.testing.assert_close(actual, expected, atol=1e-6,rtol=1e-8)
        joint = fixture(force='joint')
        torch.testing.assert_close(joint.objective(torch.cat((z,mean),-1)),
            model.objective(z)-actual,atol=1e-7,rtol=1e-8)
        self.assertGreater(float(torch.linalg.eigvalsh(cov).min()), 0)

    def test_noise_is_causal_and_force_independent(self):
        model = fixture(force='joint'); z = torch.zeros(1,model.dimension,dtype=torch.float64)
        _, c0, _, _ = model.terms(z)
        z[:,model.base_dimension:] = 5
        _, c1, _, _ = model.terms(z)
        torch.testing.assert_close(c0,c1,atol=0,rtol=0)
        # Last next-pose coordinate cannot change ANY preceding diffusion.
        z[:,2+model.modes+12*2*model.modes] = .2
        _, c2, _, _ = model.terms(z)
        torch.testing.assert_close(c1,c2,atol=0,rtol=0)

    def test_full_target_directional_gradient(self):
        model = fixture(); z = torch.zeros(model.dimension,dtype=torch.float64,requires_grad=True)
        value = model.objective(z[None]).sum(); g,=torch.autograd.grad(value,z)
        rng=torch.Generator().manual_seed(7); d=torch.randn(z.shape,dtype=z.dtype,generator=rng);d=d/d.norm()
        eps=1e-5
        fd=(model.objective((z+eps*d)[None])-model.objective((z-eps*d)[None]))/(2*eps)
        torch.testing.assert_close(fd[0],g@d,atol=2e-5,rtol=2e-5)

    def test_exact_linear_propagator_semigroup(self):
        model=fixture();z=torch.zeros(1,model.dimension,dtype=torch.float64)
        a=model.transition(z,1/48);b=model.transition(z,1/24)
        torch.testing.assert_close(a@a,b,atol=1e-12,rtol=1e-12)

    def test_ply_parser_rejects_truncated_data(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            p=Path(folder)/'test.ply'
            p.write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty double x\nproperty double y\nproperty double z\nend_header\n'+bytes(24))
            with self.assertRaises(ValueError): read_particles(p)


if __name__ == '__main__': unittest.main()
