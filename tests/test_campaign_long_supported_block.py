import numpy as np
import torch
import pytest
from experiments.campaign_long_supported_block import LongSupportedGeometry,StreamingLongSupportedP2,compile_forces,SIZE
from experiments.campaign_p2_consistent_transient import scalar_mass
from experiments.campaign_newton_stable_material import StableP2Block


def test_long_domain_patch_trace_force_and_full_mass_conservation():
 for support in ['both_ends','separated_bottom_patches']:
  m=LongSupportedGeometry((4,2,2),support=support);a=compile_forces(m,[0,2.,0]);M=scalar_mass(m)
  np.testing.assert_allclose(M@np.ones(len(m.rest)),m.density*m.body_integral_weights,atol=2e-16,rtol=2e-14)
  np.testing.assert_allclose(M.sum(),1.92,atol=1e-14)
  np.testing.assert_allclose(a['body_N'].sum(0),[0,0,-18.8352],atol=1e-12)
  np.testing.assert_allclose(a['surface_N'].sum(0),[0,2,0],atol=1e-12)
  np.testing.assert_allclose(a['metadata']['rest_surface_moment_Nm'],[-.08,0,0],atol=1e-12)
  assert m.mask_trace['passed']
  assert np.linalg.eigvalsh(M.toarray()).min()>0


def test_new_geometry_operator_parity_and_energy_force_finite_difference():
 m=LongSupportedGeometry((4,2,2),support='both_ends');op=StreamingLongSupportedP2(m,'cpu',4000,5,support_kind='both_ends');rng=np.random.default_rng(49);u=rng.normal(size=m.rest.shape)*2e-5;u[m.mask]=0;d=rng.normal(size=u.shape);d[m.mask]=0;E,g,_,_=StableP2Block.elastic(m,m.rest+u);cache=op.evaluate(torch.tensor(u,dtype=torch.float64))
 np.testing.assert_allclose(float(cache['energy']),E,rtol=1e-11,atol=1e-14);np.testing.assert_allclose(cache['gradient'].numpy(),g,rtol=1e-11,atol=1e-11)
 eps=1e-7;Ep=StableP2Block.elastic(m,m.rest+u+eps*d)[0];Em=StableP2Block.elastic(m,m.rest+u-eps*d)[0]
 np.testing.assert_allclose((Ep-Em)/(2*eps),np.sum(g*d),rtol=1e-7,atol=1e-8)


def test_matching_domain_coarse_adjoint_affine_reproduction_SPD_and_solve():
 m=LongSupportedGeometry((4,2,2),support='separated_bottom_patches');op=StreamingLongSupportedP2(m,'cpu',4000,8,support_kind=m.support_kind);rng=np.random.default_rng(51)
 np.testing.assert_allclose(op.prolongation_csr@op.coarse_rest.numpy(),m.rest,rtol=0,atol=2e-15)
 a=torch.tensor(rng.normal(size=op.coarse_free.numel()),dtype=torch.float64);r=torch.tensor(rng.normal(size=m.rest.shape),dtype=torch.float64)
 torch.testing.assert_close((op.prolong(a)*r).sum(),a@op.restrict(r),rtol=1e-13,atol=1e-13)
 c=op.evaluate(torch.zeros_like(op.rest));B=op.make_preconditioner(c,1.);Br=B(r);assert float((r*Br).sum())>0
 z=torch.tensor(rng.normal(size=m.rest.shape)*1e-4,dtype=torch.float64).masked_fill(op.mask,0.);force=op.tangent(c,z);answer,_=op.cg(c,force,0.,rtol=0.,atol=1e-11)
 torch.testing.assert_close(answer,z,rtol=1e-8,atol=1e-11)
 bad=LongSupportedGeometry((4,2,2),support='both_ends');bad.mask[0]=False
 with pytest.raises(ValueError,match='mask'):StreamingLongSupportedP2(bad,'cpu',support_kind='both_ends')


def test_operating_strain_affine_patch_virtual_work_and_finite_rotation():
 from experiments.campaign_newton_stable_material import stable_material
 m=LongSupportedGeometry((4,2,2),support='both_ends');op=StreamingLongSupportedP2(m,'cpu',support_kind='both_ends');F=np.array([[1.05,.02,0.],[0.,.98,.01],[0.,0.,.99]]);H=np.array([[.1,.2,0.],[-.03,.04,.02],[0.,0.,-.05]]);u=m.rest@(F-np.eye(3)).T;c=op.evaluate(torch.tensor(u,dtype=torch.float64));W,P,_=stable_material(F[None],m.young,m.poisson,False);volume=m.volumes.sum()
 np.testing.assert_allclose(float(c['energy']),volume*W[0],rtol=2e-12,atol=1e-13)
 np.testing.assert_allclose(float((c['gradient']*torch.tensor(m.rest@H.T)).sum()),volume*np.sum(P[0]*H),rtol=2e-12,atol=1e-11)
 angle=.7;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
 q=(m.rest@F.T)@R.T;rot=op.evaluate(torch.tensor(q-m.rest,dtype=torch.float64))
 np.testing.assert_allclose(float(rot['energy']),float(c['energy']),rtol=2e-12,atol=1e-12)
 np.testing.assert_allclose(rot['gradient'].numpy(),c['gradient'].numpy()@R.T,rtol=2e-11,atol=1e-10)
