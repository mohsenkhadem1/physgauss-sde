import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from boundary_block import block_mesh, lumped_masses, episode, forces_at, reference_gaussians


class BoundaryBlockTests(unittest.TestCase):
    def test_tetrahedra_fill_block_with_positive_volumes_and_closed_surface(self):
        p,t,f=block_mesh(cells=(4,3,2))
        mass=lumped_masses(p,t,500.)
        self.assertAlmostEqual(mass.sum(),.24*.12*.08*500.,places=12)
        self.assertTrue(np.all(mass>0))
        edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
        _,counts=np.unique(edges,axis=0,return_counts=True)
        self.assertTrue(np.all(counts==2))

    def test_seeded_boundaries_disjoint_and_force_impulse_has_newton_seconds_units(self):
        p,t,f=block_mesh()
        a=episode(p,f);self.assertEqual(a,episode(p,f))
        self.assertNotEqual(a['fixture_nodes'],episode(p,f,seed=31002)['fixture_nodes'])
        self.assertFalse(set(a['fixture_nodes']) & set(a['load_nodes']))
        time=np.linspace(0,a['seconds'],10001)
        forces=np.asarray([forces_at(a,x) for x in time])
        impulse=np.trapezoid(forces,time,axis=0)
        expected=np.asarray([np.asarray(x['direction_world'])*x['peak_newtons']*x['duration_s']/2 for x in a['pulses']])
        np.testing.assert_allclose(impulse,expected,atol=2e-6,rtol=2e-6)
        np.testing.assert_array_equal(forces_at(episode(p,f,force_scale=0.),1.),np.zeros((3,3)))

    def test_reference_gaussians_remain_attached_to_material_faces(self):
        p,t,f=block_mesh(cells=(3,2,2));g=reference_gaussians(p,f,count=256)
        np.testing.assert_allclose(g['means_m'],np.einsum('ni,nij->nj',g['barycentric_weights'],p[g['face_vertex_ids']]),atol=3e-8)
        np.testing.assert_allclose(g['barycentric_weights'].sum(-1),1,atol=1e-7)

    def test_finite_load_patches_are_compact_disjoint_and_clear_of_fixture(self):
        p,t,f=block_mesh();s=episode(p,f,fixture_nodes=20,load_patch_nodes=9)
        used=set(s['fixture_nodes'])
        for center,patch in zip(s['load_nodes'],s['load_patches']):
            self.assertIn(center,patch);self.assertEqual(len(patch),9)
            self.assertFalse(used & set(patch));used.update(patch)
            self.assertLessEqual(float(np.linalg.norm(p[patch]-p[center],axis=-1).max()),.065)


if __name__=='__main__':unittest.main()
