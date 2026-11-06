"""Independent data-contract checks; no final-population outcomes are inspected."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scripts.joint_field_identification import bank,campaign
from scripts.joint_field_identification.inference import Posterior


class TestCampaignContract(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(440)
        self.data=dict(theta=bank.grid(3),points=bank.points())
        n=len(self.data['theta']);p=len(self.data['points'])
        self.data.update(unit=rng.normal(size=(n,p,3,2))*.01,
                         probe=rng.normal(size=(n,p,3,10))*.01,
                         material=rng.uniform(8000,32000,size=(n,p)))
        self.maps=bank.maps(self.data)

    def test_target_excludes_acquired_original_coordinates_and_probe_loads(self):
        hidden=[p for p in np.flatnonzero(self.data['points'][:,1]>0) if p not in (3,11)]
        observed={1,5,9,13};acquired={a['point'] for a in self.maps['actions'] if a['kind']=='view'}
        self.assertFalse(set(hidden)&(observed|acquired))
        self.assertEqual(len(hidden),8)
        self.assertEqual(self.maps['split'],192)
        self.assertEqual(self.maps['T'].shape[1],384)
        probe_ids={a['probe'] for a in self.maps['actions'] if a['kind']=='probe'}
        self.assertFalse(probe_ids&set(range(6,10)))
        assert_allclose(self.maps['T'][:,192:],0)
        assert_allclose(self.maps['offset'][:,:192],0)

    def test_force_ordering_and_physical_units(self):
        F=np.arange(16,dtype=float)*.001
        y=self.maps['H'][0]@F
        direct=[]
        for t in [0,3,7]:
            for p in [1,5,9,13]:
                for axis in [0,2]:direct.append(self.data['unit'][0,p,axis]@F[2*t:2*t+2])
        assert_allclose(y,direct)
        for action in self.maps['actions']:
            if action['kind']=='probe':
                assert_allclose(action['h'],0)
                assert_allclose(action['offset'],bank.PROBE_FORCE*self.data['probe'][:,action['point'],action['axis'],action['probe']])

    def test_generation_reproducibility_and_disjoint_development_seed(self):
        a=campaign.generate('block',self.data,8,19001)
        b=campaign.generate('block',self.data,8,19001)
        c=campaign.generate('block',self.data,8,29001)
        for key in a:assert_allclose(a[key],b[key]) if key!='kind' else self.assertEqual(str(a[key]),str(b[key]))
        self.assertFalse(np.array_equal(a['y'],c['y']))
        for i,j in enumerate(a['indices']):
            expected=self.maps['T'][j]@a['force'][i]+self.maps['offset'][j]
            assert_allclose(a['target'][i],expected)

    def test_joint_map_uses_conditional_density_volume(self):
        # The lower-weight component has much higher joint density at its force mode.
        p=Posterior(np.array([.8,.2]),np.array([[0.,0.],[1.,1.]]),np.array([np.eye(2),np.eye(2)*.01]))
        self.assertEqual(campaign.plugin(p,'marginal-map').weights.argmax(),0)
        self.assertEqual(campaign.plugin(p,'joint-map').weights.argmax(),1)

    def test_representation_ablation_prior_support_nonempty(self):
        theta=self.data['theta']
        self.assertTrue(np.isclose(theta[:,1],0).any())
        self.assertTrue((np.isclose(theta[:,2],0)&np.isclose(theta[:,4],0)&np.isclose(theta[:,3],bank.BOUNDS[3].mean())).any())


if __name__=='__main__':unittest.main()
