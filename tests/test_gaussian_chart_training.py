import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from train_gaussian_chart_inverse import draw_observation_mask,ChartGaussianInverseModel


class ChartTrainingTests(unittest.TestCase):
    def test_prefix_mixture_never_leaks_later_observations(self):
        times=torch.tensor([0.,.25,.5,.75,1.,2.,2.25,2.5,2.75,3.]).repeat(2000,1)
        mask,prefix=draw_observation_mask(times,torch.Generator().manual_seed(7))
        self.assertTrue(mask[:,0].all());self.assertTrue(torch.equal(mask[prefix],times[prefix]<=1.))
        self.assertGreater(float(prefix.float().mean()),.46);self.assertLess(float(prefix.float().mean()),.54)
        frequency=float(mask[~prefix,1:].float().mean());self.assertGreater(frequency,.62);self.assertLess(frequency,.68)

    def test_unobserved_future_features_cannot_change_context(self):
        model=ChartGaussianInverseModel(65,hidden=16,layers=2).eval()
        features=torch.randn(2,10,65);mask=torch.zeros(2,10,dtype=torch.bool);mask[:,:5]=True
        altered=features.clone();altered[:,5:]=torch.randn_like(altered[:,5:])*1000
        with torch.no_grad():
            first=model.context({'features':features},mask);second=model.context({'features':altered},mask)
        torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
