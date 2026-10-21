import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from gaussian_chart_posterior import (SurfaceChartPosterior,ChartTransform,encode_labels,decode_chart,dequantize_chart,LOWER,UPPER)
from isaac_dataset import geometry,make_plan


class ChartPosteriorTests(unittest.TestCase):
    def test_all_declared_bc_load_labels_roundtrip(self):
        seen=set()
        for case in make_plan()['episodes']:
            points,_,_,spec=geometry(case);labels=encode_labels(case,spec,points)
            bc=labels['boundary_type'];force=labels['force_type'];seen.add((bc,force))
            decoded=decode_chart(torch.tensor(labels['chart']),torch.tensor(bc),torch.tensor(force),
                torch.tensor(labels['support_face']),torch.tensor(labels['load_faces']),
                (points.max(0)+points.min(0))/2,points.max(0)-points.min(0))
            np.testing.assert_allclose(decoded.numpy(),labels['physical'],atol=3e-7,rtol=1e-5)
        self.assertEqual(len(seen),16)

    def test_samples_have_finite_faces_and_design_force_constraints(self):
        model=SurfaceChartPosterior(context_dim=8,hidden=16,layers=2)
        draw=model.sample(torch.zeros(2,8),samples=128,generator=torch.Generator().manual_seed(4))
        p=draw['physical'];bc=draw['boundary_type'];force=draw['force_type'];sface=draw['support_face'];lface=draw['load_faces']
        center=torch.tensor([0.,0.,.25]);size=torch.tensor([.24,.12,.08])
        for start,face in [(4,sface),(10,lface[...,0]),(13,lface[...,1])]:
            coordinate=(p[...,start:start+3]-center)/size
            self.assertTrue(bool((coordinate.abs()<=.50001).all()))
            normal=coordinate.gather(-1,(face%3)[...,None]).squeeze(-1)
            torch.testing.assert_close(normal,torch.where(face<3,-.5,.5),atol=1e-6,rtol=0)
        norm=p[...,16:19].norm(dim=-1)/torch.where(bc==0,.25,1.)
        self.assertTrue(bool(((norm>=.114-1e-6)&(norm<=.252+1e-6)).all()))
        twist=force==3
        torch.testing.assert_close(p[...,19:22][twist],-p[...,16:19][twist])
        perpendicular=((p[...,13:16]-p[...,10:13])*p[...,16:19]).sum(-1)
        torch.testing.assert_close(perpendicular[twist],torch.zeros_like(perpendicular[twist]),atol=1e-8,rtol=0)
        self.assertTrue(torch.isfinite(draw['log_prob_intrinsic']).all())
        self.assertTrue(bool((p[...,22:25][bc!=3]==0).all()))

    def test_intrinsic_transform_flow_and_known_categories(self):
        model=SurfaceChartPosterior(context_dim=8,hidden=16,layers=2)
        context=torch.randn(3,8);bc=torch.tensor([0,2,3]);force=torch.tensor([0,2,3])
        support=torch.tensor([0,1,5]);loads=torch.tensor([[2,3],[4,0],[1,2]])
        draw=model.sample(context,4,bc,force,support,loads,generator=torch.Generator().manual_seed(2))
        self.assertTrue((draw['boundary_type']==bc[:,None]).all())
        chart=draw['chart'];z=model.transform(chart);torch.testing.assert_close(model.transform.inverse(z),chart)
        _,_,_,_,condition=model._conditions(context[:,None].expand(-1,4,-1),draw['boundary_type'],draw['force_type'],draw['support_face'],draw['load_faces'])
        noise,ld=model.flow(z,condition);restored,ild=model.flow.inverse(noise,condition)
        torch.testing.assert_close(restored,z,atol=1e-5,rtol=1e-5);torch.testing.assert_close(ld+ild,torch.zeros_like(ld),atol=1e-5,rtol=0)
        (-draw['log_prob_intrinsic'].mean()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_nuisance_tolerance_is_bounded_and_does_not_change_inactive_controls(self):
        chart=torch.tensor((LOWER+UPPER)/2).repeat(16,1);bc=torch.zeros(16,dtype=torch.long);force=bc.clone()
        changed=dequantize_chart(chart,bc,force,generator=torch.Generator().manual_seed(1))
        self.assertTrue(bool(((changed>torch.tensor(LOWER))&(changed<torch.tensor(UPPER))).all()))
        self.assertGreater(float(changed[:,13].std()),.5)
        self.assertLess(float(changed[:,10:12].std()),.01) # inactive force1 still has an active attached handle
        p=decode_chart(changed,bc,force,bc,torch.zeros(16,2,dtype=torch.long),(0,0,.25),(.24,.12,.08))
        self.assertTrue(bool((p[:,19:25]==0).all()))


if __name__=='__main__':unittest.main()
