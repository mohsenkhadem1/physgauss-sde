import numpy as np
import torch
from scripts.complex_scenes_v18.scenarios import build_case,FAMILIES
from scripts.complex_scenes_v18.torch_boundaries import TorchBoundaries
from scripts.complex_scenes_v18.explicit_boundaries import ExplicitTorchBoundaries


def test_explicit_forces_and_state_derivatives_match_energy_autodiff():
    torch.manual_seed(182)
    for family in FAMILIES:
        for variant in (0,1):
            scene,s,_=build_case(family,variant=variant);x=[m.rest+u for m,u in zip(scene.models,s.u)]
            if family in ('curved_push','container_drop'):x[0]=x[0]+np.array([.0003,.0002,-.055])
            if family=='deformable_pair':x[1]=scene.models[1].rest+np.array([.107,.007,.003])
            if family=='rigid_impact':s.rigid['striker']['center'][2]=.045
            tx=[torch.tensor(a,requires_grad=True)for a in x];rigid={k:torch.tensor(v['center'],requires_grad=True)for k,v in s.rigid.items()}
            variables=tx+list(rigid.values());old=TorchBoundaries(scene,'cpu',torch.float64);new=ExplicitTorchBoundaries(scene,'cpu',torch.float64)
            E,F,R=old.evaluate(tx,rigid,.06);e,f,r=new.evaluate(tx,rigid,.06)
            torch.testing.assert_close(E,e,rtol=1e-11,atol=1e-10)
            a=F+list(R.values());b=f+list(r.values());directions=[torch.randn_like(z)for z in a]
            for u,v in zip(a,b):torch.testing.assert_close(u,v,rtol=1e-9,atol=1e-8)
            ga=torch.autograd.grad(sum((u*d).sum()for u,d in zip(a,directions)),variables,allow_unused=True)
            gb=torch.autograd.grad(sum((u*d).sum()for u,d in zip(b,directions)),variables,allow_unused=True)
            for u,v in zip(ga,gb):
                if u is not None and v is not None:torch.testing.assert_close(u,v,rtol=1e-8,atol=1e-6)
