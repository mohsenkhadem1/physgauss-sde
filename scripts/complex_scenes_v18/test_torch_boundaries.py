import numpy as np
import torch
from scripts.complex_scenes_v18.scenarios import build_case,FAMILIES
from scripts.complex_scenes_v18.torch_boundaries import TorchBoundaries


def test_all_family_external_forces_match_reference_and_have_state_derivatives():
    for family in FAMILIES:
        scene,s,_=build_case(family,variant=0);time=.06
        x=[m.rest+u for m,u in zip(scene.models,s.u)]
        if family in ('curved_push','container_drop'):x[0]=x[0]+np.array([0,0,-.055])
        if family=='deformable_pair':x[1]=scene.models[1].rest+np.array([.107,.007,.003])
        if family=='rigid_impact':s.rigid['striker']['center'][2]=.045
        op=TorchBoundaries(scene,device='cpu',dtype=torch.float64)
        tx=[torch.tensor(a,requires_grad=True)for a in x];rigid={k:torch.tensor(v['center'],requires_grad=True)for k,v in s.rigid.items()}
        E,F,rf=op.evaluate(tx,rigid,time)
        reference=[];reaction={k:np.zeros(3)for k in rigid}
        from dataclasses import replace
        for i,(m,xi)in enumerate(zip(scene.models,x)):
            scene.controls(i,time);u=xi-m.rest
            m.colliders=tuple(replace(c,center=s.rigid[c.name]['center'])if c.name in s.rigid else c for c in m.colliders)
            _,r,_,records=m.contact_response(m.engine.tensor(u));force=-r.detach().numpy()
            for A,target in zip(m.anchor_matrices,m.anchor_targets):force-=A@(u-target)
            for record in records:
                if record['name']in reaction:reaction[record['name']]+=record['force_on_rigid_N'].detach().numpy()
            reference.append(force)
        if scene.pair is not None:
            _,extra,_=scene.pair.evaluate(x);reference=[a+b for a,b in zip(reference,extra)]
        for a,b in zip(F,reference):np.testing.assert_allclose(a.detach().numpy(),b,rtol=1e-10,atol=1e-9)
        for k in rigid:np.testing.assert_allclose(rf[k].detach().numpy(),reaction[k],rtol=1e-10,atol=1e-9)
        derivative=torch.autograd.grad(sum(f.square().sum()for f in F),tx,allow_unused=True)
        assert all(g is None or torch.isfinite(g).all()for g in derivative)
