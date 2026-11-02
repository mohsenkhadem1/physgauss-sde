"""Small learned bulk+interface common-scale control, zero body loads."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.compositional_neural_mechanics.mechanics import Q2Mechanics,GeometrySpec
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.interfaces import SharedInterfaceLaw
from scripts.compositional_neural_mechanics.qualification import ROOT,write,sha


def main():
    out=ROOT/'runs/compositional-neural-mechanics-v16/coupled-scale-control'
    if out.exists():raise ValueError('Preserve existing scale control')
    bp=ROOT/'runs/compositional-neural-mechanics-v16/scale-separated-prototype/shared-law.pt';ip=ROOT/'runs/compositional-neural-mechanics-v16/interface-prototype/shared-interface.pt'
    bulk=ScaleSeparatedMaterialLaw().double();bulk.load_state_dict(torch.load(bp,weights_only=False,map_location='cpu')['state_dict'])
    interface=SharedInterfaceLaw().double();interface.load_state_dict(torch.load(ip,weights_only=False,map_location='cpu')['state_dict'])
    for law in (bulk,interface):
        for parameter in law.parameters():parameter.requires_grad_(False)
    sources=[Path(__file__),ROOT/'scripts/compositional_neural_mechanics/mechanics.py',ROOT/'scripts/compositional_neural_mechanics/scale_separated_material.py',ROOT/'scripts/compositional_neural_mechanics/materials.py',ROOT/'scripts/compositional_neural_mechanics/interfaces.py',bp,ip]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources};write(out/'plan.json',dict(source_sha256=hashes,scales=[1.,2.],zero_body_load=True,field_tolerance_m=1e-9,force_tolerance_N=1e-7,torque_tolerance_Nm=1e-8))
    parameters=torch.tensor([[.45,.25,np.log(.35),np.log(.3)],[.6,.7,np.log(.3),np.log(.4)]],dtype=torch.float64)
    pose=np.zeros((2,6));pose[0,0]=-.004;pose[1,0]=.004;pose[0,3]=-.1;pose[1,5]=.15
    states=[]
    for scale in (1.,2.):
        def material(q):return torch.stack([q[...,0]*.6-.3+np.log(scale),q[...,0]*0+.2,q[...,0]*0-1],dim=-1)
        def support(q):return q.new_tensor([np.log(6000*scale),.3,.7]).expand(*q.shape[:-1],3)
        model=Q2Mechanics(GeometrySpec(taper=.2,arch=.01),n=1,bulk_energy=bulk.energy,material_latent_field=material,
                          interface_energy=interface.energy,interface_latent_fields=(support,support),interface_geometry_parameters=parameters)
        state=None
        for fraction in (.5,1.):
            model.set_controls(pose*fraction);state=model.solve(initial=None if state is None else state['displacement_local_m'],force_tolerance_N=1e-9)
        wrench=model.reaction_wrenches(state);states.append((state,wrench))
        np.savez_compressed(out/f'scale-{scale:g}.npz',displacement=state['displacement_local_m'],**wrench)
    field=float(np.sqrt(np.mean((states[0][0]['displacement_local_m']-states[1][0]['displacement_local_m'])**2)))
    force=float(np.max(abs(2*states[0][1]['forces_N']-states[1][1]['forces_N'])));torque=float(np.max(abs(2*states[0][1]['torques_Nm']-states[1][1]['torques_Nm'])))
    if any(sha(ROOT/p)!=h for p,h in hashes.items()):raise ValueError('Sources changed')
    result=dict(scope=__doc__,source_sha256=hashes,field_difference_m=field,force_scaling_error_N=force,torque_scaling_error_Nm=torque,
                residuals_N=[float(np.linalg.norm(s['residual'])) for s,w in states],minimum_J=[float(s['J'].min()) for s,w in states],
                force_balance_N=[w['force_balance_N'] for s,w in states],passed=field<1e-9 and force<1e-7 and torque<1e-8,
                limitation='Single development geometry/pose numerical control, not global learned-law qualification; zero-load absolute common scale remains visually unidentifiable')
    write(ROOT/'reports/compositional-neural-mechanics-v16/coupled-scale-control.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
