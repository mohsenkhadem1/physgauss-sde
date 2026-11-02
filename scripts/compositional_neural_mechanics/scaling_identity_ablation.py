"""Structural identity check, NOT a matched-training-budget accuracy comparison."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.materials import SharedMaterialLaw
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.interfaces import SharedInterfaceLaw
from scripts.compositional_neural_mechanics.joint_inference import JointMAP,neutral_parameters
from scripts.compositional_neural_mechanics.conditional_inference_pilot import POINTS,GEOMETRY,predict
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
from scripts.compositional_neural_mechanics.dataset_runner import write
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/compositional-neural-mechanics-v16';OUT=ROOT/'runs/compositional-neural-mechanics-v16/scaling-identity-ablation'

def main():
    torch.set_num_threads(1)
    if OUT.exists():raise ValueError('Preserve identity diagnostic')
    OUT.mkdir(parents=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    paths={name:ROOT/'runs/compositional-neural-mechanics-v16'/folder/file for name,folder,file in (
        ('unfactored','constitutive-prototype','shared-law.pt'),('factored','scale-separated-prototype','shared-law.pt'),('interface','interface-wide-range-warm','shared-interface.pt'))}
    files=closure([Path(__file__)])+list(paths.values());pins={str(p.relative_to(ROOT)):sha(p) for p in files}
    plan=dict(scope=__doc__,alpha=np.linspace(-.2,.2,5).tolist(),base_logscale=.2,logscale_range=[0.,.4],n=2,gravity=[0,0,0],source_sha256=pins,
              identity='Multiply bulk and both attachment stiffnesses together: positions invariant, reaction wrenches scale by exp(alpha).',
              caveat='Checkpoints have different training budgets/seeds; compare exact structural invariance only, not model accuracy or statistical confidence.')
    write(REPORT/'scaling-identity-ablation-plan.json',plan)
    poses=[];p=np.zeros((2,6));p[0,0]=-.008;p[1,0]=.008;poses.append(p)
    p=np.zeros((2,6));p[1,5]=.25;poses.append(p)
    observations=[dict(pose=p,points=POINTS,displacement=np.zeros_like(POINTS)) for p in poses]
    base=neutral_parameters();base[[0,3,5]]=[.2,np.log(350.),np.log(800.)]
    rows=[];start=time.monotonic()
    for name,law in [('unfactored',SharedMaterialLaw(latent_dim=3).double()),('factored',ScaleSeparatedMaterialLaw().double())]:
        law.load_state_dict(torch.load(paths[name],weights_only=False)['state_dict']);interface=SharedInterfaceLaw();interface.load_state_dict(torch.load(paths['interface'],weights_only=False)['state_dict'])
        estimator=JointMAP(GEOMETRY,bulk_family='mooney_rivlin',interface_family=0,heterogeneity=0,observations=observations,position_sigma=50e-6,n=2,bulk_law=law,interface_law=interface)
        reference=[predict(estimator,base,p) for p in poses]
        for alpha in plan['alpha']:
            theta=base.copy();theta[[0,3,5]]+=alpha
            for index,pose in enumerate(poses):
                if time.monotonic()-start>600:raise TimeoutError('Bounded structural identity study')
                value=predict(estimator,theta,pose);original=reference[index]
                rows.append(dict(model=name,alpha=alpha,action=index,
                                 position_drift_coordinate_rms_m=float(np.sqrt(np.mean((value['displacement']-original['displacement'])**2))),
                                 reaction_scale_error_coordinate_rms_N=float(np.sqrt(np.mean((value['forces']-np.exp(alpha)*original['forces'])**2))),
                                 torque_scale_error_coordinate_rms_Nm=float(np.sqrt(np.mean((value['torques']-np.exp(alpha)*original['torques'])**2)))))
    if any(sha(ROOT/p)!=h for p,h in pins.items()):raise ValueError('Identity source/checkpoint changed')
    summary={name:{key:max(r[key] for r in rows if r['model']==name) for key in rows[0] if key.endswith(('_m','_N','_Nm'))} for name in ('unfactored','factored')}
    write(REPORT/'scaling-identity-ablation-results.json',dict(plan_sha256=sha(REPORT/'scaling-identity-ablation-plan.json'),rows=rows,maximum_deviation=summary,seconds=time.monotonic()-start))

if __name__=='__main__':main()
