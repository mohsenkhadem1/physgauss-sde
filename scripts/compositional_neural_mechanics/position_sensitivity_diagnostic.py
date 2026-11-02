"""Read-only local DEVELOPMENT sensitivity; no fitting or posterior claim."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.linalg import splu
from scripts.compositional_neural_mechanics.full_continuous_inference_pilot import build,pack
from scripts.compositional_neural_mechanics.conditional_inference_pilot import POINTS
from scripts.compositional_neural_mechanics.scenes import sample_scene,split_cells,poses,make_model
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.joint_inference import NAMES,neutral_parameters
from scripts.compositional_neural_mechanics.inference import observation_matrix
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/compositional-neural-mechanics-v16/position-sensitivity-development'
REPORT=ROOT/'reports/compositional-neural-mechanics-v16/position-sensitivity-development.json'

def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    parent_path=ROOT/'reports/compositional-neural-mechanics-v16/full-continuous-inference-pilot-plan.json'
    parent=json.loads(parent_path.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if any(sha(ROOT/p)!=h for p,h in parent['source_sha256'].items()):raise ValueError('Running pilot sources changed')
    files=closure([Path(__file__)])
    plan=dict(scope='One development scene, analytic teacher, neutral and true parameters, positions only with known geometry/categories/gravity/density',parent_plan_sha256=sha(parent_path),
              world_index=0,parameter_names=NAMES,position_sigma_m=50e-6,prior_std=[1.]*15,n=2,method='15 residual JVPs per equilibrium; sparse H inverse; normalize outputs by observation sigma and inputs by prior std',
              local_only='Local differential conditioning, not global or practical identifiability; no force observations, no fitting or posterior calibration',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rng=np.random.default_rng(parent['scene_seed']);scene=sample_scene(split_cells()['train'][parent['train_cell_indices'][0]],rng,'full-continuous-dev-0',gravity=(0,0,-9.81))
    g=GeometrySpec(**scene['geometry']);categories=dict(bulk_family=scene['bulk_family'],interface_family=int(scene['interface_latents'][0][2]),heterogeneity=scene['factors']['heterogeneity'])
    controls=poses(g);obs=[dict(pose=controls[k],points=POINTS,displacement=np.zeros_like(POINTS)) for k in parent['observed_actions']]
    estimator=build(g,categories,obs,False);rows=[]
    for label,values in [('neutral',neutral_parameters()),('truth',pack(scene))]:
        blocks=[];diagnostics=[]
        for index,o in enumerate(obs):
            if time.monotonic()-start>280:raise TimeoutError('Five-minute diagnostic budget')
            model=make_model(estimator.scene,**estimator.kw);z=torch.tensor(values,dtype=torch.float64);estimator.bind(model,z);u=None
            for fraction in estimator.continuation:
                model.set_controls(o['pose']*fraction);state=model.solve(initial=u,force_tolerance_N=1e-10,maximum_iterations=60);u=state['displacement_local_m']
            H=model.evaluate(u,True)['tangent'];factor=splu(H.tocsc());ut=torch.tensor(u,dtype=torch.float64)
            def residual(live):estimator.bind(model,live);return model.torch_residual(ut)
            rz=[]
            for j in range(15):
                direction=torch.zeros_like(z);direction[j]=estimator.std[j]
                _,derivative=torch.autograd.functional.jvp(residual,z,direction,create_graph=False,strict=False)
                rz.append(derivative.detach().numpy().ravel())
            Rz=np.stack(rz,axis=1);du=factor.solve(-Rz)
            M=observation_matrix(model,o['points']);J=np.einsum('pn,ncj->pcj',M,du.reshape(-1,3,15)).reshape(-1,15)/estimator.sigmas[0]
            if not np.isfinite(J).all():raise ValueError('Nonfinite sensitivity')
            blocks.append(J);diagnostics.append(dict(action=parent['observed_actions'][index],equilibrium_residual_N=float(np.linalg.norm(state['residual'])),linear_relative_residual=float(np.linalg.norm(H@du+Rz)/max(1e-30,np.linalg.norm(Rz)))))
        J=np.concatenate(blocks);U,s,Vt=np.linalg.svd(J,full_matrices=False);conditional=[]
        for j,name in enumerate(NAMES):
            others=np.delete(J,j,axis=1);projection=others@np.linalg.lstsq(others,J[:,j],rcond=None)[0]
            conditional.append(dict(name=name,column_norm=float(np.linalg.norm(J[:,j])),after_other14_projection_norm=float(np.linalg.norm(J[:,j]-projection))))
        common=np.zeros(15);common[[0,3,5]]=1/np.sqrt(3)
        known=J[:,[0,3,5]];ks=np.linalg.svd(known,compute_uv=False)
        weak=[]
        for i in range(1,6):
            v=Vt[-i];ids=np.argsort(abs(v))[::-1][:6]
            weak.append(dict(singular_value=float(s[-i]),coefficients={NAMES[j]:float(v[j]) for j in ids}))
        np.savez_compressed(OUT/(label+'.npz'),normalized_jacobian=J,singular_values=s,right_vectors=Vt,parameters=values)
        rows.append(dict(at=label,singular_values=s.tolist(),condition_number=float(s[0]/s[-1]),modes_above_unit_noise_sensitivity=int((s>1).sum()),
                         weakest_modes=weak,columns=conditional,known_nuisance_three_singular_values=ks.tolist(),known_nuisance_condition_number=float(ks[0]/ks[-1]),
                         common_scale_direction_sensitivity=float(np.linalg.norm(J@common)),diagnostics=diagnostics))
    report=dict(plan=plan,results=rows,elapsed_seconds=time.monotonic()-start,all_source_hashes_unchanged=all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items()),
                interpretation='Unit singular value means a unit prior-scaled infinitesimal direction gives total standardized observation change of one. Independent Gaussian noise assumed. Weak modes may be masked by prior precision; nonzero singular values do not establish unique physical recovery.')
    REPORT.write_text(json.dumps(report,indent=2)+'\n');print(REPORT,flush=True)

if __name__=='__main__':main()
