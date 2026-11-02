"""Bounded Stage-A development oracle checks; no training/test dataset generation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.compositional_neural_mechanics.mechanics import Q2Mechanics,GeometrySpec
from scripts.compositional_neural_mechanics.materials import SharedMaterialLaw,teacher_energy
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def nonrigid(rest,u):
    a=rest-rest.mean(0);b=rest+u;b=b-b.mean(0);U,_,V=np.linalg.svd(b.T@a);fix=np.eye(3);fix[-1,-1]=np.linalg.det(U@V)
    return b@(U@fix@V)-a


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1',choices=['cpu','cuda:1','cuda:2']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior qualification')
    checkpoint=ROOT/'runs/compositional-neural-mechanics-v16/constitutive-prototype/shared-law.pt'
    sources=[Path(__file__),ROOT/'scripts/compositional_neural_mechanics/mechanics.py',ROOT/'scripts/compositional_neural_mechanics/materials.py',
             ROOT/'scripts/phystwin/tensor_fixture.py',ROOT/'scripts/phystwin/consistent_fixture.py',checkpoint]
    hashes={str(x.relative_to(ROOT)):sha(x) for x in sources};geometries={'rectangular':GeometrySpec(),'tapered_arch':GeometrySpec((.20,.07,.09),.25,.012)}
    actions={};x=np.zeros((2,6));x[0,0]=-.01;x[1,0]=.01;actions['stretch']=x
    x=np.zeros((2,6));x[0,3]=-.25;x[1,3]=.25;actions['opposed_twist']=x
    x=np.zeros((2,6));x[1,5]=.25;actions['right_bend']=x
    plan=dict(scope=__doc__,source_sha256=hashes,device=a.device,families=['mooney_rivlin','yeoh'],levels=[2,3],learned_level=2,
              geometries={k:vars(v) for k,v in geometries.items()},poses={k:v.tolist() for k,v in actions.items()},
              gates=dict(field_rms_m=2e-5,force_absolute_N=1e-4,force_relative=.005,torque_absolute_Nm=1e-5,torque_relative=.005),
              reset='Every action starts zero displacement, with two within-action continuation steps',
              material='Uniform oracle descriptor logscale0,shape.2,familyflag±1; development diagnostic only, not heldouttest')
    write(a.output/'plan.json',plan)
    law=SharedMaterialLaw(latent_dim=3).double().to(a.device);law.load_state_dict(torch.load(checkpoint,map_location=a.device,weights_only=False)['state_dict']);law.eval()
    for parameter in law.parameters():parameter.requires_grad_(False)
    q=np.stack(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij'),-1).reshape(-1,3);rows=[];results={};started=time.monotonic()
    for geometry_name,geometry in geometries.items():
        for family in plan['families']:
            for model_name,n in [('teacher',2),('teacher',3),('shared',2)]:
                zvalue=[0.,.2,-1. if family=='mooney_rivlin' else 1.]
                def latent(q):return q.new_tensor(zvalue).expand(*q.shape[:-1],3)
                energy=(lambda F,z:teacher_energy(F,z[...,:2],family=family)) if model_name=='teacher' else law.energy_density
                model=Q2Mechanics(geometry,n=n,bulk_energy=energy,material_latent_field=latent,device=a.device)
                for action,pose in actions.items():
                    record=dict(geometry=geometry_name,family=family,model=model_name,n=n,action=action);key=(geometry_name,family,model_name,n,action)
                    try:
                        state=None
                        for fraction in (.5,1.):
                            model.set_controls(pose*fraction);state=model.solve(initial=None if state is None else state['displacement_local_m'],force_tolerance_N=1e-8)
                        field=model.sample_material(state['displacement_local_m'],q);w=model.reaction_wrenches(state);shape=nonrigid(geometry.map(q),field)
                        data=dict(field=field,shape=shape,forces_N=w['forces_N'],torques_Nm=w['torques_Nm'],displacement=state['displacement_local_m'],rest=model.rest)
                        target=a.output/geometry_name/family/f'{model_name}-n{n}'/f'{action}.npz';target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**data)
                        record.update(state='complete',residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),force_balance_N=w['force_balance_N'],nonrigid_rms_mm=float(1000*np.sqrt(np.mean(shape**2))),artifact=str(target.relative_to(ROOT)),sha256=sha(target))
                        results[key]=data
                    except Exception as error:record.update(state='failed',error=repr(error))
                    rows.append(record);write(a.output/'progress.json',dict(rows=rows));print(record,flush=True)
    comparisons=[]
    for geometry in geometries:
        for family in plan['families']:
            for action in actions:
                for kind,other in [('mesh',('teacher',3)),('learned_oracle',('shared',2))]:
                    first=results.get((geometry,family,'teacher',2,action));second=results.get((geometry,family,*other,action));r=dict(geometry=geometry,family=family,action=action,comparison=kind)
                    if first is None or second is None:r.update(passed=False,reason='Missing solve')
                    else:
                        field=float(np.sqrt(np.mean((first['field']-second['field'])**2)));force=np.linalg.norm(first['forces_N']-second['forces_N'],axis=1);torque=np.linalg.norm(first['torques_Nm']-second['torques_Nm'],axis=1)
                        r.update(field_rms_mm=1000*field,force_difference_N=force.tolist(),torque_difference_Nm=torque.tolist())
                        if kind=='mesh':r['passed']=bool(field<=2e-5 and (force<=1e-4+.005*np.linalg.norm(second['forces_N'],axis=1)).all() and (torque<=1e-5+.005*np.linalg.norm(second['torques_Nm'],axis=1)).all())
                    comparisons.append(r)
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('Qualification source/checkpoint changed')
    write(a.output/'report.json',dict(plan_sha256=sha(a.output/'plan.json'),source_sha256=hashes,rows=rows,comparisons=comparisons,
        all_mesh_gates_pass=all(r['passed'] for r in comparisons if r['comparison']=='mesh'),failed_solves=sum(r['state']!='complete' for r in rows),elapsed_seconds=time.monotonic()-started,
        limitations=['Same solver and analytic families, not independent experimental truth','Linear attachments here; nonlinear shared interface requires its own qualification','NonrigidRMS uses125uniform material queries, not volume integration','Teacher/shared comparison usesoracle descriptors andsame n2mesh; no inverseidentificationclaim']))
if __name__=='__main__':main()
