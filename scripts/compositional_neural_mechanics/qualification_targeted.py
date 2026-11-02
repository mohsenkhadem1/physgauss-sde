"""Exactly three n6→8 torque follow-ups; single bounded15-minute stage.

Prior failures stay unchanged. This is the K=6000N/m linear-support diagnostic,
not qualification of the forthcoming nonlinear-support parameter range.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,concurrent.futures,json,queue,signal,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.compositional_neural_mechanics.mechanics import Q2Mechanics,GeometrySpec,constitutive_response
from scripts.compositional_neural_mechanics.materials import teacher_energy,invariants
from scripts.compositional_neural_mechanics.qualification import ROOT,write,sha,nonrigid
BASE=ROOT/'runs/compositional-neural-mechanics-v16/mechanics-reference-fine'
OUT=ROOT/'runs/compositional-neural-mechanics-v16/mechanics-reference-targeted8'
CASES=[('rectangular','mooney_rivlin','right_bend'),('tapered_arch','mooney_rivlin','stretch'),('tapered_arch','yeoh','stretch')]


def strain_stats(model,u):
    engine=model.engine;ue=engine.tensor(u)[engine.elements];F=engine.I+torch.einsum('eai,qeaj->qeij',ue,engine.grad)
    J=torch.linalg.det(F);q1,q2,_=invariants(F);weight=engine.weights/engine.weights.sum();stretch=torch.linalg.svdvals(F)
    result={}
    for name,value in [('q1',q1),('q2',q2),('J',J)]:
        flat=value.detach().cpu().numpy().ravel();result[name]=dict(min=float(flat.min()),max=float(flat.max()),volume_weighted_mean=float((weight*value).sum()),quadrature_unweighted_percentiles_50_95_99=np.quantile(flat,[.5,.95,.99]).tolist())
    result['principal_stretch_min_max']=[float(stretch.min()),float(stretch.max())]
    z=F.new_tensor([0.,.2]).expand(*F.shape[:-2],2)
    energies=[];stresses=[]
    for family in ('mooney_rivlin','yeoh'):
        W,P,_=constitutive_response(lambda f,z:teacher_energy(f,z,family),F,z);energies.append(W);stresses.append(P)
    result['same_F_family_difference']=dict(energy_integral_MR_J=float((engine.weights*energies[0]).sum()),energy_integral_Yeoh_J=float((engine.weights*energies[1]).sum()),
        volume_weighted_stress_difference_rms_Pa=float(torch.sqrt((weight*(stresses[0]-stresses[1]).square().sum((-1,-2))/9).sum())),
        relative_stress_L2_to_MR=float(torch.sqrt((weight*(stresses[0]-stresses[1]).square().sum((-1,-2))).sum()/(weight*stresses[0].square().sum((-1,-2))).sum())))
    return result


def guard(plan):
    if any(sha(ROOT/p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('Targeted qualification source/evidence changed')


def worker(index,device):
    plan=json.loads((OUT/'plan.json').read_text());guard(plan);remaining=int(plan['stage_deadline_unix']-time.time())
    if remaining<=0:raise TimeoutError('Stage budget expired')
    signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Targeted stage cap')));signal.alarm(remaining)
    geometry_name,family,action=CASES[index];spec=plan['geometries'][geometry_name];geometry=GeometrySpec(**spec);pose=np.asarray(plan['poses'][action]);record=dict(index=index,geometry=geometry_name,family=family,action=action,device=device)
    try:
        def latent(q):return q.new_tensor([0.,.2]).expand(*q.shape[:-1],2)
        energy=lambda F,z:teacher_energy(F,z,family)
        prior_path=BASE/geometry_name/family/f'{action}.npz'
        with np.load(prior_path) as data:prior={k:data[k].copy() for k in data.files}
        coarse=Q2Mechanics(geometry,n=6,bulk_energy=energy,material_latent_field=latent,device=device)
        fine=Q2Mechanics(geometry,n=8,bulk_energy=energy,material_latent_field=latent,device=device)
        initial=coarse.sample_material(prior['displacement'],fine.material_coordinates);fine.set_controls(pose)
        state=fine.solve(initial=initial,force_tolerance_N=1e-8);w=fine.reaction_wrenches(state)
        q=np.stack(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij'),-1).reshape(-1,3);field=fine.sample_material(state['displacement_local_m'],q)
        path=OUT/f'case-{index}.npz';np.savez_compressed(path,field=field,shape=nonrigid(geometry.map(q),field),forces_N=w['forces_N'],torques_Nm=w['torques_Nm'],rest=fine.rest,displacement=state['displacement_local_m'])
        delta=float(np.sqrt(np.mean((field-prior['field'])**2)));force=np.linalg.norm(w['forces_N']-prior['forces_N'],axis=1);torque=np.linalg.norm(w['torques_Nm']-prior['torques_Nm'],axis=1)
        ft=1e-4+.005*np.linalg.norm(w['forces_N'],axis=1);tt=1e-5+.005*np.linalg.norm(w['torques_Nm'],axis=1)
        record.update(state='complete',passed=bool(delta<=2e-5 and (force<=ft).all() and (torque<=tt).all()),field_rms_mm=delta*1000,
                      force_difference_N=force.tolist(),force_limits_N=ft.tolist(),torque_difference_Nm=torque.tolist(),torque_limits_Nm=tt.tolist(),
                      residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),force_balance_N=w['force_balance_N'],
                      artifact=str(path.relative_to(ROOT)),sha256=sha(path),n6_strain=strain_stats(coarse,prior['displacement']),n8_strain=strain_stats(fine,state['displacement_local_m']))
        guard(plan)
    except Exception as error:record.update(state='failed',passed=False,error=repr(error))
    write(OUT/f'case-{index}.json',record);print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int);p.add_argument('--device');a=p.parse_args()
    if a.case is not None:
        if a.case not in range(3) or a.device not in ('cuda:1','cuda:2'):raise ValueError('Declared case and GPU1/2 only')
        return worker(a.case,a.device)
    if OUT.exists():raise ValueError('Preserve previous targeted stage')
    old=ROOT/'runs/compositional-neural-mechanics-v16/mechanics-oracle-qualified/plan.json';original=json.loads(old.read_text())
    sources=[Path(__file__),ROOT/'scripts/compositional_neural_mechanics/mechanics.py',ROOT/'scripts/compositional_neural_mechanics/materials.py',ROOT/'scripts/compositional_neural_mechanics/qualification.py',old]
    sources.extend(BASE/g/'report.json' for g in ('rectangular','tapered_arch'))
    sources.extend(BASE/g/f/f'{act}.npz' for g,f,act in CASES)
    plan=dict(scope=__doc__,cases=CASES,geometries=original['geometries'],poses=original['poses'],gates=original['gates'],
              source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},stage_deadline_unix=time.time()+900,
              initial_state='Same-action n6equilibrium interpolated in materialcoordinates to n8; no unrelated probe warmstart',workers=2,devices=['cuda:1','cuda:2'])
    write(OUT/'plan.json',plan);pending=queue.Queue()
    for i in range(3):pending.put(i)
    def runner(device):
        while True:
            try:i=pending.get_nowait()
            except queue.Empty:return
            remaining=plan['stage_deadline_unix']-time.time()
            if remaining<=0:write(OUT/f'case-{i}.json',dict(index=i,state='failed',passed=False,error='Stage budget expired before launch'));continue
            with (OUT/f'case-{i}.log').open('w') as stream:
                child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--case',str(i),'--device',device],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            if not (OUT/f'case-{i}.json').exists():write(OUT/f'case-{i}.json',dict(index=i,state='failed',passed=False,error='Worker exited without complete record',exit_code=code))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(runner,['cuda:1','cuda:2']))
    guard(plan);rows=[json.loads((OUT/f'case-{i}.json').read_text()) for i in range(3)]
    write(OUT/'report.json',dict(plan_sha256=sha(OUT/'plan.json'),source_sha256=plan['source_sha256'],rows=rows,all_gates_pass=all(r['passed'] for r in rows),
          limitation='Three targeted cases selected for prior torquequalification failure; no replacement/removal of failed4→6results, no transfer to newnonlinear-support range'))
if __name__=='__main__':main()
