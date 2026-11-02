"""Pinned4-scene DEVELOPMENT-only nonlinear-support operating-range qualification."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,concurrent.futures,gc,json,queue,signal,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.compositional_neural_mechanics.scenes import make_model,poses,split_cells
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec,constitutive_response
from scripts.compositional_neural_mechanics.materials import invariants,teacher_energy
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.interfaces import SharedInterfaceLaw
from scripts.compositional_neural_mechanics.qualification import ROOT,sha,write,nonrigid
OUT=ROOT/'runs/compositional-neural-mechanics-v16/nonlinear-range-development'
PLAN=ROOT/'reports/compositional-neural-mechanics-v16/nonlinear-range-development-plan.json'


def guard(plan):
    if any(sha(ROOT/p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('Nonlinear-range stage source/checkpoint changed')


def stats(model,u):
    e=model.engine;ue=e.tensor(u)[e.elements];F=e.I+torch.einsum('eai,qeaj->qeij',ue,e.grad);J=torch.linalg.det(F);q1,q2,_=invariants(F);weights=e.weights/e.weights.sum();stretch=torch.linalg.svdvals(F)
    result={}
    for name,value in [('q1',q1),('q2',q2),('J',J)]:
        flat=value.detach().cpu().numpy().ravel();result[name]=dict(min=float(flat.min()),max=float(flat.max()),volume_weighted_mean=float((weights*value).sum()),quadrature_unweighted_percentiles50_95_99=np.quantile(flat,[.5,.95,.99]).tolist())
    result['principal_stretch_min_max']=[float(stretch.min()),float(stretch.max())];result['interface_slip_over_delta_ref']=[]
    for side,data in enumerate(model.patch_data):
        delta=np.einsum('qa,eai->qei',data['N'],(u-model.anchor_targets[side])[data['ids']]);s=np.linalg.norm(delta,axis=-1)/.002
        result['interface_slip_over_delta_ref'].append(dict(min=float(s.min()),max=float(s.max()),weighted_mean=float((data['weights']*s).sum()),quadrature_unweighted_percentiles50_95_99=np.quantile(s,[.5,.95,.99]).tolist()))
    z=model.material_latent_field(e.reference_points)[...,:2];stresses=[];energies=[]
    for family in ('mooney_rivlin','yeoh'):
        W,P,_=constitutive_response(lambda F,z:teacher_energy(F,z,family),F,z);energies.append(float((e.weights*W).sum()));stresses.append(P)
    result['same_F_family_difference']=dict(energy_integral_MR_J=energies[0],energy_integral_Yeoh_J=energies[1],
        volume_weighted_stress_difference_rms_Pa=float(torch.sqrt((weights*(stresses[0]-stresses[1]).square().sum((-1,-2))/9).sum())),
        relative_stress_L2_to_MR=float(torch.sqrt((weights*(stresses[0]-stresses[1]).square().sum((-1,-2))).sum()/(weights*stresses[0].square().sum((-1,-2))).sum())))
    return result


def worker(index,device):
    stage=json.loads((OUT/'manifest.json').read_text());guard(stage);plan=stage['development_plan'];scene=plan['cases'][index]
    remaining=int(stage['deadline_unix']-time.time())
    if remaining<=0:raise TimeoutError('Whole-stage budget expired')
    signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Whole-stage cap')));signal.alarm(remaining)
    bulk=ScaleSeparatedMaterialLaw().double().to(device);bulk.load_state_dict(torch.load(ROOT/stage['bulk_checkpoint'],map_location=device,weights_only=False)['state_dict'])
    interface=SharedInterfaceLaw().double().to(device);interface.load_state_dict(torch.load(ROOT/stage['interface_checkpoint'],map_location=device,weights_only=False)['state_dict'])
    for law in (bulk,interface):
        law.eval()
        for parameter in law.parameters():parameter.requires_grad_(False)
    q=np.stack(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij'),-1).reshape(-1,3);geometry=GeometrySpec(**scene['geometry']);actions=poses(geometry);rows=[];began=time.monotonic()
    for kind,n in [('teacher',2),('teacher',4),('teacher',6),('shared',2)]:
        model=make_model(scene,n=n,device=device,bulk_law=bulk if kind=='shared' else None,interface_law=interface if kind=='shared' else None)
        for action in plan['actions']:
            started=time.monotonic();row=dict(case=scene['name'],model=kind,n=n,action=action)
            try:
                state=None
                for fraction in plan['continuation_fractions']:
                    model.set_controls(actions[action]*fraction);state=model.solve(initial=None if state is None else state['displacement_local_m'],force_tolerance_N=1e-8)
                field=model.sample_material(state['displacement_local_m'],q);shape=nonrigid(geometry.map(q),field);w=model.reaction_wrenches(state)
                path=OUT/scene['name']/f'{kind}-n{n}'/f'{action}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,field=field,shape=shape,forces_N=w['forces_N'],torques_Nm=w['torques_Nm'],origins_world_m=w['origins_world_m'],rest=model.rest,displacement=state['displacement_local_m'])
                row.update(state='complete',residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),force_balance_N=w['force_balance_N'],
                    nonrigid_rms_mm=float(1000*np.sqrt(np.mean(shape**2))),artifact=str(path.relative_to(ROOT)),sha256=sha(path),diagnostics=stats(model,state['displacement_local_m']))
            except Exception as error:row.update(state='failed',error=repr(error))
            row['elapsed_seconds']=time.monotonic()-started;rows.append(row);write(OUT/f'case-{index}-progress.json',dict(rows=rows));print({k:v for k,v in row.items() if k!='diagnostics'},flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    guard(stage);write(OUT/f'case-{index}.json',dict(rows=rows,elapsed_seconds=time.monotonic()-began))


def aggregate(stage):
    rows=[]
    for i in range(4):
        path=OUT/f'case-{i}.json'
        if path.exists():rows.extend(json.loads(path.read_text())['rows'])
        else:
            progress=OUT/f'case-{i}-progress.json';existing=json.loads(progress.read_text())['rows'] if progress.exists() else [];rows.extend(existing)
            keys={(r['model'],r['n'],r['action']) for r in existing}
            for model,n in [('teacher',2),('teacher',4),('teacher',6),('shared',2)]:
                for action in stage['development_plan']['actions']:
                    if (model,n,action) not in keys:rows.append(dict(case=stage['development_plan']['cases'][i]['name'],model=model,n=n,action=action,state='failed',error='Stage worker did not finish this declared solve'))
    comparisons=[]
    for scene in stage['development_plan']['cases']:
        for action in stage['development_plan']['actions']:
            for label,pair in [('mesh4to6',[('teacher',4),('teacher',6)]),('learned_same_mesh',[('shared',2),('teacher',2)]),('coarse_teacher_bias',[('teacher',2),('teacher',6)])]:
                records=[next(r for r in rows if r['case']==scene['name'] and r['action']==action and (r['model'],r['n'])==item) for item in pair]
                record=dict(case=scene['name'],action=action,comparison=label)
                if any(r['state']!='complete' for r in records):record.update(complete=False,passed=False,reason='Missing qualified solve')
                else:
                    values=[]
                    for r in records:
                        path=ROOT/r['artifact']
                        if sha(path)!=r['sha256']:raise ValueError('Completed nonlinear state changed')
                        with np.load(path) as data:values.append({k:data[k].copy() for k in data.files})
                    a,b=values;field=float(np.sqrt(np.mean((a['field']-b['field'])**2)));force=np.linalg.norm(a['forces_N']-b['forces_N'],axis=1);torque=np.linalg.norm(a['torques_Nm']-b['torques_Nm'],axis=1)
                    record.update(complete=True,field_rms_mm=1000*field,canonical_shape_difference_rms_mm=float(1000*np.sqrt(np.mean((a['shape']-b['shape'])**2))),force_difference_N=force.tolist(),torque_difference_Nm=torque.tolist())
                    if label=='mesh4to6':
                        g=stage['development_plan']['gates'];ft=g['force_absolute_N']+g['force_relative']*np.linalg.norm(b['forces_N'],axis=1);tt=g['torque_absolute_Nm']+g['torque_relative']*np.linalg.norm(b['torques_Nm'],axis=1)
                        record.update(passed=bool(field<=g['field_rms_m'] and (force<=ft).all() and (torque<=tt).all()),force_limits_N=ft.tolist(),torque_limits_Nm=tt.tolist())
                comparisons.append(record)
    guard(stage);result=dict(scope=__doc__,source_sha256=stage['source_sha256'],manifest_sha256=sha(OUT/'manifest.json'),rows=rows,comparisons=comparisons,
        failed_solves=sum(r['state']!='complete' for r in rows),all_mesh_gates_pass=all(r['passed'] for r in comparisons if r['comparison']=='mesh4to6'),
        limitations=['Four declared development scenes only; no heldout generation','No transfer from earlier K6000linear-interface qualification','NonrigidRMS uses125uniform materialqueries, notvolumeweightednorm','Sharedmodel oraclelatents; no identificationclaim'])
    write(OUT/'report.json',result);write(ROOT/'reports/compositional-neural-mechanics-v16/nonlinear-range-qualification.json',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int);p.add_argument('--device');a=p.parse_args()
    if a.case is not None:
        if a.case not in range(4) or a.device not in ('cuda:1','cuda:2'):raise ValueError('Declared case/GPU required')
        return worker(a.case,a.device)
    if OUT.exists():raise ValueError('Preserve previous nonlinear qualification')
    plan=json.loads(PLAN.read_text())
    if len(plan['cases'])!=4:raise ValueError('Exactly four development scenes required')
    for scene in plan['cases']:
        factors=scene['factors'];cell=tuple(factors[k] for k in ('geometry','bulk','support','heterogeneity'))
        if cell not in split_cells()['train']:raise ValueError('Reserved composition cannot enter developmentpilot')
    bulk='runs/compositional-neural-mechanics-v16/scale-separated-prototype/shared-law.pt';interface='runs/compositional-neural-mechanics-v16/interface-prototype/shared-interface.pt'
    source=[Path(__file__),PLAN,ROOT/bulk,ROOT/interface,*[ROOT/'scripts/compositional_neural_mechanics'/f'{x}.py' for x in ('mechanics','scenes','materials','interfaces','scale_separated_material','qualification')],ROOT/'scripts/phystwin/tensor_fixture.py',ROOT/'scripts/phystwin/consistent_fixture.py']
    stage=dict(scope=__doc__,development_plan=plan,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source},bulk_checkpoint=bulk,interface_checkpoint=interface,
               deadline_unix=time.time()+1800,devices=['cuda:1','cuda:2'],workers=2,reset='Four within-action continuation steps fromzero for EVERY action')
    write(OUT/'manifest.json',stage);pending=queue.Queue()
    for i in range(4):pending.put(i)
    def runner(device):
        while True:
            try:i=pending.get_nowait()
            except queue.Empty:return
            remaining=stage['deadline_unix']-time.time()
            if remaining<=0:return
            with (OUT/f'case-{i}.log').open('w') as stream:
                child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--case',str(i),'--device',device],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            write(OUT/f'case-{i}-exit.json',dict(exit_code=code))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(runner,['cuda:1','cuda:2']))
    aggregate(stage)
if __name__=='__main__':main()
