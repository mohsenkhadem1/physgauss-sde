"""Separate widened-interface development rerun; preserve original model failures.

GPU0 only. n4 follows only if all15 n2 solves satisfy numerical validity; no
accuracy-based case selection or tolerance adjustment is allowed.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import gc,json,signal,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.compositional_neural_mechanics.scenes import make_model,poses
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.interfaces import SharedInterfaceLaw
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.qualification import ROOT,sha,write,nonrigid
from scripts.compositional_neural_mechanics.qualification_nonlinear_range import stats
OUT=ROOT/'runs/compositional-neural-mechanics-v16/wide-interface-equilibrium-development'


def main():
    if OUT.exists():raise ValueError('Preserve prior widened diagnostic')
    base=ROOT/'runs/compositional-neural-mechanics-v16/nonlinear-range-development';cross=ROOT/'runs/compositional-neural-mechanics-v16/crossed-soft-saturated-development'
    parents=[base,cross];plans=[json.loads((p/'manifest.json').read_text()) for p in parents];reports=[json.loads((p/'report.json').read_text()) for p in parents]
    scenes=plans[0]['development_plan']['cases']+plans[1]['development_plan']['cases'];action_names=plans[0]['development_plan']['actions'];fractions=plans[0]['development_plan']['continuation_fractions']
    if len(scenes)!=5:raise ValueError('All five development scenes required')
    bp=ROOT/'runs/compositional-neural-mechanics-v16/scale-separated-prototype/shared-law.pt';ip=ROOT/'runs/compositional-neural-mechanics-v16/interface-wide-range-warm/shared-interface.pt'
    paths=[Path(__file__),bp,ip,*[ROOT/'scripts/compositional_neural_mechanics'/f'{x}.py' for x in ('mechanics','scenes','interfaces','materials','scale_separated_material','qualification','qualification_nonlinear_range')],*[p/name for p in parents for name in ('manifest.json','report.json')]]
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths};deadline=time.time()+1740
    def guard():
        if any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('Widened rerun source/checkpoint changed')
    write(OUT/'plan.json',dict(scope=__doc__,source_sha256=sources,scenes=scenes,actions=action_names,levels=[2,4],continuation_fractions=fractions,device='cuda:0',
        stage_budget_seconds=1740,n4_condition='All15 n2states finite, residual≤1e−8N andJ>.1; accuracy never used to filter',
        checkpoint_pedigree='Original interface checkpoint warm-started for fixed6000steps over narrow+wide δref ranges; original and failed random-start artifacts remain separate'))
    signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Widened stage cap')));signal.alarm(1740)
    bulk=ScaleSeparatedMaterialLaw().double().to('cuda:0');bulk.load_state_dict(torch.load(bp,map_location='cuda:0',weights_only=False)['state_dict'])
    interface=SharedInterfaceLaw().double().to('cuda:0');interface.load_state_dict(torch.load(ip,map_location='cuda:0',weights_only=False)['state_dict'])
    for law in (bulk,interface):
        law.eval()
        for parameter in law.parameters():parameter.requires_grad_(False)
    q=np.stack(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij'),-1).reshape(-1,3);rows=[];comparisons=[]
    for n in (2,4):
        permitted=n==2 or (len(rows)==15 and all(r['state']=='complete' for r in rows))
        for scene in scenes:
            guard();geometry=GeometrySpec(**scene['geometry']);model=make_model(scene,n=n,device='cuda:0',bulk_law=bulk,interface_law=interface) if permitted else None
            for action in action_names:
                row=dict(case=scene['name'],model='shared_wide',n=n,action=action);started=time.monotonic()
                try:
                    if not permitted:raise RuntimeError('n4notrun because original15n2numerical-validity condition failed')
                    if time.time()>=deadline:raise TimeoutError('Stage cap')
                    state=None;pose=poses(geometry)[action]
                    for fraction in fractions:
                        model.set_controls(pose*fraction);state=model.solve(initial=None if state is None else state['displacement_local_m'],force_tolerance_N=1e-8)
                    residual=float(np.linalg.norm(state['residual']));minimum=float(state['J'].min())
                    if not np.isfinite(residual) or residual>1e-8 or minimum<=.1:raise ValueError('Numerical validity failed')
                    field=model.sample_material(state['displacement_local_m'],q);shape=nonrigid(geometry.map(q),field);w=model.reaction_wrenches(state)
                    path=OUT/scene['name']/f'n{n}'/f'{action}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,field=field,shape=shape,forces_N=w['forces_N'],torques_Nm=w['torques_Nm'],rest=model.rest,displacement=state['displacement_local_m'])
                    row.update(state='complete',artifact=str(path.relative_to(ROOT)),sha256=sha(path),residual_N=residual,minimum_J=minimum,force_balance_N=w['force_balance_N'],nonrigid_rms_mm=float(1000*np.sqrt(np.mean(shape**2))),diagnostics=stats(model,state['displacement_local_m']))
                    parent=reports[0] if scene['name']!='development-crossed-soft-saturated' else reports[1]
                    for ref_n,label in [(n,'matched_teacher'),(6,'fine_teacher_total_error')]:
                        reference=next(r for r in parent['rows'] if r['case']==scene['name'] and r['model']=='teacher' and r['n']==ref_n and r['action']==action)
                        if reference['state']!='complete':raise ValueError('Missing declared teacher reference')
                        refpath=ROOT/reference['artifact']
                        if sha(refpath)!=reference['sha256']:raise ValueError('Teacher reference changed')
                        with np.load(refpath) as data:
                            comparison=dict(case=scene['name'],action=action,n=n,comparison=label,reference_sha256=sha(refpath),field_rms_mm=float(1000*np.sqrt(np.mean((field-data['field'])**2))),
                                force_difference_N=np.linalg.norm(w['forces_N']-data['forces_N'],axis=1).tolist(),torque_difference_Nm=np.linalg.norm(w['torques_Nm']-data['torques_Nm'],axis=1).tolist())
                        comparisons.append(comparison)
                except Exception as error:row.update(state='failed',error=repr(error))
                row['elapsed_seconds']=time.monotonic()-started;rows.append(row);write(OUT/'progress.json',dict(rows=rows,comparisons=comparisons));print({k:v for k,v in row.items() if k!='diagnostics'},flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
    guard();result=dict(scope=__doc__,source_sha256=sources,plan_sha256=sha(OUT/'plan.json'),rows=rows,comparisons=comparisons,failed_solves=sum(r['state']!='complete' for r in rows),
        limitation='Knownoraclelatents anddevelopment-only scenes; meshbias andmodelerror separatelyreported; noreserved scoring')
    write(OUT/'report.json',result);write(ROOT/'reports/compositional-neural-mechanics-v16/wide-interface-equilibrium-qualification.json',result)
if __name__=='__main__':main()
