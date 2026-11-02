"""Fixed two-case contact resolution development, no physics/model changes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,multiprocessing
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.contact_resolution import ResolvedContactQ2,sampled_contact_areas,exact_flat_sphere_patch
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane,RigidSphere,dense_surface_gap_check
from scripts.compositional_neural_mechanics.conditional_inference_pilot import sha,write
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';OLD=ROOT/'runs/compositional-neural-mechanics-v16/contact-development';OUT=ROOT/'runs/compositional-neural-mechanics-v16/contact-resolution-refinement';PLAN=R/'contact-resolution-refinement-plan.json'
CASES=('sphere_indentation','gravity_plane_sphere_indentation')


def prepare():
    if PLAN.exists() or OUT.exists():raise ValueError('Preserve previous plan/output')
    old=R/'contact-development.json';prior=json.loads(old.read_text())
    if any(sha(ROOT/p)!=h for p,h in prior['plan']['source_sha256'].items()):raise ValueError('Original contact source changed')
    inputs=[old]+[OLD/f'{c}-n3-q16-k1e+07{s}'for c in CASES for s in ('.json','.npz')]
    files=closure([Path(__file__)])+[ROOT/'tests/test_contact_resolution.py']
    plan=dict(scope='Resolution-only development of two previously failed curved-rigid/soft cases; no general contact/friction/dynamics claim',cases=list(CASES),meshes=[3,4,5],surface_orders=[16,32],penalty_N_m3=1e7,
              geometry_loading='Same block, supports, sphereR.045 prescribedz .09→.080 in5steps; known gravity−9.81 onlycombined, floorz−.04001; reset everyjob',
              gates=prior['plan']['gates'],comparisons='n3→4 and4→5 atq32; q16→32 atallthree meshes. Acceptspecificcaseonlyif at leastone meshpair passes and BOTHmembers passquadraturegates. Retainallfailedpairs.',
              additional_force_gate='Each tool/collider forcevector also meets .005N+1% independently so a large floor force cannot mask sphere error.',diagnostics='Per-collider contactarea/force, sampled and independent65/129face-grid penetration; area descriptive, not a new tuned acceptancegate.',
              dense_check='65→129 maxpenetrationdifference <= original20µm depthgate on every equilibrium',
              tiny_patch='Static originalmissed radius31mm/center30mm below flatface; n1q16/32/64 versus analyticforce/area/energy. Requireq32/64same originalforce/depthgates andq64analyticforcewithin.005N+1%,energyrelative1%,denseexactdepthwithin20µm; areareported separately. No collider/framechange.',
              workers=2,seconds=3600,device='cpu',source_sha256={str(p.relative_to(ROOT)):sha(p)for p in files},input_sha256={str(p.relative_to(ROOT)):sha(p)for p in inputs})
    write(PLAN,plan);print(PLAN)


def verify(plan):
    for mapping in (plan['source_sha256'],plan['input_sha256']):
        if any(sha(ROOT/p)!=h for p,h in mapping.items()):raise ValueError('Frozen refinement input/source changed')


def job(task):
    torch.set_num_threads(1);case,n,order=task;plan=json.loads(PLAN.read_text());verify(plan);start=time.monotonic();key=f'{case}-n{n}-q{order}';row=dict(key=key,case=case,n=n,order=order,plan_sha256=sha(PLAN))
    try:
        colliders=([RigidPlane(point=(0,0,-.04001))]if 'gravity' in case else [])+[RigidSphere(center=(.12,0,.09),radius=.045)]
        m=ResolvedContactQ2(n=n,surface_order=order,colliders=colliders,gravity=(0,0,-9.81)if 'gravity' in case else (0,0,0))
        if n==3 and order==16:
            source=OLD/f'{case}-n3-q16-k1e+07.npz';u=np.load(source)['u'];state=m.evaluate(u);state['displacement_local_m']=u;row['reused_sha256']=sha(source)
        else:
            state=None
            for z in (.09,.0875,.085,.0825,.080):
                m.colliders=tuple([c for c in m.colliders if isinstance(c,RigidPlane)]+[RigidSphere(center=(.12,0,z),radius=.045)])
                state=m.solve(initial=None if state is None else state['displacement_local_m'],force_tolerance_N=1e-8);u=state['displacement_local_m']
        # Reused state also has the identical final prescribed collider pose.
        m.colliders=tuple([c for c in m.colliders if isinstance(c,RigidPlane)]+[RigidSphere(center=(.12,0,.080),radius=.045)])
        state=m.evaluate(u);state['displacement_local_m']=u;w=m.reaction_wrenches(state)
        dense65=dense_surface_gap_check(m,u,65);dense129=dense_surface_gap_check(m,u,129);q=np.stack(np.meshgrid(*[np.linspace(0,1,9)]*3,indexing='ij'),-1).reshape(-1,3)
        reaction=np.concatenate([w['forces_N'].ravel(),*[c['force_on_rigid_N']for c in w['contact_reactions']]])
        np.savez_compressed(OUT/(key+'.npz'),u=u,field=m.sample_material(u,q),reaction=reaction)
        depth=max(d['max_dense_penetration_m']for d in dense129);change=max(abs(a['max_dense_penetration_m']-b['max_dense_penetration_m'])for a,b in zip(dense65,dense129))
        row.update(completed=True,residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),force_balance_N=w['force_balance_N'],torque_balance_Nm=w['world_torque_balance_Nm'],dense65=dense65,dense129=dense129,dense_refinement_change_m=change,max_dense_penetration_m=depth,contact_area=sampled_contact_areas(m,u),collider_forces_N=[x['force_on_rigid_N'].tolist()for x in w['contact_reactions']],response_sha256=sha(OUT/(key+'.npz')))
        row['equilibrium_gate']=row['residual_N']<1e-8 and row['minimum_J']>.1 and row['force_balance_N']<1e-7 and row['torque_balance_Nm']<1e-7 and change<=2e-5 and not any(d['missed_contact']for d in dense129)
    except Exception as error:row.update(completed=False,error=f'{type(error).__name__}: {error}',equilibrium_gate=False)
    row['seconds']=time.monotonic()-start;write(OUT/(key+'.json'),row);return row


def run():
    plan=json.loads(PLAN.read_text());verify(plan);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[];report=dict(plan_sha256=sha(PLAN),rows=rows,comparisons=[],case_qualified={},complete=False,expected_jobs=12,terminal_jobs=0,completed_jobs=0)
    pool=multiprocessing.get_context('spawn').Pool(2)
    try:
        tasks=[(case,n,q)for case in CASES for n in (3,4,5)for q in (16,32)];pending=pool.imap_unordered(job,tasks)
        for _ in tasks:
            rows.append(pending.next(timeout=max(.01,3600-(time.monotonic()-start))));report['terminal_jobs']=len(rows);report['completed_jobs']=sum(r['completed']for r in rows);write(R/'contact-resolution-refinement-results.json',report)
        for case in CASES:
            def find(n,q):return next(r for r in rows if (r['case'],r['n'],r['order'])==(case,n,q))
            checks={}
            for label,a,b in [(f'q{n}',find(n,16),find(n,32))for n in (3,4,5)]+[('mesh3_4',find(3,32),find(4,32)),('mesh4_5',find(4,32),find(5,32))]:
                c=dict(case=case,label=label,passed=False)
                if a['completed']and b['completed']:
                    x=np.load(OUT/(a['key']+'.npz'));y=np.load(OUT/(b['key']+'.npz'));field=float(np.sqrt(np.mean((x['field']-y['field'])**2)));force=float(np.linalg.norm(x['reaction']-y['reaction']));depth=abs(a['max_dense_penetration_m']-b['max_dense_penetration_m']);tolerance=.005+.01*np.linalg.norm(y['reaction'])
                    perbody_error=np.linalg.norm((x['reaction']-y['reaction']).reshape(-1,3),axis=1);perbody_tolerance=.005+.01*np.linalg.norm(y['reaction'].reshape(-1,3),axis=1)
                    c.update(per_body_force_difference_N=perbody_error.tolist(),per_body_force_tolerance_N=perbody_tolerance.tolist(),field_rms_m=field,reaction_difference_N=force,reaction_tolerance_N=float(tolerance),depth_difference_m=depth,passed=bool(a['equilibrium_gate']and b['equilibrium_gate']and field<=2e-5 and force<=tolerance and np.all(perbody_error<=perbody_tolerance) and depth<=2e-5))
                checks[label]=c['passed'];report['comparisons'].append(c)
            report['case_qualified'][case]=bool((checks['mesh3_4']and checks['q3']and checks['q4'])or(checks['mesh4_5']and checks['q4']and checks['q5']))
        tiny=[];exact=exact_flat_sphere_patch()
        for order in (16,32,64):
            m=ResolvedContactQ2(n=1,surface_order=order,colliders=(RigidSphere(center=(.12,0,-.07),radius=.031),));u=np.zeros_like(m.rest);U,_,_,c=m.contact_response(m.engine.tensor(u));d=dense_surface_gap_check(m,u,65)[0]
            tiny.append(dict(order=order,force_N=c[0]['force_on_rigid_N'].tolist(),energy_J=float(U),area=sampled_contact_areas(m,u)[0],dense=d))
        a,b=tiny[-2:];report['tiny_patch']=dict(exact=exact,rows=tiny,qualified=bool(np.linalg.norm(np.array(a['force_N'])-b['force_N'])<=.005+.01*abs(exact['force_normal_N']) and abs(b['force_N'][2]-exact['force_normal_N'])<=.005+.01*abs(exact['force_normal_N']) and abs(b['energy_J']/exact['energy_J']-1)<=.01 and abs(a['dense']['max_dense_penetration_m']-b['dense']['max_dense_penetration_m'])<=2e-5 and abs(b['dense']['max_dense_penetration_m']-exact['max_penetration_m'])<=2e-5))
        report['complete']=True
    except Exception as error:report['failure']=f'{type(error).__name__}: {error}'
    finally:pool.terminate();pool.join();verify(plan);report['elapsed_seconds']=time.monotonic()-start;report['missing_job_count']=12-len(rows);write(R/'contact-resolution-refinement-results.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args();prepare()if a.prepare else run()
