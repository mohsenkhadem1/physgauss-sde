"""Second bounded contact ladder; failed n3/4/5 evidence is immutable."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,importlib.util,json,time,multiprocessing,sys
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.contact_resolution import ResolvedContactQ2,exact_flat_sphere_patch,sampled_contact_areas
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane,RigidSphere
from scripts.compositional_neural_mechanics.contact_gap_search import search_gaps
from scripts.compositional_neural_mechanics.conditional_inference_pilot import sha,write
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';OUT=ROOT/'runs/compositional-neural-mechanics-v16/contact-fine-refinement';PLAN=R/'contact-fine-refinement-plan.json';CASES=('sphere_indentation','gravity_plane_sphere_indentation')


def prepare():
    if PLAN.exists()or OUT.exists():raise ValueError('Preserve previousrun')
    previous=R/'contact-resolution-refinement-results.json';old=json.loads(previous.read_text());oldplan=R/'contact-resolution-refinement-plan.json';p=json.loads(oldplan.read_text())
    if not old['complete']or any(old['case_qualified'].values()):raise ValueError('Expected completed failed development evidence')
    sources=closure([Path(__file__)])+[ROOT/'scripts/compositional_neural_mechanics/contact_refinement.py',ROOT/'tests/test_contact_gap_search.py']
    plan=dict(scope='Same two existingcurved contactcases, resolution-only follow-up; no friction/dynamics/generalcontact claim',cases=list(CASES),meshes=[6,8],optional_mesh=10,surface_orders=[16,32],
              acceptance='Latestmeshpair6/8, or8/10 ifprospectivefallbackruns, mustpass unchangedfield20µm/reaction.005N+1%perbody/global/depth20µm gates; BOTHpairmembers requireq16/32pass. Do notselectearlierpassingpair aftercontradictoryfinerevidence.',
              gap='65/129 denseface grids eachseed8spatiallyseparated localanalytic-gradient minimizations perface; takebestincludinggrid. Require65/129searchdepthdifference<=20µm and129denseunderestimate<=20µm; no certifiedglobalminimumclaim.',
              tiny='Same staticpatchq64/128; analyticforcewithin.005N+1%,energyrelative1%,area2%,depth20µm; q64/128force anddepthchanges samegates, areachange2%.',
              gates=p['gates'],optional_rule='Ifany6/8casefails and measuredcost permitswithinwhole3600s, execute n10q16/32 ONLYfailingcases; nofurthermesh.',workers=2,seconds=3600,device='cpu',
              source_sha256={str(f.relative_to(ROOT)):sha(f)for f in sources},input_sha256={str(f.relative_to(ROOT)):sha(f)for f in (previous,oldplan)})
    write(PLAN,plan);print(PLAN)


def verify(p):
    if any(sha(ROOT/f)!=h for table in ('source_sha256','input_sha256')for f,h in p[table].items()):raise ValueError('Pinned finecontactsource/input changed')


def job(task):
    torch.set_num_threads(1);start=time.monotonic();p=json.loads(PLAN.read_text());verify(p)
    # A private instance reuses exactly the prior case builder/solver, without changing its file or output namespace.
    spec=importlib.util.spec_from_file_location('_fine_contact_private',ROOT/'scripts/compositional_neural_mechanics/contact_refinement.py');base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base);base.OUT=OUT;base.PLAN=PLAN
    row=base.job(task);case,n,order=task
    if row['completed']:
        try:
            colliders=([RigidPlane(point=(0,0,-.04001))]if 'gravity'in case else [])+[RigidSphere(center=(.12,0,.080),radius=.045)]
            m=ResolvedContactQ2(n=n,surface_order=order,colliders=colliders,gravity=(0,0,-9.81)if 'gravity'in case else (0,0,0));u=np.load(OUT/(row['key']+'.npz'))['u']
            a=search_gaps(m,u,65);b=search_gaps(m,u,129);delta=max(abs(x['max_penetration_m']-y['max_penetration_m'])for x,y in zip(a,b));under=max(x['dense_underestimate_m']for x in b)
            row.update(search65=a,search129=b,continuous_depth_m=max(x['max_penetration_m']for x in b),continuous_search_change_m=delta,dense129_underestimate_m=under)
            row['equilibrium_gate']=row['residual_N']<1e-8 and row['minimum_J']>.1 and row['force_balance_N']<1e-7 and row['torque_balance_Nm']<1e-7 and delta<=2e-5 and under<=2e-5 and not any(d['missed_contact']for d in row['dense129'])
        except Exception as error:row.update(completed=False,equilibrium_gate=False,error=f'Gapsearch {type(error).__name__}: {error}')
    row['seconds']=time.monotonic()-start;write(OUT/(row['key']+'.json'),row);return row


def comparison(a,b,label):
    c=dict(case=a['case'],label=label,passed=False)
    if a['completed']and b['completed']:
        x=np.load(OUT/(a['key']+'.npz'));y=np.load(OUT/(b['key']+'.npz'));field=float(np.sqrt(np.mean((x['field']-y['field'])**2)));force=float(np.linalg.norm(x['reaction']-y['reaction']));per=np.linalg.norm((x['reaction']-y['reaction']).reshape(-1,3),axis=1);tols=.005+.01*np.linalg.norm(y['reaction'].reshape(-1,3),axis=1);tol=.005+.01*np.linalg.norm(y['reaction']);depth=abs(a['continuous_depth_m']-b['continuous_depth_m'])
        c.update(field_rms_m=field,reaction_difference_N=force,reaction_tolerance_N=float(tol),per_body_force_difference_N=per.tolist(),per_body_force_tolerance_N=tols.tolist(),continuous_depth_difference_m=depth,passed=bool(a['equilibrium_gate']and b['equilibrium_gate']and field<=2e-5 and force<=tol and np.all(per<=tols)and depth<=2e-5))
    return c


def run():
    p=json.loads(PLAN.read_text());verify(p);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[];report=dict(plan_sha256=sha(PLAN),rows=rows,comparisons=[],case_qualified={},complete=False,expected_jobs=8)
    pool=multiprocessing.get_context('spawn').Pool(2)
    def save():report['elapsed_seconds']=time.monotonic()-start;report['terminal_jobs']=len(rows);report['completed_jobs']=sum(x['completed']for x in rows);write(R/'contact-fine-refinement-results.json',report)
    def batch(tasks):
        pending=pool.imap_unordered(job,tasks)
        for _ in tasks:rows.append(pending.next(timeout=max(.01,p['seconds']-(time.monotonic()-start))));save()
    def casecheck(case,lo,hi):
        def find(n,q):return next(x for x in rows if(x['case'],x['n'],x['order'])==(case,n,q))
        c=[comparison(find(n,16),find(n,32),f'q{n}')for n in (lo,hi)]+[comparison(find(lo,32),find(hi,32),f'mesh{lo}_{hi}')];report['comparisons'].extend(c);return all(x['passed']for x in c)
    try:
        batch([(c,n,q)for c in CASES for n in (6,8)for q in (16,32)])
        failed=[]
        for c in CASES:
            report['case_qualified'][c]=casecheck(c,6,8)
            if not report['case_qualified'][c]:failed.append(c)
        save()
        if failed:
            times=[x['seconds']for x in rows if x['n']==8];estimated=np.mean(times)*(10/8)**6*len(failed)
            report['fallback_estimated_seconds']=float(estimated)
            if p['seconds']-(time.monotonic()-start)>estimated*1.15+60:
                report['expected_jobs']+=2*len(failed);batch([(c,10,q)for c in failed for q in (16,32)])
                for c in failed:report['case_qualified'][c]=casecheck(c,8,10)
            else:report['fallback_not_run']='Measuredremainingbudget insufficient; unresolvedcases retainfalse.'
        exact=exact_flat_sphere_patch();tiny=[]
        for q in (64,128):
            m=ResolvedContactQ2(n=1,surface_order=64,colliders=(RigidSphere(center=(.12,0,-.07),radius=.031),))
            if q==128:m.contact_patches=m._surface_quadrature(128);m.surface_order=128
            u=np.zeros_like(m.rest);energy,_,_,c=m.contact_response(m.engine.tensor(u));depth=search_gaps(m,u,65,seeds_per_face=2)[0]
            tiny.append(dict(order=q,force_N=c[0]['force_on_rigid_N'].tolist(),energy_J=float(energy),area_m2=sampled_contact_areas(m,u)[0]['active_reference_area_m2'],gap=depth))
        a,b=tiny;ft=.005+.01*abs(exact['force_normal_N']);report['tiny_patch']=dict(exact=exact,rows=tiny,qualified=bool(np.linalg.norm(np.array(a['force_N'])-b['force_N'])<=ft and abs(b['force_N'][2]-exact['force_normal_N'])<=ft and abs(b['energy_J']/exact['energy_J']-1)<=.01 and abs(b['area_m2']/exact['area_m2']-1)<=.02 and abs(b['area_m2']-a['area_m2'])/exact['area_m2']<=.02 and abs(b['gap']['max_penetration_m']-exact['max_penetration_m'])<=2e-5 and abs(a['gap']['max_penetration_m']-b['gap']['max_penetration_m'])<=2e-5))
        report['complete']=True
    except Exception as error:report['failure']=f'{type(error).__name__}: {error}'
    finally:pool.terminate();pool.join();verify(p);report['missing_jobs']=report['expected_jobs']-len(rows);save()

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=ap.parse_args();prepare()if a.prepare else run()
