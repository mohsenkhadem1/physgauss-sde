"""Prospective development-only learned-law + rigid-contact composition control."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,subprocess,time,signal,multiprocessing,csv,io
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.contact_transfer_model import make_contact_model,set_indenter,FAMILIES
from scripts.compositional_neural_mechanics.contact_gap_search import search_gaps
from scripts.compositional_neural_mechanics.contact_resolution import sampled_contact_areas
from scripts.compositional_neural_mechanics.conditional_inference_pilot import BULK,INTERFACE,sha,write
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';OUT=ROOT/'runs/compositional-neural-mechanics-v16/contact-transfer';PLAN=R/'contact-transfer-plan.json';OWNER='compositional-v16-oracle-missing-recovery.service'


def prepare():
    if PLAN.exists()or OUT.exists():raise ValueError('Preserveprevious study')
    files=closure([Path(__file__)])+[ROOT/'tests/test_contact_transfer_model.py',ROOT/'tests/test_contact_transfer_accounting.py',ROOT/'scripts/compositional_neural_mechanics/contact_transfer_render.py']
    plan=dict(scope='DEVELOPMENT oracle-descriptor transfer of frozen shared bulk/bilateralinterface laws into known analytic normalcontact; no retraining, no heldout/inferredparameterclaim',
              families={k:dict(bulk=v[0],bulk_latent=[0,0,v[1]],interface_latent=[float(np.log(6000.)),.5,v[2]])for k,v in FAMILIES.items()},methods=['analytic_teacher','shared_laws'],
              geometry=dict(dimensions=[.24,.08,.08],taper=0,arch=0),density=625,gravity=[0,0,-9.81],sphere_radius=.045,sphere_centers_z=[.09,.0875,.085,.0825,.080],floor_z=-.04001,penalty_N_m3=1e7,
              supports='Original normalized bilateralfootprints, center(.5,.1875),width(.375,.375),K6000N/m/side; nowexplicitnonlinearinterfaceenergy',
              preflight=dict(n=3,surface_order=16,device='cpu',workers=2,seconds=1200,per_call_seconds=600),fine=dict(meshes=[6,8],surface_orders=[16,32],preferred_devices=['cuda:0','cuda:1'],fallback_device='cpu',workers=2,seconds=7200,per_call_seconds=3600),
              gpu_gate='Before GPU mainstage, oracle-missing-recovery service inactive/failed, MainPID0 and its entire cgroup has no remainingprocesses. CPU fallback explicit CLI option. NoGPU3.',
              gates=dict(field_rms_m=2e-5,force_absolute_N=.005,force_relative=.01,depth_change_m=2e-5,residual_N=1e-8,force_balance_N=1e-7,torque_balance_Nm=1e-7,minimum_J=.1),
              qualification='Each new family ANDmethod needsitsown6/8meshcheck andbothmeshesq16/32check; per-bodyforces, continuousgap65/129+129densecorroboration. NoinheritanceofNH/linear-supportqualification.',
              comparisons='Matchedteacher/shared atsame mesh/q; n8teacher numericalreference onlyifqualified; field/contact/toolforces+torques/penetration, no arbitraryshared-error threshold',
              render='Bothfamilies final n8q32 teacher/shared physicalscale withfloor/sphere andfield-error map; allavailablecases shownwithqualificationlabels, missingcasesexplicit',
              source_sha256={str(p.relative_to(ROOT)):sha(p)for p in files},artifact_sha256={str(p.relative_to(ROOT)):sha(p)for p in (BULK,INTERFACE)})
    write(PLAN,plan);print(PLAN)


def verify(p):
    if any(sha(ROOT/f)!=h for t in ('source_sha256','artifact_sha256')for f,h in p[t].items()):raise ValueError('Frozen contacttransfer source/checkpoint changed')


def gpu_gate():
    text=subprocess.check_output(['systemctl','--user','show',OWNER,'-p','ActiveState','-p','MainPID','-p','ControlGroup'],text=True);d=dict(x.split('=',1)for x in text.strip().splitlines())
    if d.get('ActiveState')in ('active','activating','deactivating')or d.get('MainPID')!='0':raise ValueError('Oracle recovery still owns GPUs')
    group=d.get('ControlGroup');pids=[]
    if group:
        path=Path('/sys/fs/cgroup')/group.lstrip('/')
        if path.exists():
            for file in path.rglob('cgroup.procs'):pids.extend(file.read_text().split())
    if pids:raise ValueError('Oracle recovery cgroup retains workerPIDs')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[dict(index=int(a.strip()),uuid=b.strip(),total_MiB=int(c.strip()),free_MiB=int(f.strip()),utilization_percent=int(u.strip()))for a,b,c,f,u in csv.reader(io.StringIO(raw))]
    rawapps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    apps=[dict(uuid=a.strip(),pid=b.strip(),process=c.strip(),used_MiB=f.strip())for a,b,c,f in csv.reader(io.StringIO(rawapps))if a.strip()]
    selected=[g for g in gpus if g['index']in (0,1)];uuids={g['uuid']for g in selected};census=dict(gpus=gpus,compute_apps=apps,minimum_free_MiB=8192,maximum_initial_utilization_percent=10)
    if len(selected)!=2 or any(g['free_MiB']<8192 or g['utilization_percent']>10 for g in selected)or any(a['uuid']in uuids for a in apps):raise ValueError('GPU headroom/unrelatedjob census rejected: '+json.dumps(census))
    d['census']=census
    return d


def job(task):
    family,method,n,q,device=task;torch.set_num_threads(1);p=json.loads(PLAN.read_text());verify(p);key=f'{family}-{method}-n{n}-q{q}';folder=OUT/'states';folder.mkdir(parents=True,exist_ok=True);start=time.monotonic();row=dict(key=key,family=family,method=method,n=n,q=q,device=device,plan_sha256=sha(PLAN))
    def alarm(*args):raise TimeoutError('Contacttransfer callcap')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,p['preflight']['per_call_seconds']if n==3 else p['fine']['per_call_seconds'])
    try:
        m=make_contact_model(family,method,n,q,device);state=None;frames=[]
        for z in p['sphere_centers_z']:
            set_indenter(m,z);state=m.solve(initial=None if state is None else state['displacement_local_m'],force_tolerance_N=1e-8);frames.append(state['displacement_local_m'].copy())
        u=frames[-1];w=m.reaction_wrenches(state);a=search_gaps(m,u,65);b=search_gaps(m,u,129);gapchange=max(abs(x['max_penetration_m']-y['max_penetration_m'])for x,y in zip(a,b));under=max(x['dense_underestimate_m']for x in b)
        query=np.stack(np.meshgrid(*[np.linspace(0,1,9)]*3,indexing='ij'),-1).reshape(-1,3);force=np.stack([*w['forces_N'],*[x['force_on_rigid_N']for x in w['contact_reactions']]]);torque=np.stack([*w['torques_Nm'],*[x['torque_on_rigid_Nm']for x in w['contact_reactions']]])
        surface=[];uv=np.stack(np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,33),indexing='ij'),-1).reshape(-1,2)
        for axis in range(3):
            for side in (0.,1.):
                qs=np.empty((len(uv),3));qs[:,axis]=side;qs[:,[j for j in range(3)if j!=axis]]=uv;surface.append(qs)
        surface_q=np.concatenate(surface);surface_rest=m.geometry.map(surface_q).reshape(6,33,33,3)
        surface_frames=np.stack([(m.geometry.map(surface_q)+m.sample_material(displacement,surface_q)).reshape(6,33,33,3)for displacement in frames])
        np.savez_compressed(folder/(key+'.npz'),u=u,frames=np.stack(frames),field=m.sample_material(u,query),query=query,forces=force,torques=torque,surface_rest=surface_rest,surface_frames=surface_frames)
        missed=any(g['max_penetration_m']>1e-10 and c['active_quadrature_samples']==0 for g,c in zip(b,w['contact_reactions']))
        row.update(completed=True,response_sha256=sha(folder/(key+'.npz')),residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),force_balance_N=w['force_balance_N'],torque_balance_Nm=w['world_torque_balance_Nm'],search65=a,search129=b,depth_m=max(x['max_penetration_m']for x in b),gap_search_change_m=gapchange,dense_underestimate_m=under,contact_area=sampled_contact_areas(m,u),missed_contact=missed)
        row['numerical_gate']=row['residual_N']<1e-8 and row['minimum_J']>.1 and row['force_balance_N']<1e-7 and row['torque_balance_Nm']<1e-7 and gapchange<=2e-5 and under<=2e-5 and not missed
    except Exception as error:row.update(completed=False,numerical_gate=False,error=f'{type(error).__name__}: {error}')
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    row['seconds']=time.monotonic()-start;write(folder/(key+'.json'),row);return row


def family_worker(task):
    family,device=task;rows=[]
    for method in ('analytic_teacher','shared_laws'):
        for n in (6,8):
            for q in (16,32):rows.append(job((family,method,n,q,device)))
    return rows


def comparison(a,b):
    r=dict(a=a['key'],b=b['key'],passed=False)
    if a['completed']and b['completed']:
        x=np.load(OUT/'states'/(a['key']+'.npz'));y=np.load(OUT/'states'/(b['key']+'.npz'));field=float(np.sqrt(np.mean((x['field']-y['field'])**2)));force=np.linalg.norm(x['forces']-y['forces'],axis=1);tolerance=.005+.01*np.linalg.norm(y['forces'],axis=1);depth=abs(a['depth_m']-b['depth_m'])
        r.update(field_rms_m=field,per_body_force_difference_N=force.tolist(),per_body_force_tolerance_N=tolerance.tolist(),per_body_torque_difference_Nm=np.linalg.norm(x['torques']-y['torques'],axis=1).tolist(),depth_difference_m=depth,passed=bool(a['numerical_gate']and b['numerical_gate']and field<=2e-5 and (force<=tolerance).all()and depth<=2e-5))
    return r


def collect_declared(stage,root,plan_sha256):
    rows=[];missing=[]
    for family in FAMILIES:
        for method in ('analytic_teacher','shared_laws'):
            for n in ([3]if stage=='preflight'else[6,8]):
                for q in ([16]if stage=='preflight'else[16,32]):
                    key=f'{family}-{method}-n{n}-q{q}';path=Path(root)/'states'/(key+'.json')
                    if not path.exists():
                        missing.append(key);rows.append(dict(key=key,family=family,method=method,n=n,q=q,completed=False,numerical_gate=False,state='missing',error='No terminal state artifact at stage end'));continue
                    row=json.loads(path.read_text())
                    if row['plan_sha256']!=plan_sha256 or (row['key'],row['family'],row['method'],row['n'],row['q'])!=(key,family,method,n,q):raise ValueError('Declared state identity/provenance mismatch')
                    if row['completed'] and sha(path.with_suffix('.npz'))!=row['response_sha256']:raise ValueError('Changed completedstate response')
                    rows.append(row)
    return rows,missing


def analyse_fine(report,rows):
    report['qualification']={};report['transfer_comparisons']={}
    for family in FAMILIES:
        def find(method,n,q):return next(x for x in rows if(x['family'],x['method'],x['n'],x['q'])==(family,method,n,q))
        for method in ('analytic_teacher','shared_laws'):
            checks=[comparison(find(method,n,16),find(method,n,32))for n in (6,8)]+[comparison(find(method,6,32),find(method,8,32))]
            report['qualification'][family+'_'+method]=dict(passed=all(x['passed']for x in checks),checks=checks)
        transfer=comparison(find('shared_laws',8,32),find('analytic_teacher',8,32));transfer.pop('passed',None)
        transfer['note']='Descriptive known-latent transfererror only; no performanceacceptancecriterion.';report['transfer_comparisons'][family]=transfer


def execute(stage,device_choice):
    p=json.loads(PLAN.read_text());verify(p);OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();rows=[];result=R/f'contact-transfer-{stage}.json'
    if result.exists():raise ValueError('Preserve previousstage')
    ownership=None;requested_device=device_choice
    if stage=='preflight':tasks=[(family,method,3,16,'cpu')for family in FAMILIES for method in ('analytic_teacher','shared_laws')];fn=job;cap=p['preflight']['seconds']-30
    else:
        old=json.loads((R/'contact-transfer-preflight.json').read_text())
        if old['plan_sha256']!=sha(PLAN)or len(old['rows'])!=4 or not all(x['completed']for x in old['rows']):raise ValueError('Allfour numericalpreflightattempts mustcomplete beforefinestudy; noaccuracyselection')
        if device_choice=='gpu':
            try:ownership=gpu_gate()
            except (ValueError,subprocess.CalledProcessError)as error:
                ownership=dict(eligible=False,rejection=str(error),fallback='cpu');device_choice='cpu'
        tasks=[(family,('cuda:'+str(i))if device_choice=='gpu'else'cpu')for i,family in enumerate(FAMILIES)];fn=family_worker;cap=p['fine']['seconds']-30
    pool=multiprocessing.get_context('spawn').Pool(2);report=dict(plan_sha256=sha(PLAN),stage=stage,rows=rows,device_choice=device_choice,requested_device=requested_device,gpu_owner_release=ownership,complete=False)
    try:
        pending=pool.imap_unordered(fn,tasks)
        for _ in tasks:
            x=pending.next(timeout=max(.01,cap-(time.monotonic()-start)));rows.extend(x if isinstance(x,list)else[x]);write(result,report)
        report['complete']=True
    except Exception as error:report['failure']=f'{type(error).__name__}: {error}'
    finally:
        pool.terminate();pool.join();verify(p)
        rows,missing=collect_declared(stage,OUT,sha(PLAN));report.update(rows=rows,terminal=True,expected_jobs=len(rows),terminal_jobs=len(rows)-len(missing),completed_jobs=sum(x['completed']for x in rows),missing_keys=missing)
        if missing:report['complete']=False
        if stage=='fine':analyse_fine(report,rows)
        report['elapsed_seconds']=time.monotonic()-start;write(result,report)

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--stage',choices=['preflight','fine']);ap.add_argument('--device',choices=['cpu','gpu'],default='cpu');a=ap.parse_args();prepare()if a.prepare else execute(a.stage,a.device)
