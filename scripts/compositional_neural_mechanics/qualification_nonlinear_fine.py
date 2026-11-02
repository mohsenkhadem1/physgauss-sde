"""Bounded6→8 follow-up of failed actual-range mesh gates, including crossed case."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import concurrent.futures,json,queue,signal,subprocess,sys,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.compositional_neural_mechanics.scenes import make_model,poses
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.qualification import ROOT,sha,write,nonrigid
from scripts.compositional_neural_mechanics.qualification_nonlinear_range import stats
OUT=ROOT/'runs/compositional-neural-mechanics-v16/nonlinear-range-fine8'


def guard(plan):
    if any(sha(ROOT/p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('Actual-range fine-stage source/evidence changed')


def worker(i,device):
    plan=json.loads((OUT/'plan.json').read_text());guard(plan);remaining=int(plan['deadline_unix']-time.time())
    if remaining<=0:raise TimeoutError('Stage expired')
    signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Fine stage cap')));signal.alarm(remaining)
    job=plan['jobs'][i];scene=job['scene'];action=job['action'];row=dict(index=i,case=scene['name'],action=action,n=8,device=device);started=time.monotonic()
    try:
        path=ROOT/job['reference']['artifact']
        if sha(path)!=job['reference']['sha256']:raise ValueError('n6reference changed')
        with np.load(path) as data:prior={k:data[k].copy() for k in data.files}
        coarse=make_model(scene,n=6,device=device);fine=make_model(scene,n=8,device=device);geometry=GeometrySpec(**scene['geometry']);pose=poses(geometry)[action]
        fine.set_controls(pose);initial=coarse.sample_material(prior['displacement'],fine.material_coordinates);state=fine.solve(initial=initial,force_tolerance_N=1e-8)
        q=np.stack(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij'),-1).reshape(-1,3);field=fine.sample_material(state['displacement_local_m'],q);w=fine.reaction_wrenches(state)
        result_path=OUT/f'job-{i}.npz';np.savez_compressed(result_path,field=field,shape=nonrigid(geometry.map(q),field),forces_N=w['forces_N'],torques_Nm=w['torques_Nm'],rest=fine.rest,displacement=state['displacement_local_m'])
        delta=float(np.sqrt(np.mean((field-prior['field'])**2)));force=np.linalg.norm(w['forces_N']-prior['forces_N'],axis=1);torque=np.linalg.norm(w['torques_Nm']-prior['torques_Nm'],axis=1);g=plan['gates'];ft=g['force_absolute_N']+g['force_relative']*np.linalg.norm(w['forces_N'],axis=1);tt=g['torque_absolute_Nm']+g['torque_relative']*np.linalg.norm(w['torques_Nm'],axis=1)
        row.update(state='complete',passed=bool(delta<=g['field_rms_m'] and (force<=ft).all() and (torque<=tt).all()),field_rms_mm=1000*delta,force_difference_N=force.tolist(),torque_difference_Nm=torque.tolist(),force_limits_N=ft.tolist(),torque_limits_Nm=tt.tolist(),
            residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),force_balance_N=w['force_balance_N'],nonrigid_rms_mm=float(1000*np.sqrt(np.mean(nonrigid(geometry.map(q),field)**2))),diagnostics=stats(fine,state['displacement_local_m']),artifact=str(result_path.relative_to(ROOT)),sha256=sha(result_path),reference_sha256=sha(path))
        guard(plan)
    except Exception as error:row.update(state='failed',passed=False,error=repr(error))
    row['elapsed_seconds']=time.monotonic()-started;write(OUT/f'job-{i}.json',row);print({k:v for k,v in row.items() if k!='diagnostics'},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=int);p.add_argument('--device');a=p.parse_args()
    if a.job is not None:
        if a.device not in ('cuda:1','cuda:2') or a.job not in range(len(json.loads((OUT/'plan.json').read_text())['jobs'])):raise ValueError('Declared job/GPU1–2 required')
        return worker(a.job,a.device)
    if OUT.exists():raise ValueError('Preserve earlier fine-stage results')
    roots=[ROOT/'runs/compositional-neural-mechanics-v16'/name for name in ('nonlinear-range-development','crossed-soft-saturated-development')]
    jobs=[];sources={};gates=None
    for root in roots:
        report=json.loads((root/'report.json').read_text());manifest=json.loads((root/'manifest.json').read_text());gates=manifest['development_plan']['gates']
        for file in ('report.json','manifest.json'):sources[str((root/file).relative_to(ROOT))]=sha(root/file)
        for check in report['comparisons']:
            if check['comparison']!='mesh4to6' or check['passed']:continue
            name=check.get('case',manifest['development_plan']['cases'][0]['name']);scene=next(s for s in manifest['development_plan']['cases'] if s['name']==name)
            reference=next(r for r in report['rows'] if r['case']==name and r['action']==check['action'] and r['model']=='teacher' and r['n']==6)
            if reference['state']!='complete':raise ValueError('No completed6mesh state for targeted comparison')
            if sha(ROOT/reference['artifact'])!=reference['sha256']:raise ValueError('Prior state hash mismatch')
            sources[reference['artifact']]=reference['sha256'];jobs.append(dict(scene=scene,action=check['action'],reference=reference,previous_failed_check=check))
    paths=[Path(__file__),*[ROOT/'scripts/compositional_neural_mechanics'/f'{x}.py' for x in ('mechanics','scenes','interfaces','materials','qualification','qualification_nonlinear_range')],ROOT/'scripts/phystwin/tensor_fixture.py',ROOT/'scripts/phystwin/consistent_fixture.py']
    sources.update({str(path.relative_to(ROOT)):sha(path) for path in paths});plan=dict(scope=__doc__,source_sha256=sources,jobs=jobs,gates=gates,deadline_unix=time.time()+1740,workers=2,devices=['cuda:1','cuda:2'],
         initial='Same-actionn6equilibrium interpolated inmaterialcoordinates; independent actualpose',maximum_mesh=8,stage_budget_seconds=1740)
    write(OUT/'plan.json',plan);pending=queue.Queue()
    for i in range(len(jobs)):pending.put(i)
    def runner(device):
        while True:
            try:i=pending.get_nowait()
            except queue.Empty:return
            remaining=plan['deadline_unix']-time.time()
            if remaining<=0:write(OUT/f'job-{i}.json',dict(index=i,case=jobs[i]['scene']['name'],action=jobs[i]['action'],state='failed',passed=False,error='Stage expired before launch'));continue
            with (OUT/f'job-{i}.log').open('w') as stream:
                child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--job',str(i),'--device',device],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            if not (OUT/f'job-{i}.json').exists():write(OUT/f'job-{i}.json',dict(index=i,case=jobs[i]['scene']['name'],action=jobs[i]['action'],state='failed',passed=False,error='Worker failed before result',exit_code=code))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(runner,['cuda:1','cuda:2']))
    guard(plan);rows=[json.loads((OUT/f'job-{i}.json').read_text()) for i in range(len(jobs))];result=dict(scope=__doc__,source_sha256=sources,plan_sha256=sha(OUT/'plan.json'),rows=rows,all_gates_pass=all(r['passed'] for r in rows),
        limitations=['Only previously failing development checks targeted; earlierfailedresults preserved','No mesh beyond8 authorized inthisstage','No inherited quadraturequalification forstrongnonlinearrange'])
    write(OUT/'report.json',result);write(ROOT/'reports/compositional-neural-mechanics-v16/nonlinear-range-fine-qualification.json',result)
if __name__=='__main__':main()
