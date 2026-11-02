"""Bounded development reference posterior; prepare/audit before --run."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,time,multiprocessing,signal
from pathlib import Path
import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import cholesky
from scripts.compositional_neural_mechanics.conditional_inference_pilot import problem,predict,sha,write,OUT as ORIGINAL
from scripts.compositional_neural_mechanics.joint_inference import neutral_parameters
from scripts.compositional_neural_mechanics.gauss_newton_inference import GaussNewtonMAP
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
from scripts.compositional_neural_mechanics.reference_posterior_recovery_math import integrate,compare,scale_bounds
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';OLD=ROOT/'runs/compositional-neural-mechanics-v16/reference-posterior';OUT=ROOT/'runs/compositional-neural-mechanics-v16/reference-posterior-recovery';PLAN=R/'reference-posterior-recovery-plan.json'

def observations():
    with np.load(ORIGINAL/'observations-and-development-truth.npz')as raw:
        return [{k:raw[f'obs_{i}_{k}'].copy()for k in ('pose','points','displacement','forces','torques')}for i in range(2)]

def verify(plan):
    for key in ('source_sha256','input_sha256'):
        if any(sha(ROOT/p)!=h for p,h in plan[key].items()):raise ValueError('Source/input changed')

def prepare():
    if PLAN.exists() or OUT.exists():raise ValueError('Preserve previous study')
    inputs=[ORIGINAL/'observations-and-development-truth.npz',R/'conditional-inference-pilot-plan.json',R/'conditional-inference-pilot-results.json']
    inputs += [OLD/'proposal.json',R/'reference-posterior-plan.json',R/'reference-posterior-results.json']+sorted((OLD/'nodes').glob('*.json'))+sorted((OLD/'nodes').glob('*.npz'))
    sources=closure([Path(__file__)])+[ROOT/'tests/test_reference_posterior_recovery.py']
    plan=dict(scope='DEVELOPMENT small three-parameter analytic reference posterior, not full15 or global identifiability',active=['s=logEscale','logK_left','logK_right'],known_nuisance='Exact original conditional pilot neutral nuisances; geometry/categories known; zero gravity',
              prior='Original independent unit Gaussian means0,ln500,ln500 truncated to original bounds s[-1,1],K[20,10000]; transform r=logK-s has determinant1',
              observations='Only original two observed poses/positions and optional forces+torques; no withheld reads',tracks=['V','VF'],
              forward='At s0 solve two poses for each ratio pair; exact energy co-scaling gives identicalu and exp(s) wrenches',
              proposal='Original analyticV MAP ratios; inverse(positionJ_r^T J_r + [[2,-1],[-1,2]]/3); r-space Gaussian importance density explicitly divided out',
              orders=[9,13,17],proposal_std_factors=[1,2],proposal_check_order=17,scale_points=[1025,2049],
              gates=dict(mean_sd=.02,interval_endpoint_sd=.05,covariance_relative=.02,adaptive_conditional_normalizer_relative_at_2049=1e-6),
              optional_refinement='If13/17 or17std2 check fails and budget permits, one fixed25/33 pair at standard proposal plus33std2; same gates. No other proposals/tolerances.',
              failed_nodes='Every positive-prior forward node retained. Any failed node makes corresponding posterior/gate unavailable; no drop-and-renormalize.',
              conditional_scale='Unchanged uniform Simpson moments/masses1025/2049; positive adaptive exact-density conditional CDF with stationary-point splits, Vclosed truncatednormal;2049normalizer must agreewithin1e-6. OriginalCDFfailure preserved.',recovery='Only scalarCDF numerical scheme changed; exact original proposal and81 cached forwardnodes reused withbyte hashes; failed originalplan/results untouched',
              seconds=3375,workers=2,device='cpu',source_sha256={str(p.relative_to(ROOT)):sha(p)for p in sources},input_sha256={str(p.relative_to(ROOT)):sha(p)for p in inputs})
    write(PLAN,plan);print(PLAN)

def initializer():
    global OBS,MODEL
    torch.set_num_threads(1);OBS=observations();MODEL=problem([{k:v for k,v in o.items()if k not in ('forces','torques')}for o in OBS],False)

def node_job(job):
    ratio,key=job;path=OUT/'nodes'/f'{key}.json';started=time.monotonic()
    if path.exists():return json.loads(path.read_text())
    row=dict(key=key,ratio=ratio,plan_sha256=sha(PLAN))
    oldpath=OLD/'nodes'/f'{key}.json'
    if oldpath.exists():
        plan=json.loads(PLAN.read_text());old=json.loads(oldpath.read_text())
        if sha(oldpath)!=plan['input_sha256'][str(oldpath.relative_to(ROOT))] or old['ratio']!=ratio:raise ValueError('Cached ratio/source mismatch')
        row.update(old);row['plan_sha256']=sha(PLAN);row['reused_from']=dict(path=str(oldpath.relative_to(ROOT)),sha256=sha(oldpath),original_plan_sha256=old['plan_sha256'])
        if old['completed']:
            source=oldpath.with_suffix('.npz')
            if sha(source)!=plan['input_sha256'][str(source.relative_to(ROOT))] or sha(source)!=old['response_sha256']:raise ValueError('Cached response mismatch')
            (OUT/'nodes'/source.name).write_bytes(source.read_bytes())
        write(path,row);return row
    try:
        z=neutral_parameters();z[[0,3,5]]=[0.,*ratio];pred=[predict(MODEL,z,o['pose'])for o in OBS]
        pos=np.concatenate([(p['displacement']-o['displacement']).ravel()/50e-6 for p,o in zip(pred,OBS)])
        b=np.concatenate([p[k].ravel()/sigma for p in pred for k,sigma in [('forces',.02),('torques',.001)]])
        y=np.concatenate([o[k].ravel()/sigma for o in OBS for k,sigma in [('forces',.02),('torques',.001)]])
        if not all(np.isfinite(v).all() for p in pred for v in p.values()):raise ValueError('Nonfinite forward node')
        np.savez_compressed(OUT/'nodes'/f'{key}.npz',**{f'{i}_{k}':v for i,p in enumerate(pred)for k,v in p.items()})
        row.update(completed=True,position_loglikelihood=float(-.5*pos@pos-len(pos)*np.log(50e-6*np.sqrt(2*np.pi))),wrench_abc=[float(b@b),float(b@y),float(y@y)],response_sha256=sha(OUT/'nodes'/f'{key}.npz'))
    except Exception as e:row.update(completed=False,error=f'{type(e).__name__}: {e}')
    row['seconds']=time.monotonic()-started;write(path,row);return row

def main_run():
    plan=json.loads(PLAN.read_text());verify(plan);OUT.mkdir(parents=True,exist_ok=False);(OUT/'nodes').mkdir();started=time.monotonic();deadline=started+plan['seconds'];torch.set_num_threads(1)
    previous=json.loads((OLD/'proposal.json').read_text());center=np.array(previous['center']);covariance=np.array(previous['covariance'])
    write(OUT/'proposal.json',dict(**{k:v for k,v in previous.items() if k!='plan_sha256'},plan_sha256=sha(PLAN),reused_proposal_sha256=sha(OLD/'proposal.json')))
    report=dict(plan_sha256=sha(PLAN),proposal_sha256=sha(OUT/'proposal.json'),levels={},checks={},failures=[],complete=False)
    ctx=multiprocessing.get_context('spawn');pool=ctx.Pool(2,initializer=initializer)
    def save():report['elapsed_seconds']=time.monotonic()-started;write(R/'reference-posterior-recovery-results.json',report)
    def level(order,factor):
        key=f'o{order}-std{factor}';q,w=hermgauss(order);a,b=np.meshgrid(q,q,indexing='ij');v=np.stack([a.ravel(),b.ravel()],1);L=cholesky(covariance*factor**2,lower=True);nodes=center+np.sqrt(2)*v@L.T
        loggh=(np.log(w[:,None]*w[None,:]/np.pi)).ravel();logdensity=-np.log(2*np.pi)-np.log(np.linalg.det(L))-.5*np.sum((np.sqrt(2)*v)**2,axis=1)
        support=np.array([scale_bounds(x)[1]>scale_bounds(x)[0]for x in nodes]);nodes=nodes[support];lw=(loggh-logdensity)[support]
        jobs=[(r.tolist(),hashlib.sha256(np.asarray(r,dtype='<f8').tobytes()).hexdigest())for r in nodes]
        pending=pool.imap_unordered(node_job,jobs);collected={}
        for _ in jobs:
            result=pending.next(timeout=max(.01,deadline-time.monotonic()));collected[result['key']]=result
        records=[collected[k]for _,k in jobs];failed=[x for x in records if not x['completed']]
        entry=dict(order=order,proposal_std_factor=factor,node_count=len(jobs),zero_prior_nodes=int((~support).sum()),failed_nodes=failed,node_keys=[k for _,k in jobs],posteriors={})
        if not failed:
            # Two poses, each6 force and6 torque components with original independent noise.
            constant=-12*np.log(.02*np.sqrt(2*np.pi))-12*np.log(.001*np.sqrt(2*np.pi))
            for track in ('V','VF'):
                entry['posteriors'][track]={str(points):integrate(nodes,lw,[x['position_loglikelihood']for x in records],[x['wrench_abc']for x in records],constant,track,points)for points in plan['scale_points']}
        report['levels'][key]=entry;save();return entry
    def checks(lo,hi):
        out={}
        for track in ('V','VF'):
            for label,a,b in [(f'order{lo}_{hi}',f'o{lo}-std1',f'o{hi}-std1'),('proposal_std_doubling',f'o{hi}-std1',f'o{hi}-std2')]:
                A=report['levels'][a]['posteriors'];B=report['levels'][b]['posteriors'];out[track+'_'+label]=compare(A[track]['2049'],B[track]['2049']) if A and B else dict(passed=False,reason='Failed forward node')
            for key in (f'o{lo}-std1',f'o{hi}-std1',f'o{hi}-std2'):
                d=report['levels'][key]['posteriors'];out[track+'_scale_'+key]=compare(d[track]['1025'],d[track]['2049']) if d else dict(passed=False,reason='Failed forward node')
        for key in (f'o{lo}-std1',f'o{hi}-std1',f'o{hi}-std2'):
            for track in ('V','VF'):
                data=report['levels'][key]['posteriors']
                out[track+'_adaptive_normalizer_'+key]=dict(passed=bool(data and data[track]['2049']['adaptive_normalizer_gate']),relative_error=data[track]['2049']['adaptive_normalizer_max_relative_difference'] if data else None)
        return out
    try:
        for order in (9,13,17):level(order,1)
        level(17,2);report['checks']['initial']=checks(13,17);save()
        if not all(c['passed']for c in report['checks']['initial'].values()):
            # Predict cost from measured node throughput; never exceed the hard stage deadline.
            measured=[json.loads(p.read_text())['seconds']for p in (OUT/'nodes').glob('*.json')];estimated=(25**2+2*33**2)*np.mean(measured)/2
            report['refinement_estimated_seconds']=float(estimated)
            if deadline-time.monotonic()>estimated*1.15+30:
                for order,factor in ((25,1),(33,1),(33,2)):level(order,factor)
                report['checks']['refined']=checks(25,33)
            else:report['refinement_not_run']='Remaining hard budget below measured cost estimate; no tolerance relaxation'
        report['qualified']=all(c['passed']for c in report['checks'].get('refined',report['checks']['initial']).values());report['complete']=True
    except Exception as error:report['failures'].append(f'{type(error).__name__}: {error}');report['qualified']=False
    finally:
        pool.terminate();pool.join();report['source_and_inputs_unchanged']=all(sha(ROOT/p)==h for table in ('source_sha256','input_sha256')for p,h in plan[table].items());save()
    print('completed',report['complete'],'qualified',report['qualified'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args();prepare()if a.prepare else main_run()
