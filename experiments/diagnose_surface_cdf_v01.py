"""Retrospective prefix-only CDF diagnosis; never requalifies the frozen run.

Read one archived prefix row and its independent prefix noise only. No heldout
truth, reset observation, parameter, policy choice or prediction score is read.
Raw CDFs/selected thresholds stay LOCAL; public output may redact scalar errors.
"""
import argparse,json,os,sys,time,zipfile,warnings
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.integrate import IntegrationWarning
from experiments import run_surface_p2_study_v03 as old
from experiments import benchmark_surface_amplitude_cpu_v01 as base
from experiments.campaign_surface_conditional_amplitude_v01 import ConditionalAmplitudeReference,RatioModal,ExactRatioMemo,amplitude_nodes,gauss_axis,BOUNDS,LOG_WIDTH
from experiments.campaign_amplitude_adaptive_oracle_v01 import adaptive_integral

ROOT=old.ROOT;OUT=ROOT/'runs/operator-dev/surface-cdf-diagnosis-v01'
INPUT=old.LOCAL/'object-02-LOCAL';PREFIX='normal_x180'
sha=old.sha;write=old.write


def sources():
    paths=['experiments/diagnose_surface_cdf_v01.py','tests/test_surface_cdf_diagnostic_v01.py']
    return dict(old.sources(),**base.sources(),**{name:sha(ROOT/name)for name in paths})


def selected_npy_row(archive,member,row=None):
    """Only expose the requested numeric row; never materialize other rows."""
    if member not in ('matching_linear.npy','prefix_noise.npy'):raise ValueError('Prefix-only archive member allowlist')
    with zipfile.ZipFile(archive)as z, z.open(member)as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
        else:raise ValueError('Supported NPY header1/2 required')
        if fortran or dtype.hasobject or dtype!=np.dtype('<f8'):raise ValueError('Expected C-order float64 prefix archive')
        if row is None:
            if shape!=(24,):raise ValueError('Only24-channel prefix noise vector allowed')
            count=24
        else:
            if len(shape)!=2 or shape[1]!=24 or type(row)is not int or not 0<=row<shape[0]:raise ValueError('Selected24-channel prefix row required')
            count=24;f.seek(f.tell()+row*count*dtype.itemsize)
        data=f.read(count*dtype.itemsize)
        if len(data)!=count*dtype.itemsize:raise ValueError('Truncated selected prefix member')
        return np.frombuffer(data,dtype=dtype).copy()


def old_closed():
    closure=json.loads((old.LOCAL/'controlled-stop-ownership.json').read_text())
    if not closure['all17_PID_groups_absent']or not closure['all16_workers_reaped']:raise ValueError('Old owned run must close first')
    for pid in closure['owned_absence']:
        if Path('/proc',pid).exists()or old._processes(int(pid)):raise ValueError('Old process/group remains')


def prepare():
    old_closed()
    if(OUT/'protocol.json').exists():raise ValueError('No refreeze/retry')
    paths=[INPUT/'truth-LOCAL.npz',INPUT/'truth-checkpoint-LOCAL.json',INPUT/'matching_linear-prefix-LOCAL.json',old.LOCAL/'controlled-stop-ownership.json',old.LOCAL/'stopped-completed-manifest-LOCAL.json']
    if json.loads(paths[2].read_text())['qualified']is not False:raise ValueError('Assigned failed prefix must stay failed')
    p=dict(schema='surface-cdf-diagnosis-v01',scope='Retrospective numerical diagnosis only; never requalifies frozenv03 or scores policy',object_index=2,law='matching_linear',prefix=PREFIX,
        outer=[128,256,512],inner=128,unit_CDF_knots=np.linspace(0,1,33).tolist(),CPU_worker=17,CPU_controller=16,GPU=False,threads=1,RSS_bytes=4*1024**3,worker_wall_s=300,global_wall_s=300,
        oracle_rule='Union of both endpoints of mostnegative adjacentCDFincrement perlevel/dimension if negative; plus worst outofbounds knot if any. Original-a oracle full and these selected partials at1e-8; recordwarnings/errors failclosed. All33rawCDFknots retained locally.',
        complement_rule='Positive original amplitude_nodes kernel on complement a<=threshold with all prior/q-dependent kinks; compare logsumexp(low,high) against fullZ without clipping',
        no_heldout_truth_reads=True,no_parameters_or_probe_outcomes=True,no_qualification_attempt=True,source_SHA256=sources(),input_SHA256={str(p.relative_to(ROOT)):sha(p)for p in paths})
    write(OUT/'protocol.json',p);return sha(OUT/'protocol.json')


def validate(digest):
    if sha(OUT/'protocol.json')!=digest:raise ValueError('Protocol hash changed')
    p=json.loads((OUT/'protocol.json').read_text())
    if p['source_SHA256']!=sources()or any(sha(ROOT/name)!=value for name,value in p['input_SHA256'].items()):raise ValueError('Source/input pin changed')
    old_closed();return p


def guard():
    if os.sched_getaffinity(0)!={17}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('CPU17/one-thread/noGPU guard')


def log_complement(engine,dimension,threshold,outer,inner):
    if threshold==0:return float(logsumexp(engine._nodes(outer,inner)[3]))
    if threshold==1:return -np.inf
    q=np.exp(np.log(BOUNDS[dimension,0])+threshold*LOG_WIDTH[dimension])
    cuts=[BOUNDS[1,0]/q,BOUNDS[1,1]/q]if dimension==0 else[q/BOUNDS[0,1],q/BOUNDS[0,0]]
    breaks=sorted(set(engine._breaks().tolist()+[float(np.log(v))for v in cuts if BOUNDS[1,0]/BOUNDS[0,1]<v<BOUNDS[1,1]/BOUNDS[0,0]]))
    t,w=gauss_axis(outer);z=np.concatenate([a+(b-a)*t for a,b in zip(breaks,breaks[1:])]);zw=np.concatenate([(b-a)*w for a,b in zip(breaks,breaks[1:])])
    r,mu,sd,peak=engine._statistics(z);lo=np.maximum(1/BOUNDS[0,1],r/BOUNDS[1,1]);hi=np.minimum(1/BOUNDS[0,0],r/BOUNDS[1,0]);hi=np.minimum(hi,1/q if dimension==0 else r/q);valid=hi>lo
    if not np.any(valid):return -np.inf
    _,lw=amplitude_nodes(mu[valid],sd[valid],lo[valid],hi[valid],inner)
    return float(logsumexp(lw+peak[valid,None]+np.log(zw[valid,None])-np.log(LOG_WIDTH.prod())))


def diagnose_level(engine,outer,inner=128):
    knots=np.linspace(0,1,33);full=float(logsumexp(engine._nodes(outer,inner)[3]));cdf=np.empty((2,33));closure=np.empty_like(cdf);selected=set()
    for d in range(2):
        for j,u in enumerate(knots):
            low=-np.inf if u==0 else full if u==1 else float(logsumexp(engine._nodes(outer,inner,(d,float(u)))[3]))
            high=log_complement(engine,d,float(u),outer,inner)
            cdf[d,j]=np.exp(low-full);closure[d,j]=float(np.logaddexp(low,high)-full)
        diff=np.diff(cdf[d]);j=int(np.argmin(diff))
        if diff[j]<0:selected.update(((d,j),(d,j+1)))
        excess=np.maximum(-cdf[d],cdf[d]-1);j=int(np.argmax(excess))
        if excess[j]>0:selected.add((d,j))
    return dict(outer=outer,inner=inner,log_evidence=full,minimum_CDF_increment=float(np.min(np.diff(cdf,axis=1))),maximum_CDF_bound_excess=float(np.max(np.maximum(-cdf,cdf-1))),maximum_log_low_high_closure_error=float(np.max(abs(closure)))),cdf,closure,selected


def worker(digest):
    p=validate(digest);guard();started=time.monotonic();meta=json.loads((INPUT/'truth-checkpoint-LOCAL.json').read_text());row=meta['labels'].index(PREFIX)
    observation=selected_npy_row(INPUT/'truth-LOCAL.npz','matching_linear.npy',row)+selected_npy_row(INPUT/'truth-LOCAL.npz','prefix_noise.npy')
    original=json.loads((INPUT/'matching_linear-prefix-LOCAL.json').read_text())
    if old.observation_fingerprint([(PREFIX,observation)])!=original['observations_SHA256']:raise ValueError('Exact saved prefix observation replay mismatch')
    _,model=old.make_model(device_mode='cpu')
    if base.model_fingerprint(model)!=original['model_fingerprint']:raise ValueError('Same allmode model fingerprint required')
    ratio=RatioModal(model.modal,model.labels);callback=ExactRatioMemo(lambda z,l:ratio(np.exp(z),l),model_fingerprint=base.model_fingerprint(model),maximum_bytes=256*1024**2)
    engine=ConditionalAmplitudeReference(None,[(PREFIX,observation)],[PREFIX],log_ratio_predict=callback)
    levels=[];arrays={};selected=set()
    for n in p['outer']:
        scalar,cdf,closure,knots=diagnose_level(engine,n,p['inner']);levels.append(scalar);arrays[f'CDF_{n}']=cdf;arrays[f'closure_{n}']=closure;selected.update(knots);write(OUT/'progress-LOCAL.json',dict(levels=levels,selected_oracle_knots=len(selected),protocol_SHA256=digest))
    np.savez_compressed(OUT/'CDF-LOCAL.npz',**arrays)
    with warnings.catch_warnings():
        warnings.simplefilter('error',IntegrationWarning)
        full=adaptive_integral(engine._observed_ratio,engine.y,sigma=engine.sigma,epsrel=1e-8);oracle=[]
        for d,j in sorted(selected):
            u=j/32;q=np.exp(np.log(BOUNDS[d,0])+u*LOG_WIDTH[d])
            if u in(0,1):value=u;error=0.
            else:
                part=adaptive_integral(engine._observed_ratio,engine.y,sigma=engine.sigma,CDF=('E'if d==0 else'K',float(q)),epsrel=1e-8);value=float(np.exp(part['log_integral']-full['log_integral']));error=max(part['estimated_outer_relative_error'],part['maximum_estimated_inner_relative_error'])
            oracle.append(dict(dimension=d,knot=j,CDF=value,estimated_relative_error=error,maximum_Gauss_CDF_difference=float(max(abs(arrays[f'CDF_{n}'][d,j]-value)for n in p['outer']))))
            write(OUT/'oracle-progress-LOCAL.json',dict(full_log_evidence=full['log_integral'],oracle=oracle))
    write(OUT/'result-LOCAL.json',dict(completed=True,protocol_SHA256=digest,wall_s=time.monotonic()-started,levels=levels,independent_full=full,oracle=oracle,full_log_evidence_differences=[abs(v['log_evidence']-full['log_integral'])for v in levels],old_prefix_qualified=False,no_qualification_attempt=True,no_heldout_truth_reads=True))


def command(digest,guard_only=False):
    return ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','taskset','-c','17',sys.executable,'-m','experiments.diagnose_surface_cdf_v01','guard'if guard_only else'worker','--digest',digest]


def run(digest):
    validate(digest)
    if os.sched_getaffinity(0)!={16}or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('ControllerCPU16/noGPU')
    from experiments.run_campaign_surrogate_sbc_v03 import owned_jobs
    from experiments.campaign_owned_proc import proc_rss
    def check():
        path=OUT/'journal.json';live=[]if not path.exists()else json.loads(path.read_text()).get('live',[])
        if proc_rss(os.getpid())+sum(proc_rss(v['pid'])for v in live)>4*1024**3:raise MemoryError('Aggregate4GiB cap')
    records=owned_jobs(lambda i,s:command(digest),[0],1,OUT,worker_wall_s=300,global_wall_s=300,RSS_bytes=4*1024**3,controller_check=check)
    write(OUT/'owned-receipt.json',dict(completed=True,all_reaped=True,controller_pid=os.getpid(),records=records,protocol_SHA256=digest))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run','worker','guard']);a.add_argument('--digest');x=a.parse_args()
    if x.mode=='prepare':print(prepare())
    elif x.mode=='run':run(x.digest)
    elif x.mode=='guard':guard();print(json.dumps(dict(guard_passed=True)))
    else:worker(x.digest)
