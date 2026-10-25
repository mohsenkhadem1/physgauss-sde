"""Read-only diagnosis of one frozen nonlinear cost state; no probability repair."""
from __future__ import annotations
import argparse,json,os,resource,signal,threading,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_surface_nonlinear_posterior_cost_v01 as old
from experiments.campaign_surface_conditional_amplitude_v01 import BOUNDS,LOG_WIDTH,log_gaussian_interval
ROOT=old.ROOT
LOCAL=ROOT/'runs/operator-dev/surface-tilted-cdf-diagnostic-v01'
SOURCE='experiments/run_surface_tilted_cdf_diagnostic_v01.py'
TEST='tests/test_surface_tilted_cdf_diagnostic_v01.py'
ORIGIN='8e48a46387dae1936e47650ccabd9b74f40f99a66ddc4481607bb7154f1091ee'
LIMITS=dict(CPU=17,threads=1,GPU=False,wall_s=300,RSS_bytes=4*1024**3)
sha,record,write=old.sha,old.record,old.write


def sources():
    return {**old.sources(),SOURCE:sha(ROOT/SOURCE),TEST:sha(ROOT/TEST)}


def origin():
    p,a=old.validate(ORIGIN);old.release_gate(p,ORIGIN)
    e=json.loads((old.LOCAL/'execution.json').read_text());w=json.loads((old.LOCAL/'worker-result-LOCAL.json').read_text())
    if not(e['terminal']and e['completed']and not e['qualified']and e['worker_exit_code']==0 and e['cleanup_exit_code']==0 and w['completed']and not w['passed']):raise ValueError('Exact completed numerical-negative origin')
    old.absent(e['controller_pid']);old.absent(e['worker_pid'])
    if e['result']!=record(old.LOCAL/'worker-result-LOCAL.json')or w['observation']!=record(old.LOCAL/'observation-LOCAL.npz'):raise ValueError('Original result/observation hash')
    return p,a,dict(protocol=record(old.LOCAL/'protocol.json'),execution=record(old.LOCAL/'execution.json'),result=record(old.LOCAL/'worker-result-LOCAL.json'),observation=w['observation'],release=record(old.LOCAL/'release.json'),atlas=p['full_atlas']['archive'])


def extrema(x):
    x=np.asarray(x,float);finite=np.isfinite(x)
    return dict(shape=list(x.shape),nonfinite=int((~finite).sum()),minimum=float(x[finite].min())if finite.any()else None,maximum=float(x[finite].max())if finite.any()else None)


def mass_diagnostics(p):
    width=np.diff(p.knots);rw=np.diff(1-p.knots[::-1]);h=p.h
    analytic=width[None,:]*(h[:,:-1]+h[:,1:])/2
    reflected=rw[None,:]*(h[:,::-1][:,:-1]+h[:,::-1][:,1:])/2
    return dict(density_pairwise_relative=extrema((p.density_weights.sum(axis=1)-p.mass0)/p.mass0),panel_pairwise_relative=extrema((p.panel_mass.sum(axis=1)-p.mass0)/p.mass0),forward_sequential_relative=extrema((np.cumsum(p.panel_mass,axis=1)[:,-1]-p.mass0)/p.mass0),reverse_sequential_relative=extrema((np.cumsum(p.panel_mass[:,::-1],axis=1)[:,-1]-p.mass0)/p.mass0),reflected_width_relative=extrema((rw-width[::-1])/width[::-1]),forward_full_panel_minus_stored_over_mass0=extrema((analytic-p.panel_mass)/p.mass0[:,None]),reflected_full_panel_minus_stored_over_mass0=extrema((reflected-p.panel_mass[:,::-1])/p.mass0[:,None]))


def raw_probability(p,q,tail):
    """Duplicate the frozen arithmetic without its final range rejection only.

    No clipping or normalization change. Alternative quantities are diagnosis,
    never returned to the engine or used to admit a posterior.
    """
    q=np.asarray(q,float);below=q<=p.lower;above=q>=p.upper;inside=~(below|above)
    out=np.empty_like(q);out[below]=1. if tail else 0.;out[above]=0. if tail else 1.
    t=np.full_like(q,np.nan);normal=inside&~p.zero;zero=inside&p.zero
    if normal.any():
        m=p.mu[normal];s=p.SD[normal];alpha=(p.lower[normal]-m)/s;beta=(p.upper[normal]-m)/s;cut=(q[normal]-m)/s
        left=log_gaussian_interval(alpha,cut);right=log_gaussian_interval(cut,beta)
        t[normal]=np.exp((right if tail else left)-np.logaddexp(left,right))
    if zero.any():t[zero]=(np.log(p.upper[zero]/q[zero])if tail else np.log(q[zero]/p.lower[zero]))/np.log(p.upper[zero]/p.lower[zero])
    rows=np.flatnonzero(inside);details=[];parity=0.
    if len(rows):
        knots=1-p.knots[::-1]if tail else p.knots;h=p.h[inside,::-1]if tail else p.h[inside];mass=p.panel_mass[inside,::-1]if tail else p.panel_mass[inside]
        x=t[inside];idx=np.minimum(np.searchsorted(knots,x,side='right')-1,len(knots)-2);ar=np.arange(len(rows));delta=x-knots[idx];width=knots[idx+1]-knots[idx];fraction=delta/width
        local=delta*((1-.5*fraction)*h[ar,idx]+.5*fraction*h[ar,idx+1]);prefix=np.concatenate((np.zeros((len(rows),1)),np.cumsum(mass,axis=1)),axis=1);numerator=prefix[ar,idx]+local
        direct=p._primitive(knots,h,mass,x);parity=float(np.max(abs(direct-numerator)));out[inside]=numerator/p.mass0[inside]
        end=prefix[:,-1];original_width=np.diff(p.knots)[::-1]if tail else np.diff(p.knots)
        # Include all violating rows and bounded extrema, not only high-weight rows.
        selected=set(np.flatnonzero((out[inside]<0)|(out[inside]>1+32*np.finfo(float).eps)|~np.isfinite(out[inside])).tolist())
        selected.update([int(np.argmin(out[inside])),int(np.argmax(out[inside]))])
        for j in sorted(selected):
            r=int(rows[j]);k=int(idx[j]);full=width[j]*(h[j,k]+h[j,k+1])/2;stored=mass[j,k];m0=p.mass0[r]
            details.append(dict(row=r,probability=float(out[r]),excess_over_one=float(out[r]-1),threshold=float(q[r]),coordinate=float(x[j]),panel=k,mass0=float(m0),sequential_end=float(end[j]),prefix=float(prefix[j,k]),local=float(local[j]),stored_panel_mass=float(stored),local_full_panel=float(full),width=float(width[j]),original_corresponding_width=float(original_width[k]),width_relative=float(width[j]/original_width[k]-1),sequential_minus_mass0_relative=float((end[j]-m0)/m0),local_minus_remaining_relative=float((local[j]-(end[j]-prefix[j,k]))/m0),local_full_minus_stored_relative=float((full-stored)/m0),geometry_only_local_delta_relative=float(local[j]*(1-original_width[k]/width[j])/m0)))
    finite=np.isfinite(out);stats=dict(**extrema(out),below_zero=int((out<0).sum()),above_one=int((out>1).sum()),above_original_guard=int((out>1+32*np.finfo(float).eps).sum()),inside_rows=len(rows),primitive_bit_arithmetic_maximum_difference=parity,offending_and_extreme_rows=details)
    return out,stats


def prepare():
    _,_,records=origin()
    p=dict(schema='surface-tilted-cdf-diagnostic-v01',status='FROZEN_ROOT_GO_REQUIRED',source_SHA256=sources(),origin=records,resources=LIMITS,state=dict(atlas=9,outer=512,inner=128),thresholds_unit=np.linspace(0,1,33).tolist(),sides=['CDF','tail'],original_guard_epsilon_multiplier=32,scope='One original failed state/same saved observation, no draws/PDE/qualifier/repair. Alternative sums and reflected widths are diagnostic only; no calibration admission.')
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text());_,_,records=origin()
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['origin']!=records or p['resources']!=LIMITS:raise ValueError('Frozen source/protocol/origin')
    return p


def run(digest):
    old.cost.guard('worker');p=validate(digest)
    release=json.loads((LOCAL/'release.json').read_text())
    if not release.get('approved')or release.get('protocol_SHA256')!=digest:raise ValueError('Explicit root GO before arrays')
    if(LOCAL/'result-LOCAL.json').exists():raise FileExistsError('No diagnostic retry')
    start=time.monotonic();state=dict(completed=False,pid=os.getpid(),protocol_SHA256=digest,source_SHA256=p['source_SHA256'],release=record(LOCAL/'release.json'),calls=[],no_engine_changes=True,no_draws=True,no_qualifier_calls=True);done=threading.Event()
    def save():
        state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'result-LOCAL.json',state)
    def stop(*_):raise TimeoutError('Frozen300s/RSS4GiB diagnostic guard')
    previous={sig:signal.signal(sig,stop)for sig in(signal.SIGALRM,signal.SIGTERM)};signal.alarm(300)
    def watch():
        while not done.wait(.25):
            try:
                rss=next(int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines()if l.startswith('VmRSS:'))
                if rss>LIMITS['RSS_bytes']:os.kill(os.getpid(),signal.SIGTERM);return
            except(FileNotFoundError,StopIteration):return
    threading.Thread(target=watch,daemon=True).start()
    try:
        from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas
        from experiments.campaign_surface_p2_static_v02 import SparseStaticForward,point_query_matrix
        from experiments.campaign_surface_p2_force_v03 import load_catalog
        from experiments.campaign_surface_p2_study_design_v01 import observation_points
        from experiments.campaign_surface_p2_inference_v01 import LabelledModal
        from experiments.campaign_surface_conditional_amplitude_v01 import RatioModal,ExactRatioMemo
        from experiments.campaign_surface_response_cache_v01 import model_fingerprint
        from experiments.campaign_nonlinear_conditional_reference_v01 import NonlinearConditionalReference
        op,_=old.validate(ORIGIN)
        with np.load(ROOT/p['origin']['atlas']['path'],allow_pickle=False)as z:atlas=NonlinearResidualAtlas(z['delta'],tuple(str(v)for v in z['labels']),provenance_fingerprint=op['full_protocol_SHA256'])
        with np.load(ROOT/p['origin']['observation']['path'],allow_pickle=False)as z:observation=z['observation_m'].copy()
        fwd=SparseStaticForward(4);linear=LabelledModal.from_forward(fwd,loads=load_catalog(4),points=observation_points(),condensed=True,query_matrix=point_query_matrix);ratio=RatioModal(linear.modal,linear.labels);memo=ExactRatioMemo(lambda z,labels:ratio(np.exp(z),labels),model_fingerprint=model_fingerprint(linear),maximum_bytes=256*1024**2,minimum_rows=1)
        engine=NonlinearConditionalReference(memo,None,[(op['prefix'],observation)],atlas.labels if hasattr(atlas,'labels')else tuple(op['all_target_labels']),residual_predict=atlas.predict,retain_hypotheses=False,evaluation_chunk=1024)
        s=engine._state(512,128);primitive=s['primitive'];state.update(state_build_s=time.monotonic()-start,rows=len(s['rho']),log_evidence=s['log_evidence'],mass_diagnostics=mass_diagnostics(primitive));save()
        raw_arrays={}
        for dim in(0,1):
            for j,unit in enumerate(p['thresholds_unit']):
                physical=float(np.exp(np.log(BOUNDS[dim,0])+unit*LOG_WIDTH[dim]));cut=np.full_like(s['r'],1/physical)if dim==0 else s['r']/physical
                try:original=dict(status='returned',value=engine.cdf(dim,unit,outer=512,inner=128,log_evidence=s['log_evidence']))
                except(FloatingPointError,ValueError)as error:original=dict(status='rejected',type=type(error).__name__,message=str(error))
                sides={}
                for tail in(False,True):
                    raw,stats=raw_probability(primitive,cut,tail);name='tail'if tail else'CDF';raw_arrays[f'd{dim}_q{j}_{name}']=raw
                    try:answer=primitive._probability(cut,tail=tail);stats['original_primitive_status']='returned';stats['original_manual_maximum_difference']=float(np.max(abs(answer-raw)))
                    except(FloatingPointError,ValueError)as error:stats['original_primitive_status']='rejected';stats['original_error']=str(error)
                    for row in stats['offending_and_extreme_rows']:row['rho']=float(s['rho'][row['row']])
                    stats['rho_weighted_raw_probability']=float(s['rho']@raw);sides[name]=stats
                state['calls'].append(dict(dimension=dim,unit_threshold=unit,physical_threshold=physical,original_engine=original,sides=sides));save();old.cost.guard('worker')
        np.savez_compressed(LOCAL/'raw-probabilities-LOCAL.npz',**raw_arrays,rho=s['rho'],z=s['z']);state.update(completed=True,raw=record(LOCAL/'raw-probabilities-LOCAL.npz'),memo=memo.diagnostics());save()
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());save();raise
    finally:
        done.set();signal.alarm(0)
        for sig,handler in previous.items():signal.signal(sig,handler)
        save()
    return state

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--sha');a=parser.parse_args()
    if a.action=='prepare':print(json.dumps(prepare()))
    else:r=run(a.sha);print(json.dumps(dict(completed=r['completed'],wall_s=r['wall_s'],result=record(LOCAL/'result-LOCAL.json'))))
