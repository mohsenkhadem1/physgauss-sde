"""Post-campaign dependence control; same posterior/evidence, no model retuning.

The control p(theta|y) N(F; global posterior mean,covariance) retains force
moments but Gaussianizes its marginal. Target mean/covariance equal those of
the exact independent product p(theta|y)p(F|y); marginal CDFs need not match.
"""
from pathlib import Path
import argparse,collections,csv,datetime,hashlib,json,time
import numpy as np
from . import surface as bank, surface_campaign as campaign
from .inference import Posterior
ROOT=bank.ROOT;BASE=ROOT/'reports/joint-field-surface-v04';OUT=BASE/'dependence'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)
def independent(p):
    mean=p.weights@p.means;within=np.einsum('h,hfg->fg',p.weights,p.covariances)
    between=np.einsum('h,hf,hg->fg',p.weights,p.means-mean,p.means-mean)
    total=within+between
    q=Posterior(p.weights,np.broadcast_to(mean,p.means.shape),np.broadcast_to(total,p.covariances.shape))
    return q,dict(within_force_covariance_trace_N2=float(np.trace(within)),between_force_covariance_trace_N2=float(np.trace(between)),total_force_covariance_trace_N2=float(np.trace(total)),between_force_covariance_fraction=float(np.trace(between)/np.trace(total)))
def product_mc(p,q,m,truth,i,seed,n=100000):
    selected=np.r_[np.linspace(0,m['split']-1,12,dtype=int),np.linspace(m['split'],m['T'].shape[1]-1,12,dtype=int)]
    T=m['T'][:,selected];offset=m['offset'][:,selected];rng=np.random.default_rng(seed)
    L=np.linalg.cholesky(p.covariances);samples=np.empty((n,len(selected)))
    for start in range(0,n,2048):
        count=min(2048,n-start);materialidx=rng.choice(len(p.weights),count,p=p.weights);forceidx=rng.choice(len(p.weights),count,p=p.weights)
        force=p.means[forceidx]+np.einsum('bij,bj->bi',L[forceidx],rng.normal(size=(count,p.means.shape[1])))
        samples[start:start+count]=np.einsum('bdf,bf->bd',T[materialidx],force)+offset[materialidx]
    exact=q.target(T,offset);lo,hi=np.quantile(samples,[.05,.95],axis=0);true=truth['target'][i,selected]
    covered=(true>=lo)&(true<=hi)
    return dict(samples=n,seed=seed,selected_coordinates=selected.tolist(),coverage90=float(covered.mean()),hidden_coverage90=float(covered[:12].mean()),changed_coverage90=float(covered[12:].mean()),interval_width_mm=float(np.mean(hi-lo)*1000),mean_rms_difference_from_exact_product_moments_mm=float(np.sqrt(np.mean((samples.mean(0)-exact.mean)**2))*1000),variance_rms_difference_m2=float(np.sqrt(np.mean((samples.var(0)-exact.variance)**2))),scope='Independent sampling from exact theta marginal and exact Gaussian-mixture force marginal; MC equal-tail intervals on same24coordinates. Not an exact CDF.')
def paired_summary(rows):
    summary=[];comparisons=[]
    keys=['rmse_mm','hidden_rmse_mm','changed_load_rmse_mm','coverage90','hidden_coverage90','changed_coverage90','interval_width_mm','force_coverage90','material_coverage90','force_profile_rmse_N_per_m']
    for stage in ['initial','active']:
      for shape in ['block','step','pooled']:
        rr=[r for r in rows if r['stage']==stage and (shape=='pooled' or r['shape']==shape)]
        for model in ['full-joint','independent-force-moments']:
            vv=[r for r in rr if r['model']==model]
            if vv:summary.append(dict(stage=stage,shape=shape,model=model,n=len(vv),**{k:float(np.mean([r[k] for r in vv])) for k in keys}))
        a={(r['shape'],r['case']):r for r in rr if r['model']=='full-joint'};b={(r['shape'],r['case']):r for r in rr if r['model']=='independent-force-moments'};ids=sorted(set(a)&set(b))
        if not ids:continue
        rng=np.random.default_rng(192001);bootidx=rng.integers(len(ids),size=(2000,len(ids)))
        for k in keys:
            delta=np.array([a[i][k]-b[i][k] for i in ids]);boot=delta[bootidx].mean(1)
            comparisons.append(dict(stage=stage,shape=shape,metric=k,n=len(ids),full_minus_independent=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    return summary,comparisons

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mc-samples',type=int,default=100000);ap.add_argument('--wait',action='store_true');a=ap.parse_args()
    if a.wait:
        deadline=datetime.datetime.fromisoformat(json.loads((BASE/'protocol.json').read_text())['deadline_utc'])
        while any(not (BASE/shape/'matched'/f'case-{i:03d}.json').exists() for shape in ['block','step'] for i in range(64)):
            if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('Dependence cohort incomplete at deadline')
            time.sleep(30)
    OUT.mkdir(exist_ok=True);protocol=json.loads((BASE/'protocol.json').read_text());fingerprint=sha(BASE/'protocol.json')
    for f,h in protocol['source_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError(f'Frozen source changed: {f}')
    plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),main_protocol_sha256=fingerprint,source_sha256=sha(__file__),cohort='All64matchedcases pergeometry;initial and same4recordedactiveactions',models=['full-joint','independent-force-moments'],exact_product_mc='First4cases pergeometry, bothstages, no outcome selection',mc_samples=a.mc_samples,mc_seed_rule='191001+1000*(shape==step)+10*case+(stage==active)',scope='Additional diagnostic on completed fixed model, not a newly untouched test set. No favorable outcome required.')
    planpath=OUT/'plan.json'
    if planpath.exists():
        old=json.loads(planpath.read_text())
        if old['source_sha256']!=plan['source_sha256'] or old['mc_samples']!=a.mc_samples:raise ValueError('Audit plan changed; do not overwrite prior results')
    else:save(planpath,plan)
    started=time.time();rows=[];documents=[]
    for shape in ['block','step']:
        data=dict(np.load(ROOT/f'runs/joint-field-surface-v04/{shape}.npz'));m=bank.maps(data);truth=dict(np.load(ROOT/f'runs/joint-field-surface-v04/{shape}/matched-truth.npz'))
        for i in range(64):
            result=OUT/f'{shape}-case-{i:03d}.json'
            if result.exists():
                doc=json.loads(result.read_text());documents.append(doc);rows.extend(doc['rows']);continue
            recordpath=BASE/shape/'matched'/f'case-{i:03d}.json';record=json.loads(recordpath.read_text());actions=[r['action'] for r in record['rows'] if r['policy']=='active' and r['step']>0]
            p=campaign.initial(m,truth['y'][i]);rr=[];diagnostics=[]
            for stage in ['initial','active']:
                if stage=='active':
                    for action in actions:p=campaign.action_update(p,m,action,truth['measurements'][i,action])
                q,decomp=independent(p)
                # Force/material first moments and force second moments remain exactly preserved.
                fullF=p.target(np.eye(16));indF=q.target(np.eye(16))
                momentcheck=dict(force_mean_max_abs_N=float(np.max(np.abs(fullF.mean-indF.mean))),force_variance_max_abs_N2=float(np.max(np.abs(fullF.variance-indF.variance))),material_weight_max_abs=float(np.max(np.abs(p.weights-q.weights))))
                assert momentcheck['force_mean_max_abs_N']<1e-12 and momentcheck['force_variance_max_abs_N2']<1e-12
                for model,post in [('full-joint',p),('independent-force-moments',q)]:
                    metrics=campaign.metrics(post,data,m,truth,i,intervals=True)
                    rr.append(dict(shape=shape,case=i,stage=stage,model=model,**metrics))
                    if model=='full-joint':
                        saved=next(r for r in record['rows'] if r['policy']=='active' and r['step']==(0 if stage=='initial' else 4))
                        assert all(np.isclose(metrics[k],saved[k],rtol=1e-9,atol=1e-12) for k in metrics), 'Reconstructed full posterior differs from frozen run'
                diag=dict(stage=stage,**decomp,marginal_moment_identity=momentcheck)
                if i<4:diag['exact_independent_product_mc']=product_mc(p,q,m,truth,i,191001+1000*(shape=='step')+10*i+(stage=='active'),a.mc_samples)
                diagnostics.append(diag)
            doc=dict(shape=shape,case=i,main_protocol_sha256=fingerprint,case_record_sha256=sha(recordpath),rows=rr,diagnostics=diagnostics,actions=actions)
            save(result,doc);documents.append(doc);rows.extend(rr)
            print(shape,i,'elapsed',round(time.time()-started,1),flush=True)
    summary,comparisons=paired_summary(rows)
    fractions={stage:float(np.mean([d['between_force_covariance_fraction'] for doc in documents for d in doc['diagnostics'] if d['stage']==stage])) for stage in ['initial','active']}
    mc=[dict(shape=doc['shape'],case=doc['case'],stage=d['stage'],**d['exact_independent_product_mc']) for doc in documents for d in doc['diagnostics'] if 'exact_independent_product_mc' in d]
    report=dict(complete=len(documents)==128,cases=len(documents),rows=len(rows),summary=summary,paired_comparisons=comparisons,mean_between_force_covariance_fraction=fractions,exact_product_mc=mc,elapsed_seconds=time.time()-started,scope='Same frozen evidence and theta marginal. Independent control preserves full force mean/cov but Gaussianizes its marginal. Its target mean/cov equal exact independent-product moments; its CDF/coverage do not generally equal the exact product. Material marginal is unchanged. Changed-load predictions depend only on theta, so are identical by construction. Paired bootstrap is over cases in two fixed shapes, not shape-population inference.')
    save(OUT/'summary.json',report)
    with (OUT/'all-case-metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps(dict(complete=report['complete'],cases=len(documents),between_force_covariance_fraction=fractions)),flush=True)
if __name__=='__main__':main()
