"""Surface-visible v04 campaign. Explicit snapshot of v01 campaign with new surface import, roots and seeds; no mutation of frozen v01 globals."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from . import surface as bank
from .inference import fit_posterior,Posterior,expected_risk_mc,marginal_quantiles
BASE=bank.ROOT/'runs/joint-field-surface-v04';REPORT=bank.ROOT/'reports/joint-field-surface-v04'

def load_bank(shape):
    return dict(np.load(BASE/f'{shape}.npz'))

def generate(shape,data,n,seed,kind='matched',device='cpu'):
    rng=np.random.default_rng(seed);idx=rng.integers(len(data['theta']),size=n)
    if kind in ('matched','force-prior'):
        truth={k:v[idx] for k,v in data.items() if k in ('theta','unit','probe','material')};truth['points']=data['points']
    elif kind in ('offgrid-fine','continuous-coarse'):
        th=rng.uniform(bank.BOUNDS[:,0],bank.BOUNDS[:,1],size=(n,5))
        truth,_=bank.build(shape,th,(12,6,6) if kind=='offgrid-fine' else (8,4,4),device,batch=8)
    else:raise ValueError(kind)
    tm=bank.maps(truth);K=bank.load_cov()
    if kind=='force-prior':
        d=np.abs(bank.TIMES[:,None]-bank.TIMES[None,:]);r=np.sqrt(3)*d/.10
        K=bank.FORCE_STD**2*np.kron((1+r)*np.exp(-r),np.eye(2))
    force=rng.normal(size=(n,16))@np.linalg.cholesky(K).T
    y=np.einsum('hof,hf->ho',tm['H'],force)+rng.normal(scale=bank.NOISE,size=(n,tm['H'].shape[1]))
    target=np.einsum('hdf,hf->hd',tm['T'],force)+tm['offset']
    measured=np.stack([np.einsum('hf,hf->h',a['h'],force)+a['offset'] for a in tm['actions']],1)+rng.normal(scale=bank.NOISE,size=(n,len(tm['actions'])))
    return dict(theta=truth['theta'],material=truth['material'],force=force,y=y,target=target,measurements=measured,indices=idx,seed=np.array(seed),kind=np.array(kind))

def initial(m,y,prior=None,scale=1.):
    return fit_posterior(m['H'],y,bank.load_cov(scale),bank.NOISE**2,prior_weights=prior)

def action_update(p,m,action,value):
    a=m['actions'][action];return p.update_scalar(a['h'],value,bank.NOISE**2,a['offset'])

def metrics(p,data,m,truth,i,intervals=False):
    pred=p.target(m['T'],m['offset']);split=m['split'];err=pred.mean-truth['target'][i]
    material=p.weights@data['material'];force=p.weights@p.means
    result=dict(rmse_mm=float(np.sqrt(np.mean(err**2))*1000),hidden_rmse_mm=float(np.sqrt(np.mean(err[:split]**2))*1000),changed_load_rmse_mm=float(np.sqrt(np.mean(err[split:]**2))*1000),predictive_sd_mm=float(np.sqrt(pred.risk)*1000),material_relative_rmse=float(np.sqrt(np.mean((material-truth['material'][i])**2)/np.mean(truth['material'][i]**2))),force_rmse_N=float(np.sqrt(np.mean((force-truth['force'][i])**2))),support_center_error_mm=float(abs(p.weights@data['theta'][:,2]-truth['theta'][i,2])*1000),load_center_error_mm=float(abs(p.weights@data['theta'][:,4]-truth['theta'][i,4])*1000),hypothesis_ess=float(1/np.sum(p.weights**2)))
    # Normalized longitudinal load-basis proxy, N/m; not exact surface-integrated traction.
    x=np.linspace(-.06,.06,41);width=.018
    profile=np.exp(-.5*((x[None]-data['theta'][:,4,None])/width)**2)
    profile/=np.trapezoid(profile,x,axis=1)[:,None]
    pred_field=np.einsum('h,hx,htd->txd',p.weights,profile,p.means.reshape(-1,8,2),optimize=True)
    true_profile=np.exp(-.5*((x-truth['theta'][i,4])/width)**2);true_profile/=np.trapezoid(true_profile,x)
    true_field=truth['force'][i].reshape(8,2)[:,None,:]*true_profile[None,:,None]
    result['force_profile_rmse_N_per_m']=float(np.sqrt(np.mean((pred_field-true_field)**2)))
    # Equal-tail mixture intervals on predeclared sparse target coordinates.
    if intervals:
        selected=np.r_[np.linspace(0,split-1,12,dtype=int),np.linspace(split,len(err)-1,12,dtype=int)]
        lo,hi=marginal_quantiles(p,m['T'][:,selected],offset=m['offset'][:,selected]);true=truth['target'][i,selected]
        covered=(true>=lo)&(true<=hi)
        result.update(coverage90=float(covered.mean()),hidden_coverage90=float(covered[:12].mean()),changed_coverage90=float(covered[12:].mean()),interval_width_mm=float((hi-lo).mean()*1000))
        # deterministic field uncertainty under finite material hypotheses
        loE,hiE=marginal_quantiles(p,np.zeros((len(p.weights),len(material),16)),offset=data['material'])
        result['material_coverage90']=float(((truth['material'][i]>=loE)&(truth['material'][i]<=hiE)).mean())
        loF,hiF=marginal_quantiles(p,np.eye(16));trueF=truth['force'][i]
        result['force_coverage90']=float(((trueF>=loF)&(trueF<=hiF)).mean())
    return result

def plugin(p,kind):
    if kind=='marginal-map':j=int(np.argmax(p.weights))
    elif kind=='joint-map':
        _,ld=np.linalg.slogdet(p.covariances)
        j=int(np.argmax(np.log(np.maximum(p.weights,1e-300))-.5*ld))
    else:raise ValueError(kind)
    w=np.zeros_like(p.weights);w[j]=1
    return Posterior(w,p.means,np.zeros_like(p.covariances))

def select_fixed(data,m,truth,budget):
    """Strong fixed schedules selected on separate development truths only."""
    out={}
    for name,allowed in [('fixed',list(range(len(m['actions'])))),('fixed-view',[i for i,a in enumerate(m['actions']) if a['kind']=='view']),('fixed-probe',[i for i,a in enumerate(m['actions']) if a['kind']=='probe'])]:
        ps=[initial(m,y) for y in truth['y']];chosen=[];history=[]
        for step in range(budget):
            scores=[]
            for a in allowed:
                if a in chosen:continue
                score=[]
                for i,p in enumerate(ps):
                    new=action_update(p,m,a,truth['measurements'][i,a]);mu=np.einsum('hdf,hf->d',m['T'],new.weights[:,None]*new.means,optimize=True)+new.weights@m['offset']
                    score.append(np.mean((mu-truth['target'][i])**2))
                scores.append((float(np.mean(score)),a))
            value,a=min(scores);chosen.append(a);history.append(dict(action=a,development_mse_m2=value))
            ps=[action_update(p,m,a,truth['measurements'][i,a]) for i,p in enumerate(ps)]
        out[name]=dict(actions=chosen,history=history)
    return out

def run_case(data,m,truth,i,fixed,budget,mc,seed):
    p0=initial(m,truth['y'][i]);rows=[];acquisition=[]
    for method in ['marginal-map','joint-map']:
        rows.append(dict(policy=method,step=0,**metrics(plugin(p0,method),data,m,truth,i)))
    # Representation ablations: declared uniform priors on constrained dictionaries.
    for name,mask in [('uniform-material',np.isclose(data['theta'][:,1],0)),('fixed-interaction',np.isclose(data['theta'][:,2],0)&np.isclose(data['theta'][:,4],0)&np.isclose(data['theta'][:,3],bank.BOUNDS[3].mean()))]:
        p=initial(m,truth['y'][i],mask.astype(float));rows.append(dict(policy=name,step=0,**metrics(p,data,m,truth,i)))
    # Preserve force marginal first two moments and material marginal; remove dependence.
    mean=p0.weights@p0.means;cov=np.einsum('h,hfg->fg',p0.weights,p0.covariances)+np.einsum('h,hf,hg->fg',p0.weights,p0.means-mean,p0.means-mean)
    independent=Posterior(p0.weights,np.broadcast_to(mean,p0.means.shape),np.broadcast_to(cov,p0.covariances.shape))
    rows.append(dict(policy='independent-moments',step=0,**metrics(independent,data,m,truth,i)))
    base=metrics(p0,data,m,truth,i,intervals=True)
    for policy in ['active','fixed','fixed-view','fixed-probe','random']:
        p=p0;chosen=[];rows.append(dict(policy=policy,step=0,**base))
        randomorder=np.random.default_rng(seed+901).permutation(len(m['actions']))
        for step in range(budget):
            if policy=='active':
                scores=[];cached_target=p.target(m['T'],m['offset'])
                for a,act in enumerate(m['actions']):
                    if a in chosen:continue
                    score=expected_risk_mc(p,m['T'],act['h'],bank.NOISE**2,act['offset'],n_samples=mc,seed=seed+step,target_offset=m['offset'],return_details=True,target_moments=cached_target)
                    scores.append(dict(action=a,**score))
                a=min(scores,key=lambda v:v['expected_risk'])['action']
                acquisition.append(dict(step=step,chosen=a,scores=scores))
            elif policy=='random':a=int(randomorder[step])
            else:a=fixed[policy]['actions'][step]
            chosen.append(a);p=action_update(p,m,a,truth['measurements'][i,a])
            rows.append(dict(policy=policy,step=step+1,action=a,action_kind=m['actions'][a]['kind'],**metrics(p,data,m,truth,i,intervals=(step==budget-1))))
    return dict(rows=rows,acquisition=acquisition)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shape',choices=['block','step'],required=True);ap.add_argument('--phase',choices=['development','test'],required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--cases',type=int,default=64);ap.add_argument('--budget',type=int,default=4);ap.add_argument('--mc',type=int,default=128);ap.add_argument('--kind',default='matched',choices=['matched','offgrid-fine','force-prior','continuous-coarse']);args=ap.parse_args()
    data=load_bank(args.shape);m=bank.maps(data);root=BASE/args.shape;root.mkdir(parents=True,exist_ok=True);report=REPORT/args.shape;report.mkdir(parents=True,exist_ok=True)
    if args.phase=='development':
        truth=generate(args.shape,data,12,121001+(args.shape=='step'),device=args.device);np.savez(root/'development.npz',**truth)
        fixed=select_fixed(data,m,truth,args.budget);(report/'fixed-schedules.json').write_text(json.dumps(fixed,indent=2));print(fixed,flush=True);return
    protocol_path=REPORT/'protocol.json'
    protocol=json.loads(protocol_path.read_text());protocol_hash=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    for source,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((bank.ROOT/source).read_bytes()).hexdigest()==digest, f'Frozen source changed: {source}'
    assert args.budget==protocol['budget'] and args.mc==protocol['acquisition_mc']
    assert args.cases==protocol['case_counts'][args.kind]
    seed=131001+10000*(args.shape=='step')+{'matched':0,'offgrid-fine':1000,'force-prior':2000,'continuous-coarse':3000}[args.kind]
    path=root/f'{args.kind}-truth.npz'
    if path.exists():
        truth=dict(np.load(path));assert str(truth['protocol_sha256'])==protocol_hash
    else:
        truth=generate(args.shape,data,args.cases,seed,args.kind,args.device);truth['protocol_sha256']=np.array(protocol_hash);np.savez(path,**truth)
    assert len(truth['y'])==args.cases
    fixed=json.loads((report/'fixed-schedules.json').read_text());out=report/args.kind;out.mkdir(exist_ok=True)
    for i in range(args.cases):
        from datetime import datetime, timezone
        if datetime.now(timezone.utc)>=datetime.fromisoformat(protocol['deadline_utc']):
            print('Deadline reached; remaining cases incomplete',flush=True);return
        result=out/f'case-{i:03d}.json'
        if result.exists():
            assert json.loads(result.read_text())['protocol_sha256']==protocol_hash
            continue
        start=time.time();doc=run_case(data,m,truth,i,fixed,args.budget,args.mc,seed+1_000_000+i*100)
        doc.update(protocol_sha256=protocol_hash,scene=args.shape,population=args.kind,case=i,seed=seed+1_000_000+i*100,seconds=time.time()-start)
        result.write_text(json.dumps(doc,indent=2));print(args.shape,args.kind,i,round(doc['seconds'],2),flush=True)
if __name__=='__main__':main()
