"""Independent read-only outcome audit; no imports from campaign/inference code."""
import datetime, hashlib, json
from pathlib import Path
import numpy as np
from scipy.special import softmax
R=Path('reports/gap-first-v01/bowl'); U=Path('runs/gap-first-v01/bowl')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def intervals(w,q):
    order=np.argsort(q,axis=0,kind='stable'); values=np.take_along_axis(q,order,axis=0)
    mass=np.cumsum(w[order].astype(np.longdouble),axis=0);mass/=mass[-1]
    return [np.take_along_axis(values,np.argmax(mass>=p,axis=0)[None],axis=0)[0] for p in (.05,.95)]
def point_errors(q,x): return np.linalg.norm((q-x).reshape(len(q),-1,3),axis=2).__pow__(2).mean(1)**.5
def ci(values,worlds):
    v=np.array([np.mean(np.array(values)[np.array(worlds)==h]) for h in range(16)])
    draws=np.random.default_rng(1107901).integers(16,size=(5000,16))
    return v.mean(),np.quantile(v[draws].mean(1),[.025,.975])
def main():
    for f in ['case-results.json','summary.json','prediction-seal.json','bank-seal.json']:
        if not (R/f).exists(): raise SystemExit('Not ready: '+f)
    checks=0;max_diff=0.
    def eq(a,b):
        nonlocal checks,max_diff
        aa,bb=np.asarray(a),np.asarray(b);diff=float(np.max(np.abs(aa.astype(float)-bb.astype(float))))
        assert np.allclose(aa,bb,rtol=1e-9,atol=1e-11),(a,b,diff)
        max_diff=max(max_diff,diff);checks+=1
    protocol=read(R/'protocol.json')
    for p,h in protocol['source_sha256'].items():assert sha(p)==h,p
    deps=read(R/'imported-dependency-seal.json')
    for p,h in deps['sha256'].items():assert sha(p)==h,p
    for p,h in read(R/'bank-seal.json').items():assert sha(p)==h,p
    seal=read(R/'prediction-seal.json');assert sha(R/'prospective-predictions.json')==seal['sha256'];assert sha(R/'protocol.json')==seal['protocol_sha256']
    assert (R/'protocol.json').stat().st_mtime<(R/'bank-seal.json').stat().st_mtime<=(R/'prospective-predictions.json').stat().st_mtime<=(R/'case-results.json').stat().st_mtime
    coarse=dict(np.load(U/'fem-n6-bank.npz'));fine=dict(np.load(U/'fem-n12-bank.npz'))
    banks={'fem':coarse,'resolution-mixture':{k:np.concatenate([coarse[k],fine[k]]) for k in ('initial','action','target','reaction')}}
    rows=read(R/'case-results.json');pred=read(R/'prospective-predictions.json');summary=read(R/'summary.json');assert len(rows)==len(pred)==576
    index={(r['model'],r['cohort'],r['hypothesis'],r['replicate'],r['stage']):r for r in rows};assert len(index)==576
    utilities={};shared=0
    for r,p in zip(rows,pred):
        for key in ('model','cohort','hypothesis','replicate','stage'):assert r[key]==p[key]
        for key in ('weights','map_index','predicted_failure_probability'):eq(r[key],p[key])
        model,cohort,h,rep=r['model'],r['cohort'],r['hypothesis'],r['replicate'];bank=banks[model];truthbank=coarse if cohort=='matched' else fine
        rng=np.random.default_rng(1107100+(0 if cohort=='matched' else 1000)+10*h+rep)
        y=truthbank['initial'][h]+.0005*rng.normal(size=truthbank['initial'][h].shape)
        ya=truthbank['action'][h]+.0005*rng.normal(size=truthbank['action'][h].shape)
        ll=-.5*np.sum(((bank['initial']-y)/.0005)**2,axis=1)
        key=(model,cohort,h,rep)
        if key not in utilities:
            w0=softmax(ll); z=np.random.default_rng(110701).normal(size=(128,bank['action'].shape[-1]));q=bank['target'];fail=np.array([point_errors(q,t)>.003 for t in q]);scores=[]
            for a in range(2):
                vals=[]
                for j in range(len(w0)):
                    ys=bank['action'][j,a]+.0005*z
                    ws=softmax(ll[None]-.5*np.sum(((bank['action'][None,:,a]-ys[:,None])/.0005)**2,axis=2),axis=1)
                    vals.append(np.mean(np.sum(ws*fail[np.argmax(ws,axis=1)],axis=1)))
                scores.append(float(w0@vals))
            utilities[key]=scores
        scores=utilities[key];eq(scores,r['pre_outcome_expected_risks']);assert r['chosen_action']==('calibration_push','calibration_tilt')[int(np.argmin(scores))]
        stage=r['stage'];a=None
        if stage=='fixed_push':a=0
        elif stage=='fixed_tilt':a=1
        elif stage=='active':a=int(np.argmin(scores))
        elif stage=='shared_coarse_active':
            other=index['fem',cohort,h,rep,'active'];a=('calibration_push','calibration_tilt').index(other['chosen_action']);shared+=1
        if a is not None:ll-=.5*np.sum(((bank['action'][:,a]-ya[a])/.0005)**2,axis=1)
        w=softmax(ll);eq(w,r['weights']);j=int(w.argmax());eq(j,r['map_index']);q=bank['target'];truth=truthbank['target'][h];err=float(point_errors(truth[None],q[j])[0]);risk=float(w@(point_errors(q,q[j])>.003));lo,hi=intervals(w,q)
        metrics={'map_rms_mm':err*1000,'posterior_mean_rms_mm':float(point_errors(truth[None],w@q)[0])*1000,'predicted_failure_probability':risk,'actual_failure':err>.003,'brier':(risk-float(err>.003))**2,'coordinate_coverage90':np.mean((truth>=lo)&(truth<=hi)),'coordinate_width90_mm':np.mean(hi-lo)*1000,'effective_hypotheses':1/(w@w)}
        f=bank['reaction'];ft=truthbank['reaction'][h].reshape(3,2,3);flo,fhi=intervals(w,f)
        for estimator,fp in [('mean',w@f),('map',f[j])]:
            eq(fp,p['reaction_'+estimator+'_N'])
            for c,name in enumerate(('bowl','tool')):metrics[name+'_reaction_'+estimator+'_vector_rms_N']=np.sqrt(np.mean(np.sum((fp.reshape(3,2,3)[:,c]-ft[:,c])**2,axis=1)))
        for c,name in enumerate(('bowl','tool')):
            l,u=flo.reshape(3,2,3)[:,c],fhi.reshape(3,2,3)[:,c]
            metrics[name+'_reaction_coordinate_coverage90']=np.mean((ft[:,c]>=l)&(ft[:,c]<=u));metrics[name+'_reaction_coordinate_width90_N']=np.mean(u-l)
        for k,v in metrics.items():eq(v,r[k])
    for key,group in summary['groups'].items():
        model,cohort,stage=key.split('/');rr=[r for r in rows if (r['model'],r['cohort'],r['stage'])==(model,cohort,stage)];assert len(rr)==32
        for metric,value in group.items():
            if isinstance(value,dict) and 'cluster_bootstrap95' in value:
                m,c=ci([r[metric] for r in rr],[r['hypothesis'] for r in rr]);eq(m,value['mean']);eq(c,value['cluster_bootstrap95'])
    for key,group in summary['paired'].items():
        parts=key.split('/')
        if len(parts)==3:
            model,cohort,contrast=parts;otherstage=contrast.removeprefix('active-minus-')
            left=[index[model,cohort,h,r,'active'] for h in range(16) for r in range(2)]
            right=[index[model,cohort,h,r,otherstage] for h in range(16) for r in range(2)]
        else:
            cohort=parts[0]
            left=[index['resolution-mixture',cohort,h,r,'shared_coarse_active'] for h in range(16) for r in range(2)]
            right=[index['fem',cohort,h,r,'active'] for h in range(16) for r in range(2)]
        for metric,value in group.items():
            m,c=ci([float(a[metric])-float(b[metric]) for a,b in zip(left,right)],[r['hypothesis'] for r in left]);eq(m,value['mean']);eq(c,value['cluster_bootstrap95'])
    out={'passed':True,'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':576,'independent_checks':checks,'maximum_numeric_discrepancy':max_diff,'independent_action_utility_recomputations':len(utilities),'shared_coarse_action_evidence_checks':shared,'independent_world_clusters':16,'source_and_bank_and_prediction_seals_verified':True,'chronology':'File timestamps verify protocol/bank/prediction/outcome order; code separates indexed truth targets until score stage.','scope':['Primary coarse matched finite bank is conditional model validation.','Coarse versus finer truth is discretization stress, not continuum certification.','Secondary mixture includes fine model by construction; pooled matched/fine equal-weight distribution matches resolution prior, per-resolution90% coverage need not.','Force outputs are withheld trailing20ms vector reactions, not observed evidence.','No native-framework ranking or real RGB reconstruction claim.'],'source_sha256':{str(p):sha(p) for p in [R/'protocol.json',R/'bank-seal.json',R/'prediction-seal.json',R/'case-results.json',R/'summary.json',R/'imported-dependency-seal.json',Path(__file__)]}}
    (R/'independent-scientific-final-review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
