"""Read-only all-eight audit; target loading gated on complete forecasts and scores."""
import sys
from pathlib import Path
import datetime
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.benchmark_improvement_v14.common import BASE,REPORT,read,write,sha,remaining
from scripts.benchmark_improvement_v14.evaluate import cases,verify_freeze
from scripts.benchmark_improvement_v14.phystwin_evaluation import validate_new
from scripts.benchmark_improvement_v14.independent_audit_v14 import paired,frame_scores


def manual_score(expanded,manual,start):
    if not np.isfinite(expanded[0]).all():return None
    initial_valid=np.isfinite(manual[0]).all(1);ids=cKDTree(expanded[0]).query(manual[0,initial_valid])[1];frames=[]
    for t in range(start,len(expanded)):
        target=manual[t,initial_valid];p=expanded[t,ids];valid=np.isfinite(target).all(1)&np.isfinite(p).all(1)
        if valid.any():frames.append(float(np.linalg.norm(p[valid]-target[valid],axis=1).mean()*1000))
    return float(np.mean(frames)) if frames else None


def main():
    remaining();frozen=verify_freeze('PhysTwin');declared=cases('PhysTwin');errors=[];checks=0;evidence={}
    def check(ok,msg):
        nonlocal checks
        checks+=1
        if not ok:errors.append(msg)
    def bind(p):evidence[str(Path(p).relative_to(ROOT))]=sha(p)
    if len(declared)!=8:raise ValueError('Exacteight required')
    for c in declared:
        out=BASE/'evaluation'/c['cohort']/c['name'];validate_new(c,out);record=read(out/'complete.json')
        for p in [out/'complete.json',out/'predictions.npz',out/'metadata.json']:bind(p)
        for p,h in dict(record['sources'],**record['new_fit_sources']).items():evidence[str(Path(p).relative_to(ROOT))]=h
    scores=read(REPORT/'phystwin-scores.json');summary=read(REPORT/'phystwin-summary.json');rows=scores['cases'];lookup={r['case']['name']:r for r in rows}
    if len(rows)!=8 or {r['case']['name']:r['case'] for r in rows}!={c['name']:c for c in declared}:raise ValueError('Score census mismatch')
    check(scores['freeze_sha256']==frozen and summary['freeze_sha256']==frozen,'freeze')
    check(summary['source_sha256']==sha(REPORT/'phystwin-scores.json'),'summary hash')
    for group,comparisons in summary['groups'].items():
        selected=rows if group=='all_historical_exposed' else [r for r in rows if r['case']['cohort']==group]
        for label,r in comparisons.items():
            metric='aligned_shape_rmse_mm' if label=='shape_vs_original' else 'manual_track_mean_mm';v=paired(selected,r['baseline'],metric)
            check(r['available_pairs']==len(v) and r['declared_cases']==len(selected),'pair census '+group+label)
            if not v:continue
            a=np.array([x[1] for x in v]);b=np.array([x[2] for x in v]);g=b-a
            for k,x in [('proposal_mean_mm',a.mean()),('baseline_mean_mm',b.mean()),('mean_improvement_mm',g.mean())]:check(np.isclose(r[k],x,rtol=1e-10,atol=1e-8),'paired metric '+group+label+k)
            check(r['wins']==int((g>1e-8).sum()) and r['losses']==int((g< -1e-8).sum()) and r['ties']==int((abs(g)<=1e-8).sum()),'wins '+group+label)
            boot=g[np.random.default_rng(140031).integers(0,len(g),(10000,len(g)))].mean(1);check(np.allclose(np.quantile(boot,[.025,.975]),r['paired_episode_bootstrap_95_interval_mm'],rtol=1e-10,atol=1e-8),'bootstrap '+group+label)
    # First target loader import is after exact8 committed forecasts and scores.
    from scripts.external_benchmark.data import phystwin,interpolate_matrix
    raw=[]
    for c in declared:
        remaining();name=c['name'];r=lookup[name];out=BASE/'evaluation'/c['cohort']/name;data=phystwin(name,300);target=data['evaluation']['points'];mask=data['evaluation']['mask'];manual=data['evaluation']['manual'];initial=data['support_geometry'];A=interpolate_matrix(data['nodes'],initial);end=data['train_end'];meta=read(out/'metadata.json')
        check(r['prediction_sha256']==sha(out/'predictions.npz') and r['freeze_sha256']==frozen,'scores identity '+name)
        method_results={}
        with np.load(out/'predictions.npz') as saved:
            if not meta.get('switched'):check(saved['benchmark_v14'].tobytes()==saved['object_frame'].tobytes(),'exact original fallback '+name)
            for method in saved.files:
                expanded=initial[None]+np.einsum('pn,tnc->tpc',A,saved[method]-data['nodes'][None]);coordinate=[];shape=[];failed=0
                for t in range(end,len(expanded)):
                    if not np.isfinite(expanded[t]).all():failed+=1;continue
                    valid=mask[t]&np.isfinite(target[t]).all(1);a=expanded[t,:len(target[t])][valid];b=target[t,valid];coordinate.append(float(np.sqrt(np.mean((a-b)**2))*1000));ac=a-a.mean(0);bc=b-b.mean(0);u,_,vt=np.linalg.svd(ac.T@bc);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vt);shape.append(float(np.sqrt(np.mean((ac@(u@fix@vt)-bc)**2))*1000))
                actual=dict(coordinate_rmse_mm=float(np.mean(coordinate)) if coordinate else None,aligned_shape_rmse_mm=float(np.mean(shape)) if shape else None,manual_track_mean_mm=manual_score(expanded,manual,end),failed_frames=failed,valid_frames=len(coordinate),total_frames=len(expanded)-end)
                wanted=r['metrics'][method]['summary']
                for k,v in actual.items():check(wanted.get(k) is None if v is None else wanted.get(k) is not None and np.isclose(wanted[k],v,rtol=1e-8,atol=1e-7),'raw '+name+method+k)
                method_results[method]=actual
        raw.append(dict(name=name,methods=method_results,switched=meta.get('switched')))
    prior=read(REPORT/'prior-preservation.json')['sha256']
    for p,h in prior.items():check(sha(ROOT/p)==h,'preserved '+p)
    for p in [REPORT/'phystwin-scores.json',REPORT/'phystwin-summary.json',REPORT/'phystwin-freeze.json',REPORT/'protocol.json',Path(__file__)]:bind(p)
    output=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=not errors,checks=checks,errors=errors,scope='All8 saved forecasts: independent per-frame coordinate/proper-shape/manual arithmetic and summary paired/bootstrap recomputation; original frozen geometry interpolation reused. All8targets loaded only after committed prediction and score census. Selection/operator tests separately cited; no independent optimizer rerun.',raw_cases=raw,prior_files_preserved=len(prior),evidence_sha256=evidence)
    write(REPORT/'phystwin-final-independent-audit.json',output)
    if errors:raise RuntimeError(str(errors[:10]))
    print('PhysTwin independent audit passed',checks,flush=True)

if __name__=='__main__':main()
