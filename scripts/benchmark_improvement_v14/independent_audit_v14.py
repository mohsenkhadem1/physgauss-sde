"""Read-only post-score audit. No fresh target read until all180 forecasts committed."""
import sys,json,hashlib,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.benchmark_improvement_v14.common import BASE,REPORT,read,sha,write,remaining
from scripts.benchmark_improvement_v14.evaluate import verify_freeze,cases
from scripts.benchmark_improvement_v14.provenance import validate_record


def paired(rows,baseline,metric):
    values=[]
    for r in rows:
        m=dict(r.get('archived_controls',{}),**r['metrics']);a=m['benchmark_v14']['summary'].get(metric);b=m.get(baseline,{}).get('summary',{}).get(metric)
        if a is not None and b is not None and np.isfinite([a,b]).all():
            if min(a,b)<0:raise ValueError('Negative physical error')
            values.append((r['case']['name'],float(a),float(b)))
    return values


def frame_scores(pred,target,start):
    """Independent equal-frame metrics; proper rigid alignment per frame."""
    coordinate=[];shape=[];failed=0
    for a,b in zip(pred[start:],target[start:]):
        if not np.isfinite(a).all():failed+=1;continue
        coordinate.append(float(np.sqrt(np.mean((a-b)**2))*1000))
        ac=a-a.mean(0);bc=b-b.mean(0);u,_,vt=np.linalg.svd(ac.T@bc);s=np.eye(3);s[-1,-1]=np.linalg.det(u@vt)
        shape.append(float(np.sqrt(np.mean((ac@(u@s@vt)-bc)**2))*1000))
    return dict(coordinate_rmse_mm=None if not coordinate else float(np.mean(coordinate)),aligned_shape_rmse_mm=None if not shape else float(np.mean(shape)),failed_frames=failed,valid_frames=len(coordinate),total_frames=len(pred)-start)


def main():
    remaining();frozen=verify_freeze('SoftVTBench');declared=cases('SoftVTBench');errors=[];checks=0;evidence={}
    def check(ok,msg):
        nonlocal checks
        checks+=1
        if not ok:errors.append(msg)
    def bind(p):evidence[str(Path(p).relative_to(ROOT))]=sha(p)
    if len(declared)!=180:raise ValueError('Fixed180cohort required')
    for c in declared:
        out=BASE/'evaluation'/c['cohort']/c['name'];record=read(out/'complete.json')
        if record['freeze_sha256']!=frozen:raise ValueError('Wrong forecast freeze')
        validate_record(c,record,out)
        for p in [out/'complete.json',out/'metadata.json',out/'predictions.npz']:bind(p)
        evidence.update({str(Path(p).relative_to(ROOT)):h for p,h in record['sources'].items()})
    # No evaluator or HDF is opened above. Scores must also be complete first.
    scores=read(REPORT/'softvt-scores.json');summary=read(REPORT/'softvt-summary.json')
    rows=scores['cases'];lookup={r['case']['name']:r for r in rows};expected={c['name']:c for c in declared}
    if len(rows)!=180 or set(lookup)!=set(expected):raise ValueError('Score census mismatch')
    for name,r in lookup.items():
        if r['case']!=expected[name] or r['freeze_sha256']!=frozen:raise ValueError('Score identity mismatch')
        out=BASE/'evaluation'/r['case']['cohort']/name
        if r['prediction_sha256']!=sha(out/'predictions.npz'):raise ValueError('Score prediction changed')
    check(scores['freeze_sha256']==frozen and summary['freeze_sha256']==frozen,'score/summary freeze')
    check(summary['source_sha256']==sha(REPORT/'softvt-scores.json'),'summary source')
    for group,comparisons in summary['groups'].items():
        selected=[r for r in rows if r['case']['cohort']!='prospective_softvt_confirmation'] if group=='all_historical_exposed' else [r for r in rows if r['case']['cohort']==group]
        for label,result in comparisons.items():
            metric='aligned_shape_rmse_mm' if label=='shape_vs_original' else 'coordinate_rmse_mm';baseline=result['baseline'];v=paired(selected,baseline,metric)
            check(result['available_pairs']==len(v),'paircount '+group+label)
            check(result['declared_cases']==len(selected),'declaredcount '+group+label)
            complete_values=[]
            for r in selected:
                methods=dict(r.get('archived_controls',{}),**r['metrics']);aa=methods['benchmark_v14']['summary'];bb=methods.get(baseline,{}).get('summary',{})
                for ss in [aa,bb]:
                    if ss:check(ss['valid_frames']+ss['failed_frames']==ss['total_frames'],'coverage '+r['case']['name']+label)
                if bb and aa['failed_frames']==bb['failed_frames']==0 and aa['total_frames']==bb['total_frames'] and any(x[0]==r['case']['name'] for x in v):complete_values.append((aa[metric],bb[metric]))
            check(result['all_cases_and_frames_complete']==(len(complete_values)==len(selected)),'complete flag '+group+label)
            if not v:continue
            cp=result['complete_pairs'];check(cp['count']==len(complete_values),'completepairs '+group+label)
            if complete_values:
                cv=np.array(complete_values);check(np.isclose(cp['proposal_mean_mm'],cv[:,0].mean()) and np.isclose(cp['baseline_mean_mm'],cv[:,1].mean()),'completepairmeans '+group+label)
            a=np.array([x[1] for x in v]);b=np.array([x[2] for x in v]);gain=b-a
            for k,x in [('proposal_mean_mm',a.mean()),('baseline_mean_mm',b.mean()),('mean_improvement_mm',gain.mean())]:check(np.isclose(result[k],x,rtol=1e-11,atol=1e-10),'summary '+group+label+k)
            check(result['wins']==int((gain>1e-8).sum()) and result['losses']==int((gain< -1e-8).sum()) and result['ties']==int((abs(gain)<=1e-8).sum()),'paired wins '+group+label)
            ids=np.random.default_rng(140031).integers(0,len(gain),(10000,len(gain)));interval=np.quantile(np.mean(gain[ids],axis=1),[.025,.975])
            check(np.allclose(result['paired_episode_bootstrap_95_interval_mm'],interval,rtol=1e-11,atol=1e-10),'bootstrap '+group+label)
    import h5py  # Raw target access occurs only after complete fixed-cohort checks.
    target_cases=['object-soft-task0-demo2','object-soft-task0-demo5','object-soft-task1-demo30'];raw_results=[]
    for name in target_cases:
        remaining();r=lookup[name];c=r['case'];spec=c['spec'];family='object' if spec['subset']=='object-soft' else 'spatial'
        folder=ROOT/'runs/external-benchmark-v11/raw/SoftVTBench'/spec['subset']/f'libero_{family}'/f'libero_{family}_task{spec["task"]}'/'replayed_demos';hdf=next(folder.glob('*.hdf5'))
        with h5py.File(hdf) as h:
            g=h[f'data/demo_{spec["episode"]}'];target=g[f'states/deformable_object/{g.attrs["asset_name"]}/nodal_pos_w'][:].astype(float)
        target_hash=hashlib.sha256(target.tobytes()).hexdigest();out=BASE/'evaluation'/c['cohort']/name;end=int(.6*len(target));per={}
        with np.load(out/'predictions.npz') as z:
            for method in z.files:
                result=frame_scores(z[method],target,end);expected_score=r['metrics'][method]['summary']
                for key,value in result.items():
                    wanted=expected_score.get(key)
                    check(wanted is None if value is None else wanted is not None and np.isclose(wanted,value,rtol=1e-8,atol=1e-7),'raw '+name+method+key)
                per[method]=result
        raw_results.append(dict(name=name,source=str(hdf.relative_to(ROOT)),target_array_sha256=target_hash,methods=per))
    prior=read(REPORT/'prior-preservation.json')['sha256']
    for p,h in prior.items():check(sha(ROOT/p)==h,'preservation '+p)
    for p in [REPORT/'softvt-scores.json',REPORT/'softvt-summary.json',REPORT/'protocol.json',REPORT/'softvt-freeze.json',Path(__file__)]:bind(p)
    output=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=not errors,checks=checks,errors=errors,scope='All180 forecast/input/source and score identities; all summary paired arithmetic/bootstrap independently recomputed from scored records; raw saved-array coordinate/shape recomputation on fixeddevelopment2/5 andfreshfirst30 only. No claim rawmetric replay for everycase.',fixed_raw_cases=raw_results,prior_files_preserved=len(prior),evidence_sha256=evidence)
    write(REPORT/'softvt-final-independent-audit.json',output)
    if errors:raise RuntimeError(str(errors[:8]))
    print('Independent SoftVT audit passed',checks,flush=True)

if __name__=='__main__':main()
