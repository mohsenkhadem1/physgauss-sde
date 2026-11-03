"""Pair methods at the independent-world level, never at the video-frame level.

Error comparisons condition on both methods completing every declared query for
that world. Completion/coverage counts are always returned alongside the errors.
"""
import numpy as np
from collections import Counter


def paired(rows,method,baseline,metric='vector_rms_mm',queries_per_world=2,bootstrap_samples=10000,seed=91):
    selected=[r for r in rows if r['method']in (method,baseline)]
    keys=[(r['method'],r['world_id'],r['query_index'])for r in selected]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate benchmark slot')
    worlds=sorted({r['world_id']for r in selected});differences=[];included=[];excluded=[]
    counts={m:dict(Counter(r['state']for r in selected if r['method']==m))for m in (method,baseline)}
    for world in worlds:
        pair=[]
        for m in (method,baseline):
            group=[r for r in selected if r['world_id']==world and r['method']==m]
            if len(group)!=queries_per_world or {r['query_index']for r in group}!=set(range(queries_per_world)) or any(r['state']!='complete'for r in group):break
            values=[r['metrics'][metric]for r in group]
            if not np.isfinite(values).all():raise ValueError('Nonfinite completed-world metric')
            pair.append(float(np.mean(values)))
        if len(pair)==2:included.append(world);differences.append(pair[0]-pair[1])
        else:excluded.append(world)
    if len(differences)>=2:
        d=np.asarray(differences);rng=np.random.default_rng(seed)
        boot=d[rng.integers(len(d),size=(bootstrap_samples,len(d)))].mean(1)
        interval=np.quantile(boot,[.025,.975]).tolist()
    else:interval=None
    return dict(method=method,baseline=baseline,metric=metric,scope='Conditional on both methods completing all queries; negative differences favor method; bootstrap resamples independent worlds',slot_counts=counts,included_worlds=included,excluded_worlds=excluded,paired_world_differences=differences,mean_difference=float(np.mean(differences))if differences else None,world_bootstrap_95_interval=interval,bootstrap_samples=bootstrap_samples,bootstrap_seed=seed)
