"""Population motion/deformation audit using strictly exposed prefix adapters.

No simulator, training, future object trajectories, RGB decoding or fresh episode
reads. Tracks are privileged SoftVT nodes or author-provided PhysTwin tracks.
These are signal diagnostics, not elasticity identifiability or ground-truth strain.
"""
import argparse,hashlib,json,time,sys,gzip
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import (
    load_softvt_predictor,load_phystwin_predictor,predictor_digest)


def proper_rigid_residual(reference,target):
    x=reference-reference.mean(0);y=target-target.mean(0)
    U,_,Vt=np.linalg.svd(x.T@y)
    d=np.ones(3);d[-1]=np.linalg.det(U@Vt)
    rotation=(U*d[None])@Vt
    return target-(x@rotation+target.mean(0))


def quantiles(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a):return None
    q=np.quantile(a,[0,.1,.25,.5,.75,.9,1])
    return {k:float(v)for k,v in zip(('min','p10','p25','median','p75','p90','max'),q)}


def measure_prefix(observed,mask,*,minimum_points=6,motion_threshold_m=.0001):
    observed=np.asarray(observed,float);mask=np.asarray(mask,bool)
    if observed.ndim!=3 or observed.shape[-1]!=3 or mask.shape!=observed.shape[:2]:
        raise ValueError('Expected prefix [T,N,3] and [T,N] mask')
    if len(observed)<2:raise ValueError('At least two observed frames required')
    initial_finite=np.isfinite(observed[0]).all(-1)
    reference_mask=mask[0]&initial_finite
    rows=[];total_ss=0.;shape_ss=0.;count=0;eligible_motion=0
    for frame in range(1,len(observed)):
        valid=reference_mask&mask[frame]&np.isfinite(observed[frame]).all(-1)
        n=int(valid.sum());row={'frame':frame,'common_valid_points':n}
        if n<minimum_points:
            row.update(eligible=False,reason='insufficient_common_valid_points');rows.append(row);continue
        x=observed[0,valid];y=observed[frame,valid]
        spectrum=np.linalg.svd(x-x.mean(0),compute_uv=False)
        if spectrum[0]<=1e-12 or spectrum[1]<=spectrum[0]*1e-8:
            row.update(eligible=False,reason='degenerate_reference_geometry');rows.append(row);continue
        delta=y-x;residual=proper_rigid_residual(x,y)
        sst=float(np.sum(delta**2));sss=float(np.sum(residual**2));den=3*n
        total=np.sqrt(sst/den);shape=np.sqrt(sss/den)
        if shape>total+1e-9:raise RuntimeError('Proper rigid fit cannot be worse than identity')
        informative_motion=bool(total>=motion_threshold_m)
        ratio=shape/total if informative_motion else None
        row.update(eligible=True,above_motion_threshold=informative_motion,total_motion_RMS_mm=float(total*1000),rigid_removed_shape_RMS_mm=float(shape*1000),shape_to_total_RMS_ratio=None if ratio is None else float(ratio))
        rows.append(row)
        total_ss+=sst;shape_ss+=sss;count+=den;eligible_motion+=informative_motion
    eligible=[r for r in rows if r.get('eligible')]
    moving=[r for r in eligible if r['above_motion_threshold']]
    if not count:
        return {'eligible':False,'reason':'no_eligible_frames','frames':rows,'prefix_frames':len(observed),'track_count':observed.shape[1]}
    overall_total=np.sqrt(total_ss/count);overall_shape=np.sqrt(shape_ss/count)
    ratio=overall_shape/overall_total if overall_total>=motion_threshold_m else None
    centered=observed[0,reference_mask]-observed[0,reference_mask].mean(0)
    radius=np.sqrt(np.mean(np.sum(centered**2,axis=-1)))
    return {'eligible':True,'prefix_frames':len(observed),'track_count':observed.shape[1],
            'initial_valid_points':int(reference_mask.sum()),'eligible_frames':len(eligible),
            'eligible_frame_fraction':len(eligible)/(len(observed)-1),'moving_frames':len(moving),
            'aggregate_total_motion_RMS_mm':float(overall_total*1000),
            'aggregate_rigid_removed_shape_RMS_mm':float(overall_shape*1000),
            'aggregate_shape_to_total_RMS_ratio':None if ratio is None else float(ratio),
            'aggregate_shape_energy_fraction':None if ratio is None else float(ratio**2),
            'shape_RMS_relative_to_initial_radius':float(overall_shape/radius),
            'initial_radius_mm':float(radius*1000),
            'frame_total_motion_mm':quantiles([r['total_motion_RMS_mm']for r in eligible]),
            'frame_shape_motion_mm':quantiles([r['rigid_removed_shape_RMS_mm']for r in eligible]),
            'moving_frame_RMS_ratio':quantiles([r['shape_to_total_RMS_ratio']for r in moving]),
            'common_valid_point_counts':quantiles([r['common_valid_points']for r in eligible]),
            'frames':rows}


def object_group(name):
    for label in ('cloth','rope','sloth','zebra','dinosor','package','baby','rabbit','elephant','bear','duck','sponge','ball'):
        if label in name:return label
    return name


def summary(rows):
    good=[r for r in rows if r.get('metrics',{}).get('eligible')]
    metrics={}
    for key in ('aggregate_total_motion_RMS_mm','aggregate_rigid_removed_shape_RMS_mm','aggregate_shape_to_total_RMS_ratio','shape_RMS_relative_to_initial_radius','eligible_frame_fraction'):
        metrics[key]=quantiles([r['metrics'][key]for r in good if r['metrics'][key]is not None])
    low=[r for r in good if r['metrics']['aggregate_shape_to_total_RMS_ratio']is not None and r['metrics']['aggregate_shape_to_total_RMS_ratio']<.05]
    return {'cases':len(rows),'eligible_cases':len(good),'ineligible_cases':len(rows)-len(good),
            'case_weighted_quantiles':metrics,'cases_below_5pct_nonrigid_RMS_fraction':len(low)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='reports/gaussian-boundary-flow-v02/prefix-informativeness');args=p.parse_args()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    source_files=[Path(__file__),Path('scripts/gaussian_boundary_flow_v02/benchmark_inputs.py'),Path('scripts/external_benchmark/data.py'),Path('scripts/prefix_ridge_switch_v13/inputs.py'),Path('reports/gaussian-boundary-flow-v02/dataset-audit.json')]
    hashes_before={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in source_files}
    catalog=json.loads(Path('reports/gaussian-boundary-flow-v02/dataset-audit.json').read_text())
    phys_cases=[r['case']for r in catalog['PhysTwin']['cases']]
    if len(phys_cases)!=22:raise ValueError('Freeze explicit 22-case historically exposed catalog')
    assets={(subset,task):next(iter(entry['assets']))for entry in catalog['SoftVTBench']['files']for subset in ('object-soft','spatial-soft')for task in range(3)if f'/{subset}/'in entry['path']and f'_task{task}/'in entry['path']}
    if len(assets)!=6:raise ValueError('Frozen asset metadata must cover all six suites/tasks')
    specs=[{'dataset':'SoftVTBench','subset':subset,'task':task,'episode':episode,
            'name':f'{subset}-task{task}-demo{episode}','object_group':assets[(subset,task)]}
           for subset in ('object-soft','spatial-soft')for task in range(3)for episode in range(10 if task==0 else 50)]
    specs += [{'dataset':'PhysTwin','case':case,'name':case,'object_group':object_group(case)}for case in phys_cases]
    rows=[];start=time.time()
    for i,spec in enumerate(specs):
        row=dict(spec)
        try:
            if spec['dataset']=='SoftVTBench':
                data=load_softvt_predictor(spec['subset'],spec['task'],spec['episode'])
            else:data=load_phystwin_predictor(spec['case'],max_nodes=64)
            row['predictor_digest']=predictor_digest(data)
            row['metrics']=measure_prefix(data['observed'],data['observed_mask'])
            row['observation_contract']=data['observation_contract']
        except Exception as e:
            row['metrics']={'eligible':False,'reason':'strict_adapter_or_metric_failure','error':f'{type(e).__name__}: {e}'}
        rows.append(row)
        if (i+1)%20==0:print('completed',i+1,'/',len(specs),'seconds',round(time.time()-start,2),flush=True)
    datasets={name:summary([r for r in rows if r['dataset']==name])for name in ('SoftVTBench','PhysTwin')}
    groups=defaultdict(list)
    for r in rows:groups[r['dataset']+'/'+r['object_group']].append(r)
    ranked={name:[{'case':r['name'],'object_group':r['object_group'],**{k:r['metrics'][k]for k in ('aggregate_rigid_removed_shape_RMS_mm','aggregate_total_motion_RMS_mm','aggregate_shape_to_total_RMS_ratio','shape_RMS_relative_to_initial_radius','eligible_frame_fraction')}}for r in sorted([r for r in rows if r['dataset']==name and r['metrics'].get('eligible')],key=lambda r:r['metrics']['aggregate_rigid_removed_shape_RMS_mm'],reverse=True)]for name in datasets}
    frame_records=[]
    for row in rows:
        frames=row['metrics'].pop('frames',[])
        reasons=defaultdict(int)
        for f in frames:
            if not f.get('eligible'):reasons[f['reason']]+=1
        row['metrics']['frame_ineligibility_reasons']=dict(reasons)
        frame_records.append({'dataset':row['dataset'],'case':row['name'],'frames':frames})
    frame_path=Path('runs/gaussian-boundary-flow-v02-audits')/(out.name+'-frames.json.gz')
    frame_path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(frame_path,'wt')as f:json.dump(frame_records,f,separators=(',',':'),allow_nan=False)
    result={'frame_diagnostics':{'local_path':str(frame_path),'sha256':hashlib.sha256(frame_path.read_bytes()).hexdigest(),'records':len(frame_records),'scope':'Per-frame metric summaries only, no position arrays; retained locally'},'scope':'Exposed prefix-only population signal audit; no new methods, future object targets, fresh episodes, force labels, or RGB decoded',
            'observations':'Privileged sparse SoftVT node positions and author processed PhysTwin 3D tracks; not raw-video observability',
            'metric':'Proper Kabsch to frame0 on each frame common finite valid track IDs; coordinate RMS in mm; frame0 excluded',
            'minimum_common_valid_points':6,'motion_threshold_mm':.1,'ratios':'RMS ratios, not variance explained; aggregate ratio uses point-coordinate weighted sums',
            'limitations':['Tracking noise and reconstruction error contribute to apparent nonrigid motion.','Changing visible subsets can affect framewise comparisons.','Strong observed deformation does not prove material/support identifiability.','Fresh episodes remain unopened; these rankings are development priorities, not test selection.','PhysTwin pickles are materialized by the quarantined adapter; suffix object arrays are discarded and never used by this analysis.','The 220 SoftVT episodes repeat three assets, not 220 independent object geometries.'],
            'dataset_summaries':datasets,'object_group_summaries':{k:summary(v)for k,v in sorted(groups.items())},'ranked_cases':ranked,'cases':rows,
            'elapsed_seconds':time.time()-start,'source_sha256':hashes_before,'sources_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes_before.items()),'environment':{'python':sys.version,'numpy':np.__version__},'softvt_suite_task_summaries':{f'{subset}/task{task}':summary([r for r in rows if r['dataset']=='SoftVTBench'and r['subset']==subset and r['task']==task])for subset in ('object-soft','spatial-soft')for task in range(3)}}
    out.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['PREFIX DEFORMATION INFORMATIVENESS — EXPOSED DATA ONLY','',result['metric'],result['observations'],'']
    for name,s in datasets.items():
        lines += [f'{name}: {s["eligible_cases"]}/{s["cases"]} eligible cases']
        for key,q in s['case_weighted_quantiles'].items():
            if q:lines += [f'  {key}: median {q["median"]:.6g}; p10-p90 {q["p10"]:.6g}-{q["p90"]:.6g}']
        lines += ['  Most shape-changing exposed cases (absolute rigid-removed RMS):']
        for r in ranked[name][:10]:lines += [f'    {r["case"]}: {r["aggregate_rigid_removed_shape_RMS_mm"]:.4f} mm; fraction {r["aggregate_shape_to_total_RMS_ratio"]}']
        lines += ['']
    lines += ['Object-group summaries:']
    for name,s in result['object_group_summaries'].items():
        q=s['case_weighted_quantiles']['aggregate_rigid_removed_shape_RMS_mm'];r=s['case_weighted_quantiles']['aggregate_shape_to_total_RMS_ratio']
        lines += [f'  {name}: {s["eligible_cases"]}/{s["cases"]} eligible; median shape RMS {None if q is None else q["median"]} mm; median ratio {None if r is None else r["median"]}']
    lines += ['',*result['limitations'],'','All 220 historically exposed SoftVT episodes and 22 PhysTwin cases were attempted.','See JSON for masks, frame eligibility, robust quantiles and all case records.','Full per-frame diagnostics are in the local compressed artifact identified by the JSON.']
    out.with_suffix('.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(datasets,indent=2))

if __name__=='__main__':main()
