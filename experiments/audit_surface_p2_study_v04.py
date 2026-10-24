"""Read-only posthoc scoring/binding audit; never calls a model or inference.

Execution is refused until all worker/controller PIDs/groups are absent. All
choice seals are validated before any target truth archive is opened. Frozen
source/dataflow and artifact timestamps support causality; this cannot prove
historical OS-level file access. Only compact scalar results are exported.
"""
from __future__ import annotations
import argparse,hashlib,json,os,resource,signal,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'runs/operator-dev/surface-p2-study-v04'
AUDIT=ROOT/'runs/operator-dev/surface-p2-study-v04-independent-audit-v01'
PROTOCOL='63db63c55ec3a10bd307b443b87386213f8df4c302ef2b4db3ce54a729719dbf'
ORIGINAL_PROTOCOL='7fe1d8a357d0dff6c6f183c9272de5101ac3e7cecd70472165485d3e658a2186'
ORIGINAL=ROOT/'runs/operator-dev/surface-p2-study-v03'
LAWS=('matching_linear','NH80_stress')
DESIGN=dict(schema='surface-p2-study-v04-independent-audit-v01',CPU=17,threads=1,GPU=False,
    wall_cap_s=900,RSS_cap_bytes=4*1024**3,expected_objects=16,expected_posteriors=608,
    expected_physical_records=320,expected_prefix_law_signals=32,
    score_absolute_tolerance=1e-14,score_relative_tolerance=1e-10,group_probability_atol=1e-14,group_first_moment_atol=1e-17,group_centered_second_moment_atol=1e-26,dense_unit_mean_atol=1e-11,dense_unit_covariance_atol=1e-12,
    no_forward_or_inference_calls=True,truth_read_gate='All32 prefix-bound seals verified before opening any truth archive',
    causal_scope='Frozen source/dataflow plus seal/hash/timestamp consistency; not OS-level historical read attestation',
    physics_scope='Recompute admissions from bound recorded kinematic/equilibrium/eigenpair scalars; no PDE or kinematic rerun')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()
def ensure(value,message):
    if not value:raise ValueError(message)


def same(expected,actual,label='value'):
    if isinstance(expected,dict):
        ensure(isinstance(actual,dict),'Missing mapping '+label)
        for k,v in expected.items():
            ensure(k in actual,'Missing '+label+'.'+k);same(v,actual[k],label+'.'+k)
    elif isinstance(expected,(list,tuple)):
        ensure(len(expected)==len(actual),'Length '+label)
        for i,(a,b)in enumerate(zip(expected,actual)):same(a,b,f'{label}[{i}]')
    elif isinstance(expected,(bool,str))or expected is None:ensure(expected==actual,'Mismatch '+label)
    else:ensure(bool(np.isfinite(expected)and np.isfinite(actual)and np.isclose(expected,actual,atol=1e-14,rtol=1e-10)),'Numerical mismatch '+label)


def fingerprint(observations):
    h=hashlib.sha256()
    for label,y in observations:h.update(label.encode());h.update(np.asarray(y,dtype='<f8').tobytes(order='C'))
    return h.hexdigest()


def stream(index,design):
    return dict(unit=np.random.default_rng(design['cohort_seed']).uniform(size=(16,2))[index],
        prefix_noise=20e-6*np.random.default_rng(design['prefix_noise_seed']).normal(size=(16,24))[index],
        probe_noise=20e-6*np.random.default_rng(design['probe_noise_seed']).normal(size=(16,3,24))[index],
        random_realization=np.random.default_rng(design['random_action_seed']).integers(3,size=(16,2))[index])


def require_absent(pid):
    ensure(not Path('/proc',str(pid)).exists(),'Owned PID still live')
    try:os.killpg(pid,0)
    except ProcessLookupError:return
    raise ValueError('Owned process group still live')


OLD_MANIFEST_SHA='9afed9ce5e80fbaf79e5e244a12332f6d8bc7cf1fe2317eccfffe009e62b2818'
OLD_STOP_SHA='6e8882f2b082618a9a8da688bfea7857de8b898b427c188ad6a94a8825eec36a'


def verify_origins(base,p,digest):
    data=p['old_data'];ensure(p['original_source_protocol_SHA256']==data['original_source_protocol_SHA256']==ORIGINAL_PROTOCOL,'Original source protocol')
    ensure(Path(data['original_data_root']).resolve()==ORIGINAL.resolve(),'Original physical namespace')
    ensure(sha(ORIGINAL/'protocol.json')==ORIGINAL_PROTOCOL,'Original protocol bytes')
    manifest_path=Path(data['original_data_manifest']['path']);ensure(manifest_path.resolve()==(ORIGINAL/'stopped-completed-manifest-LOCAL.json').resolve(),'Original manifest path')
    ensure(sha(manifest_path)==data['original_data_manifest']['SHA256']==OLD_MANIFEST_SHA,'Original immutable manifest')
    ensure(sha(ORIGINAL/'controlled-stop-ownership.json')==data['old_stop_ownership_SHA256']==OLD_STOP_SHA,'Original owned-stop binding')
    manifest=read(manifest_path)
    for name,h in manifest['files'].items():ensure(sha(ORIGINAL/name)==h,'Original stopped artifact changed')
    ensure(len(data['objects'])==16 and [r['index']for r in data['objects']]==list(range(16))and data['no_new_simulation_or_draw'],'All16 original memberships')
    for i,entry in enumerate(data['objects']):
        folder=base/f'object-{i:02d}-LOCAL'
        for source_key,hash_key,filename in [('truth_file','truth_SHA256','truth-LOCAL.npz'),('metadata_file','metadata_SHA256','truth-checkpoint-LOCAL.json')]:
            source=ROOT/entry[source_key]
            ensure(source.resolve()==(ORIGINAL/f'object-{i:02d}-LOCAL'/filename).resolve(),'Original object file mapping')
            expected=manifest['files'][str(source.relative_to(ORIGINAL))]
            ensure(sha(source)==entry[hash_key]==sha(folder/filename)==expected,'Byte-identical original data copy')
        same(dict(original_source_protocol_SHA256=ORIGINAL_PROTOCOL,new_numerical_protocol_SHA256=digest,original_data=entry,no_simulation_or_redraw=True),read(folder/'truth-origin-LOCAL.json'),'New numerical origin receipt')
    return dict(original_manifest_SHA256=OLD_MANIFEST_SHA,original_stop_SHA256=OLD_STOP_SHA,original_artifacts_verified=len(manifest['files']),objects_byte_identical=16)


def preconditions(base,protocol_digest,*,require_all_attempts=True):
    ensure(sha(base/'protocol.json')==protocol_digest,'Protocol hash')
    p=read(base/'protocol.json');e=read(base/'execution.json')
    ensure(e['completed']and e['protocol_SHA256']==protocol_digest and e['all_worker_groups_absent'],'Terminal owned execution required')
    ensure(len(e['records'])==16 and {r['slot']for r in e['records']}==set(range(16)),'All16 worker mappings')
    require_absent(e['pid'])
    for row in e['records']:
        ensure(row['reaped']and row['exit_code']==0,'Worker exit/reap');require_absent(row['pid'])
    for name,digest in p['source_SHA256'].items():ensure(sha(ROOT/name)==digest,'Frozen source changed '+name)
    object_paths=[];workers=[]
    for i in range(16):
        folder=base/f'object-{i:02d}-LOCAL';terminal=folder/'terminal-LOCAL.json';c=read(folder/'checkpoint.json')
        ensure(c['stage']=='terminal'and c['index']==i and c['protocol_SHA256']==protocol_digest and c['terminal_SHA256']==sha(terminal),'Terminal checkpoint binding')
        worker=read(base/f'worker-{i}-terminal.json');ensure(worker['slot']==i and worker['protocol_SHA256']==protocol_digest and worker['completed'],'Worker terminal binding')
        ensure(worker['pid']==next(r['pid']for r in e['records']if r['slot']==i),'Worker PID mapping')
        object_paths.append(terminal);workers.append(worker)
    # Read only JSON seals/prefix records here, never NPZ truth or target arrays.
    seals={}
    for i in range(16):
        folder=base/f'object-{i:02d}-LOCAL'
        for law in LAWS:
            prefix=folder/f'{law}-prefix-LOCAL.json';path=folder/f'{law}-choices-seal-LOCAL.json';s=read(path);r=read(prefix)
            ensure(s['protocol_SHA256']==protocol_digest and s['prefix_checkpoint_SHA256']==sha(prefix),'Seal prefix binding')
            ensure(s.get('prefix_raw_SHA256')==r.get('raw_SHA256'),'Seal raw prefix binding')
            if r.get('raw_SHA256'):ensure(sha(prefix.with_suffix('.npz'))==r['raw_SHA256'],'Sealed prefix archive hash')
            ensure(s['post_probe_outcomes_read']is False and s['truth_used_for_selection']is False,'Causal seal flags')
            ensure(prefix.stat().st_mtime_ns<=path.stat().st_mtime_ns,'Prefix artifact after seal')
            if r['qualified']:
                ensure(set(s['choices'])==set(p['design']['actions']),'Sealed families')
                for family,actions in p['design']['actions'].items():
                    choice=s['choices'][family];same(actions,choice['actions'],'sealed action labels')
                    ensure(len(choice['scores'])==3,'Three action scores')
                    for method,key in [('proposed','expected_linear_target_MSE_reduction_m2'),('information','Gaussian_observation_information_nats')]:
                        scores=np.array([v[key]for v in choice['scores']],float)
                        ensure(np.isfinite(scores).all()and choice[method]==int(scores.argmax()),'Sealed argmax mismatch')
                for action in sum(p['design']['actions'].values(),[]):
                    for rep in range(3):
                        joint=folder/f'{law}-{action}-r{rep}-LOCAL.json'
                        if joint.exists():ensure(joint.stat().st_mtime_ns>=path.stat().st_mtime_ns,'Joint artifact before seal')
            else:ensure(s['choices']is None,'Unqualified prefix chose action')
            seals[i,law]=s
    # All608 numerical attempts must be closed before any actual outcome reads.
    for i in range(16):
        folder=base/f'object-{i:02d}-LOCAL'
        for law in LAWS:
            paths=[folder/f'{law}-prefix-LOCAL.json']+[folder/f'{law}-{action}-r{rep}-LOCAL.json'for actions in p['design']['actions'].values()for action in actions for rep in range(3)]
            if require_all_attempts:
                for path in paths:ensure(path.is_file(),'All608 numerical attempts must be closed')
    # Actual target-derived terminal scores are deserialized only after every seal.
    origin=verify_origins(base,p,protocol_digest)if 'old_data'in p else None
    objects=[]
    for i,path in enumerate(object_paths):
        t=read(path);ensure(t['index']==i and t['protocol_SHA256']==protocol_digest and t['retained'],'Object identity')
        same([dict(index=i,completed=t['completed'])],workers[i]['records'],'worker object');objects.append(t)
    return p,e,objects,seals,origin


def commit(path,context):
    meta=read(path.with_suffix('.commit.json'))
    ensure(meta['context']==context and meta['raw_SHA256']==sha(path),'Truth field commit binding')


def physical_valid(d,prefix):
    eq=d['equilibrium'];st=d['stability']
    equilibrium=bool(eq['exact_clamp']and eq['free_residual_N']<=eq['tolerance_N']and eq['force_balance_error_N']<=eq['force_balance_tolerance_N']and eq['moment_balance_error_Nm']<=eq['moment_balance_tolerance_Nm'])
    stable=bool(st['minimum_eigenvalue_estimate_N_m']-st['eigenpair_residual_N_m']>0 and st['relative_eigenpair_residual']<=1e-6)
    same(equilibrium,eq['passed'],'equilibrium gate');same(stable,st['passed'],'stability gate')
    material=bool(d['minimum_J']>=.8 and d['maximum_absolute_symmetric_strain_eigenvalue']<=.05 and equilibrium and stable)
    excitation=bool(d['nonrigid_RMS_m']>=100e-6 and d['nonrigid_fraction']>=.3)
    same(material,d['material_valid'],'material gate');same(excitation,d['excitation_passed'],'excitation gate');same(prefix,d['prefix_excitation_required'],'prefix role')
    valid=bool(material and(not prefix or excitation));same(valid,d['valid'],'physical gate')
    return valid


def score_arrays(z,target,unit):
    q=z['unit_quantiles'];covered=(unit>=q[:,0])&(unit<=q[:,2])
    return dict(resolved=True,posterior_RMS_m=float(np.linalg.norm(z['mean']-target)/np.sqrt(target.size)),
        MAP_RMS_m=float(np.linalg.norm(z['MAP_prediction']-target)/np.sqrt(target.size)),
        mean_predictive_SD_m=float(np.linalg.norm(z['SD'])/np.sqrt(target.size)),
        E_90percent_covered=float(covered[0]),K_90percent_covered=float(covered[1]),
        E_unit_interval_width=float(q[0,2]-q[0,0]),K_unit_interval_width=float(q[1,2]-q[1,0]))


def verify_numerical_flags(diag):
    for axis in ('inner','outer'):
        for row in diag['successive'][axis]:
            if row['passed']:
                ensure(row.get('log_evidence_difference',0)<=.01 and row.get('maximum_CDF_difference',0)<=.005 and row.get('maximum_unit_quantile_difference',0)<=.002 and row.get('maximum_unit_mean_difference',0)<=.001,'Reported refinement passes but scalar gate fails')
    ad=diag['independent_adaptive']
    if 'log_evidence_difference'in ad:
        recomputed=bool(ad['log_evidence_difference']<=.01 and ad['maximum_CDF_error']<=.005 and ad['adaptive_refinement_log_evidence_difference']<=.01 and ad['adaptive_refinement_maximum_CDF_difference']<=.005 and max(ad['maximum_QUADPACK_relative_error_estimates'].values())<=1e-5)
        same(recomputed,ad['passed'],'Independent adaptive scalar gates')
    return all(diag['successive'][axis][-1]['passed']for axis in ('inner','outer'))and ad['passed']


def verify_grouped_archive(z):
    probability=np.asarray(z['quadrature_probability']);unit=np.asarray(z['unit_grid']);amplitude=np.asarray(z['inverse_amplitude']);ratio=np.asarray(z['log_ratio'])
    gz=np.asarray(z['group_log_ratio']);rho=np.asarray(z['group_probability']);mean=np.asarray(z['group_amplitude_mean']);variance=np.asarray(z['group_amplitude_variance'])
    ensure(probability.ndim==1 and len(probability)>0 and np.all(probability>=0)and abs(float(probability.sum())-1)<=1e-10,'Normalized atom weights')
    ensure(unit.shape==(len(probability),2)and amplitude.shape==ratio.shape==probability.shape,'Full hypothesis dimensions')
    ensure(gz.ndim==1 and len(gz)>0 and np.all(np.diff(gz)>0)and rho.shape==mean.shape==variance.shape==gz.shape,'Ordered grouped dimensions')
    ensure(np.all(rho>=0)and np.all(mean>0)and np.all(variance>=0)and abs(float(rho.sum())-1)<=1e-10,'Positive grouped moments')
    row=np.searchsorted(gz,ratio);ensure(np.all(row<len(gz))and np.array_equal(gz[row],ratio),'Exact common ratio labels')
    weight=np.bincount(row,weights=probability,minlength=len(gz))
    first=np.bincount(row,weights=probability*amplitude,minlength=len(gz))
    central=np.bincount(row,weights=probability*(amplitude-mean[row])**2,minlength=len(gz))
    ensure(np.allclose(weight,rho,rtol=1e-10,atol=1e-14),'Grouped probability agrees with full atoms')
    ensure(np.allclose(first,rho*mean,rtol=1e-10,atol=1e-17),'Grouped first moments agree with full atoms')
    ensure(np.allclose(central,rho*variance,rtol=1e-10,atol=1e-26),'Grouped centered second moments agree with full atoms')
    decoded=np.column_stack(((-np.log(amplitude)-np.log(2000.))/np.log(4.),(ratio-np.log(amplitude)-np.log(5.))/np.log(6.)))
    ensure(np.allclose(unit,decoded,rtol=0,atol=1e-12),'Hypotheses retain original material coordinates')
    return dict(groups=len(gz),atoms=len(probability),zero_mass_groups=int(np.sum(rho==0)))


def posterior(path,observations,target,unit,*,engine,recipe,model):
    record=read(path)
    same(dict(numerical_engine=engine,recipe_SHA256=recipe,model_fingerprint=model,observations_SHA256=fingerprint(observations)),record,'posterior context')
    diag=record['diagnostics'];same(record['qualified'],diag['qualified'],'posterior qualification')
    if engine=='conditional_amplitude_PWL_v02':
        computed=verify_numerical_flags(diag)
        same(bool(computed),record['qualified'],'quadrature conjunction')
    if record.get('raw_SHA256'):
        archive=path.with_suffix('.npz');ensure(sha(archive)==record['raw_SHA256'],'Posterior archive hash')
        with np.load(archive,allow_pickle=False)as archive_data:
            z={key:archive_data[key]for key in archive_data.files}
            ensure(all(np.isfinite(z[k]).all()for k in z),'Finite posterior archive')
            ensure(z['mean'].shape==z['SD'].shape==z['MAP_prediction'].shape==target.shape==(96,),'Target prediction shape')
            ensure(z['unit_quantiles'].shape==(2,3)and np.all(np.diff(z['unit_quantiles'],axis=1)>=0),'Ordered quantiles')
            ensure(np.all(z['SD']>=0),'Nonnegative predictive SD')
            cdf=z['unit_CDF'];ensure(cdf.shape==(2,33)and np.all(cdf[:,0]==0)and np.all(cdf[:,-1]==1)and np.all(np.diff(cdf,axis=1)>=-1e-12)and cdf.min()>=0 and cdf.max()<=1+32*np.finfo(float).eps,'Continuous CDF archive monotonicity/domain')
            prefix=path.name.endswith('-prefix-LOCAL.json')
            ensure(record['archive_kind']==('prefix_atoms_and_groups'if prefix else'joint_sufficient_statistics'),'Explicit posterior archive kind')
            ensure(z['unit_mean'].shape==(2,)and z['unit_covariance'].shape==(2,2)and z['MAP_unit'].shape==(2,)and z['MAP_log_likelihood'].shape==(),'Sufficient statistics shapes')
            covariance=z['unit_covariance'];ensure(np.allclose(covariance,covariance.T,atol=1e-15,rtol=1e-12)and np.linalg.eigvalsh(covariance).min()>=-1e-14,'Posterior intrinsic covariance PSD')
            ensure(np.all(z['unit_quantiles']>=0)and np.all(z['unit_quantiles']<=1),'Quantiles remain inside original prior')
            if prefix:
                verify_grouped_archive(z)
                rawmean=z['quadrature_probability']@z['unit_grid'];center=z['unit_grid']-rawmean;rawcov=center.T@(z['quadrature_probability'][:,None]*center)
                ensure(np.allclose(rawmean,z['unit_mean'],rtol=0,atol=1e-11)and np.allclose(rawcov,z['unit_covariance'],rtol=1e-10,atol=1e-12),'Prefix dense intrinsic moments agree with grouped sufficient statistics')
                likelihood=z['log_likelihood'];ensure(likelihood.shape==(len(z['quadrature_probability']),),'Full atom likelihood shape')
                index=int(np.argmax(likelihood));ensure(np.allclose(z['MAP_unit'],z['unit_grid'][index],rtol=0,atol=1e-12)and np.isclose(float(z['MAP_log_likelihood']),likelihood[index],rtol=1e-12,atol=1e-10),'Prefix approximate-MAP node binding')
            else:ensure(set(z)=={'mean','SD','unit_mean','unit_covariance','unit_CDF','unit_quantiles','MAP_unit','MAP_prediction','MAP_log_likelihood'},'Joint archive stores exactly sufficient statistics')
            scored=score_arrays(z,target,unit)if record['qualified']else dict(resolved=False)
    else:
        ensure(not path.with_suffix('.npz').exists()and not record['qualified'],'Missing posterior archive');scored=dict(resolved=False)
    return scored,record['qualified']


def average(rows):
    if not all(r['resolved']for r in rows):return dict(resolved=False)
    return dict(resolved=True,**{k:float(np.mean([r[k]for r in rows]))for k in rows[0]if k!='resolved'})


def policies(choices,joints,random_realization,families):
    result={}
    for fidx,family in enumerate(families):
        rows=joints[family];choice=choices[family]
        select=lambda indexes:average([rows[a][r]for a in indexes for r in range(3)])
        result[family]=dict(proposed=select([choice['proposed']]),information=select([choice['information']]),uniform_random=select(range(3)),ancillary_random_realization=select([int(random_realization[fidx])]),**{f'fixed_{i}':select([i])for i in range(3)})
    return result


def bootstrap(values,seed):
    x=np.asarray(values,float);ensure(x.shape==(16,)and np.isfinite(x).all(),'All16 object differences')
    b=np.random.default_rng(seed).integers(16,size=(10000,16));means=x[b].mean(axis=1)
    return dict(objects=16,mean=float(x.mean()),CI95=np.percentile(means,[2.5,97.5]).tolist(),bootstrap_seed=seed,bootstrap_repetitions=10000)


def aggregate(objects,design):
    output={}
    for law in LAWS:
        rows=[o['laws'][law]for o in objects]
        mapping={'prefix_mean_RMS_m':'posterior_RMS_m','prefix_MAP_mean_RMS_m':'MAP_RMS_m','prefix_E_interval_coverage':'E_90percent_covered','prefix_K_interval_coverage':'K_90percent_covered','prefix_mean_predictive_SD_m':'mean_predictive_SD_m'}
        report={k:float(np.mean([r['prefix'][v]for r in rows]))for k,v in mapping.items()}
        report.update(prior_mean_RMS_m=float(np.mean([r['prior_RMS_m']for r in rows])),known_parameter_linear_oracle_mean_RMS_m=float(np.mean([r['known_true_parameter_linear_oracle_RMS_m']for r in rows])),families={})
        mapping={'mean_target_RMS_m':'posterior_RMS_m','mean_MAP_RMS_m':'MAP_RMS_m','E_interval_coverage':'E_90percent_covered','K_interval_coverage':'K_90percent_covered','mean_predictive_SD_m':'mean_predictive_SD_m','mean_E_unit_interval_width':'E_unit_interval_width','mean_K_unit_interval_width':'K_unit_interval_width'}
        for family in design['actions']:
            names=['proposed','information','uniform_random','fixed_0','fixed_1','fixed_2']
            fr={name:{k:float(np.mean([r['policies'][family][name][v]for r in rows]))for k,v in mapping.items()}for name in names}
            fr['paired_baseline_minus_proposed_RMS_m']={b:bootstrap([r['policies'][family][b]['posterior_RMS_m']-r['policies'][family]['proposed']['posterior_RMS_m']for r in rows],design['bootstrap_seed'])for b in names[1:]}
            fr['positive_CI_against_all_fixed']=all(fr['paired_baseline_minus_proposed_RMS_m'][f'fixed_{i}']['CI95'][0]>0 for i in range(3))
            report['families'][family]=fr
        output[law]=report
    return output


def run(base=LOCAL,digest=PROTOCOL):
    start=time.monotonic();p,e,terminal,seals,origin=preconditions(base,digest)
    ensure('old_data'in p and origin is not None,'Original-data reuse admission required')
    design=p['design'];cache=read(base/'cache-ready.json')
    ensure(cache['protocol_SHA256']==digest and cache['qualified'],'Shared reference context')
    ensure(sha(base/'prior-predictive-LOCAL.npz')==cache['prior_SHA256'],'Prior archive binding')
    model=cache['model_fingerprint'];ensure(model==read(ORIGINAL/'cache-ready.json')['model_fingerprint'],'Original physical model fingerprint preserved');engine=p['numerical_engine']
    ensure(engine=='conditional_amplitude_PWL_v02','This audit recipe targets the actually released conditional engine')
    recipe=lambda retain:hashlib.sha256(json.dumps(dict(numerical_engine=engine,plan=p['amplitude_plan'],retain_hypotheses=retain),sort_keys=True).encode()).hexdigest()
    ensure(len(seals)==32,'All seals before truth reads')
    with np.load(base/'prior-predictive-LOCAL.npz')as z:prior=z['mean']
    ensure(prior.shape==(96,)and np.isfinite(prior).all(),'Finite prior predictive mean')
    records=0;qualified=0;physical=[];prefix_metrics=[];recomputed=[];posterior_pins={};truth_pins={}
    for i in range(16):
        folder=base/f'object-{i:02d}-LOCAL';meta=read(folder/'truth-checkpoint-LOCAL.json');truthpath=folder/'truth-LOCAL.npz'
        # The first truth archive access occurs only after all32 seals passed.
        ensure(sha(truthpath)==meta['raw_SHA256'],'Truth archive binding');truth_pins[str(truthpath.relative_to(base))]=sha(truthpath)
        expected=stream(i,design)
        with np.load(truthpath,allow_pickle=False)as z:truth={k:z[k]for k in z.files}
        for k,v in expected.items():ensure(np.array_equal(v,truth[k]),'Original seeded truth/noise stream '+k)
        theta=np.exp(np.log([2000.,5.])+expected['unit']*np.log([4.,6.]))
        same(theta.tolist(),truth['theta'].tolist(),'Original physical draw')
        labels=meta['labels'];ensure(labels==list(p['force_SHA256']),'Exact truth load labels')
        for law in LAWS:ensure(truth[law].shape==(10,24)and np.isfinite(truth[law]).all(),'Truth response dimensions')
        context=dict(protocol_SHA256=ORIGINAL_PROTOCOL,index=i,unit_SHA256=array_sha(expected['unit']),force_SHA256=p['force_SHA256'])
        physics_folder=ORIGINAL/f'object-{i:02d}-LOCAL'
        linear=physics_folder/'linear-fields-LOCAL.npz';commit(linear,context)
        with np.load(linear)as z:ensure(np.array_equal(z['unit'],expected['unit'])and z['fields'].shape==(10,2673,3)and np.isfinite(z['fields']).all(),'Linear field checkpoint')
        for label in labels:
            path=physics_folder/f'truth-load-{label}-LOCAL.npz';commit(path,dict(context,label=label))
            with np.load(path)as z:
                ensure(np.array_equal(z['unit'],expected['unit'])and str(z['force_SHA256'])==p['force_SHA256'][label],'Nonlinear field context')
                ensure(z['NH80_field'].shape==(2673,3)and np.isfinite(z['NH80_field']).all(),'Nonlinear field checkpoint')
                diags=json.loads(str(z['diagnostics_json']))
            for law in LAWS:
                ensure(diags[law]==meta['physics'][law][label],'Physical diagnostic bound to load archive')
                d=diags[law];valid=physical_valid(d,label==design['prefix']);physical.append(dict(valid=valid,minimum_J=d['minimum_J'],strain=d['maximum_absolute_symmetric_strain_eigenvalue']))
                if label==design['prefix']:prefix_metrics.append(dict(valid=valid,RMS_m=d['nonrigid_RMS_m'],fraction=d['nonrigid_fraction']))
        physical_object=all(d['valid']for d in meta['physics']['matching_linear'].values())and all(d['valid']for d in meta['physics']['NH80_stress'].values())
        same(physical_object,meta['all_physical_valid'],'Truth physical aggregate');same(physical_object,terminal[i]['physical_valid'],'Object physical aggregate')
        obj=dict(laws={})
        for law in LAWS:
            target=truth[law][[labels.index(l)for l in design['target_labels']]].reshape(-1)
            oracle=truth['matching_linear'][[labels.index(l)for l in design['target_labels']]].reshape(-1)
            prefix=truth[law][labels.index(design['prefix'])]+expected['prefix_noise'];path=folder/f'{law}-prefix-LOCAL.json'
            value,ok=posterior(path,[(design['prefix'],prefix)],target,expected['unit'],engine=engine,recipe=recipe(True),model=model);records+=1;qualified+=ok;posterior_pins[str(path.relative_to(base))]=sha(path)
            r=dict(prefix=value,prior_RMS_m=float(np.linalg.norm(prior-target)/np.sqrt(96)),known_true_parameter_linear_oracle_RMS_m=float(np.linalg.norm(oracle-target)/np.sqrt(96)),prefix_whitened_oracle_RMS=float(np.linalg.norm((truth[law][labels.index(design['prefix'])]-truth['matching_linear'][labels.index(design['prefix'])])/20e-6)/np.sqrt(24)),target_whitened_oracle_RMS=float(np.linalg.norm((oracle-target)/20e-6)/np.sqrt(96)),policies={},all_joint_qualified=bool(ok),posteriors=1)
            if ok:
                joints={}
                for family,actions in design['actions'].items():
                    joints[family]=[]
                    for action in actions:
                        actionrows=[]
                        for rep in range(3):
                            y=truth[law][labels.index(action)]+expected['probe_noise'][rep];path=folder/f'{law}-{action}-r{rep}-LOCAL.json'
                            score,passed=posterior(path,[(design['prefix'],prefix),(action,y)],target,expected['unit'],engine=engine,recipe=recipe(False),model=model)
                            records+=1;qualified+=passed;r['posteriors']+=1;r['all_joint_qualified']&=passed;actionrows.append(score);posterior_pins[str(path.relative_to(base))]=sha(path)
                        joints[family].append(actionrows)
                r['policies']=policies(seals[i,law]['choices'],joints,expected['random_realization'],list(design['actions']))
            same(r,terminal[i]['laws'][law],f'object{i}.{law}')
            obj['laws'][law]=r
        resolved=all(r['prefix']['resolved']and r['all_joint_qualified']for r in obj['laws'].values());same(resolved,terminal[i]['numerically_resolved'],'Object numerical aggregate')
        recomputed.append(obj)
    ensure(len(physical)==320 and len(prefix_metrics)==32,'Physical denominators')
    ensure(records==sum(t['laws'][law]['posteriors']for t in terminal for law in LAWS),'Posterior record count')
    actual={f for folder in base.glob('object-*-LOCAL')for f in folder.glob('*-LOCAL.json')if ('-prefix-LOCAL' in f.name or any('-'+action+'-r' in f.name for actions in design['actions'].values()for action in actions))}
    ensure(len(actual)==records,'Unexpected or uncounted posterior record')
    admitted=all(t['completed']and t['physical_valid']and t['numerically_resolved']for t in terminal)
    summary=read(base/'summary.json');same(dict(protocol_SHA256=digest,execution_SHA256=sha(base/'execution.json'),objects=16,all16_retained=True,physically_valid_objects=sum(t['physical_valid']for t in terminal),numerically_resolved_objects=sum(t['numerically_resolved']for t in terminal),all_comparison_inputs_qualified=admitted),summary,'Aggregate context')
    laws=aggregate(recomputed,design)if admitted else {}
    same(laws,summary['laws'],'Independent aggregate')
    if admitted:ensure(records==qualified==608,'All608 qualified numerical records')
    out=dict(schema=DESIGN['schema'],completed=True,audit_passed=True,protocol_SHA256=digest,source_SHA256={str(Path(__file__).relative_to(ROOT)):sha(__file__)},execution_SHA256=sha(base/'execution.json'),summary_SHA256=sha(base/'summary.json'),all17_owned_PID_and_groups_absent=True,objects_retained=16,terminal_checkpoints_bound=16,prefix_seals_bound_before_any_truth_read=32,posterior_records_verified=records,expected_posteriors=608,qualified_posteriors=qualified,physical_records_verified=320,physical_records_valid=sum(d['valid']for d in physical),prefix_law_signals_verified=32,prefix_objects=16,minimum_prefix_nonrigid_RMS_m=min(d['RMS_m']for d in prefix_metrics),minimum_prefix_nonrigid_fraction=min(d['fraction']for d in prefix_metrics),minimum_recorded_J=min(d['minimum_J']for d in physical),maximum_recorded_principal_strain=max(d['strain']for d in physical),all_comparison_inputs_qualified=admitted,independent_scores_and_policy_averages_agree=True,bootstrap_recomputed=True if admitted else False,causal_scope=DESIGN['causal_scope'],physical_scope=DESIGN['physics_scope'],all_failures_retained=True,no_forward_or_inference_rerun=True,original_data_binding=origin,no_population_or_real_calibration_claim=True,laws=laws,wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,raw_arrays_LOCAL=True)
    return out,dict(posterior_json_SHA256=posterior_pins,truth_archive_SHA256=truth_pins)


def audit_sources():
    names=['experiments/audit_surface_p2_study_v04.py','tests/test_audit_surface_p2_study_v04.py']
    return {name:sha(ROOT/name)for name in names}


def main():
    recipe=read(AUDIT/'protocol.json')
    ensure(recipe['source_SHA256']==audit_sources()and recipe['study_protocol_SHA256']==PROTOCOL and recipe['design']==DESIGN,'Prospective audit source/recipe mismatch')
    ensure(os.sched_getaffinity(0)=={17}and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Audit CPU17/noGPU guard')
    ensure(all(os.environ.get(k)=='1'for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']),'Audit single-thread guard')
    resource.setrlimit(resource.RLIMIT_AS,(DESIGN['RSS_cap_bytes'],DESIGN['RSS_cap_bytes']))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Frozen audit wall cap')));signal.alarm(DESIGN['wall_cap_s'])
    ensure(not(AUDIT/'result.json').exists()and not(AUDIT/'failed.json').exists(),'No implicit audit overwrite/retry')
    AUDIT.mkdir(exist_ok=True,parents=True)
    try:
        result,pins=run();(AUDIT/'input-pins-LOCAL.json').write_text(json.dumps(pins,indent=2)+'\n');(AUDIT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    except BaseException as exc:
        (AUDIT/'failed.json').write_text(json.dumps(dict(completed=False,audit_passed=False,error=type(exc).__name__+': '+str(exc),protocol_SHA256=PROTOCOL,all16_denominator_preserved=True,raw_arrays_LOCAL=True),indent=2)+'\n');raise
    print(json.dumps({k:v for k,v in result.items()if k not in ['laws','source_SHA256']}))


if __name__=='__main__':main()
