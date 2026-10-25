"""Independent fixed32 noisy-poke audit; no posterior/PDE/GPU replay.

All closed phases and32 prefix-only choice seals precede any actual held
array or target-derived score read. Numerical refinement/oracle scalar gates
are recomputed; posterior integrals and category moments are bound producer
witnesses. Clean metrics and every policy/random aggregation are independent.
"""
import json,os,resource,time
import numpy as np
from experiments import run_noisy_poke_benchmark_v01 as run
from experiments import audit_fixture_balanced_decision_screen_v01 as choice_audit
from experiments import audit_layout_grouped_reference_cost_v01 as numerical
from experiments import audit_layout_grouped_allactions_v01 as actions_audit
SOURCE='experiments/audit_noisy_poke_benchmark_v01.py'
TEST='tests/test_audit_noisy_poke_benchmark_v01.py'
LIMITS=dict(CPU=16,threads=1,GPU=False,wall_s=3600,RSS_bytes=4*1024**3,external_owned_watchdog_required=True)
need=choice_audit.need


def guard(start):
    need(set(os.sched_getaffinity(0))=={16}and os.environ.get('CUDA_VISIBLE_DEVICES')==''and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'Audit CPU16/noGPU/one-thread')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=LIMITS['RSS_bytes']:raise MemoryError('Audit4GiB')
    if time.monotonic()-start>=LIMITS['wall_s']:raise TimeoutError('Audit3600s')
def bound(rec):
    path=run.ROOT/rec['path'];need(run.record(path)==rec,'Exact artifact byte commitment');return path

def equal(a,b):
    if isinstance(a,dict):return isinstance(b,dict)and set(a)==set(b)and all(equal(v,b[k])for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list)and len(a)==len(b)and all(equal(v,w)for v,w in zip(a,b))
    if isinstance(a,(float,np.floating)):return isinstance(b,(int,float,np.floating))and bool(np.isclose(a,b,rtol=1e-12,atol=1e-30))
    return a==b


def metadata_barrier():
    digest=run.sha(run.LOCAL/'protocol.json');p=run.validate(digest)
    need({SOURCE,TEST}<=set(p['source_SHA256']),'Prospectively frozen independent audit/test')
    # These functions open only selection metadata and hash-bound commitments,
    # never actual target arrays or target-derived evaluation result JSONs.
    phases={name:run.completed_phase(name,digest)for name in('choices','checkpoint','remaining')};seals=run.all_seals_gate(digest);inputs=run.inputs_gate(digest)
    need(len(seals)==32 and[v['index']for v in seals]==list(range(32)),'All32 causal seals before held results')
    need(sum(e['wall_s']for e in phases.values())<=p['resources']['global_wall_s'],'Combined global wall cap')
    for name,e in phases.items():
        need(e.get('peak_aggregate_RSS_bytes',0)<=p['resources']['aggregate_RSS_bytes'],'Owned aggregate RSS')
        for w in e['workers']:need(w.get('peak_RSS_bytes',0)<=p['resources']['worker_RSS_bytes']and w.get('peak_owned_VRAM_bytes',0)<=p['resources']['owned_VRAM_bytes'],'Owned host/GPU caps')
    trials=[]
    for name in('checkpoint','remaining'):
        for worker in phases[name]['workers']:
            value=run.read(bound(worker['result']))
            for entry in value['trials']:
                index=entry['index'];need(entry['record']==run.record(run.LOCAL/f'trial-{index:02d}'/'evaluation'/'result-LOCAL.json'),'Canonical evaluation result commitment');trials.append(entry)
    trials.sort(key=lambda v:v['index']);need([v['index']for v in trials]==list(range(32)),'All32 evaluation slots before score deserialization')
    return p,digest,phases,seals,inputs,trials


def check_posterior(value,directory,observations,p,*,prefix=False):
    context=dict(observations_fingerprint=run.observations_fingerprint(observations),reference_plan=run.CONTEXT_PLAN,target_labels=[s['id']for s in run.design.physics.measurement_specs()[12:]],sigma_m=p['sigma_m'])
    refs=value.get('categories',[]);need(len(refs)==4 and[v['layout']for v in refs]==list(range(4)),'All4 category references retained')
    all_ref=True
    for k,entry in enumerate(refs):
        need(entry['record']==run.record(directory/f'category-{k}-LOCAL.json'),'Exact category reference path');v=run.read(bound(entry['record']));choice_audit.reference_metadata(v);check=numerical.reference_checks(v)
        expected=dict(context,model_fingerprint=p['origin']['models'][k]['model_fingerprint']);need(v['reference_context']==expected,'Actual observation and frozen full-model fingerprint context')
        need(entry['qualified']is check['passed'],'Category admission from exact scalar gates');all_ref=all_ref and check['passed']
    banks=value.get('decision_levels',[]);need([(v['outer'],v['amplitude'])for v in banks]in([],list(run.base.DECISION_LEVELS)),'All5 banks or explicit early reference negative')
    bank_checks={}
    for entry in banks:
        o,a=entry['outer'],entry['amplitude'];need(entry['record']==run.record(directory/f'decision-{o}-{a}-LOCAL.json'),'Canonical bank path');v=run.read(bound(entry['record']));need(entry['checks']==v['checks'],'Bank checkpoint equality');bank_checks[o,a]=numerical.decision_checks(dict(reference_checks=v['checks']))['passed']
    bank_pass=bool(len(banks)==5 and all(bank_checks[key]for key in run.base.old.VARIANTS));mixture=value.get('mixture_interval_checks',{})
    mix_pass=False
    if mixture:
        need(mixture.get('levels')==[[512,512],[512,256],[256,512]]and mixture.get('weights_from_continuous_category_logZ')is True and mixture.get('CDF_gate')==.005 and mixture.get('unit_quantile_gate')==.002,'Continuous full-mixture refinement prescription')
        mix_pass=bool(numerical.finite_le(mixture.get('maximum_CDF_difference'),.005)and numerical.finite_le(mixture.get('maximum_unit_quantile_difference'),.002));need(mixture['passed']is mix_pass,'Mixture refinement scalar gates')
    density=bool(all_ref and bank_pass and mix_pass)
    if density:need(value.get('snapshot')==run.record(directory/'posterior-LOCAL.npz'),'Qualified density snapshot commitment')
    if not prefix:need(value.get('qualified')is density,'Joint posterior whole numerical admission')
    return dict(density_qualified=density,references=4,banks=len(banks))


def snapshot(path):
    with np.load(path,allow_pickle=False)as z:
        expected=dict(mean_m=(48,),SD_m=(48,),unit_mean=(2,),layout_probability=(4,),material_quantiles=(2,2),material_unit_CDF=(2,33),material_unit_quantiles=(2,2))
        need(set(z.files)==set(expected),'Exact posterior snapshot schema');out={k:z[k].copy()for k in expected}
    need(all(out[k].shape==shape and np.isfinite(out[k]).all()for k,shape in expected.items()),'Finite snapshot dimensions')
    need(np.all(out['SD_m']>=0)and np.all(out['layout_probability']>=0)and abs(out['layout_probability'].sum()-1)<=32*np.finfo(float).eps,'Positive posterior moments/category mass')
    q=out['material_unit_quantiles'];cdf=out['material_unit_CDF'];need(np.all((q>=0)&(q<=1))and np.all(q[:,0]<=q[:,1])and np.all((cdf>=0)&(cdf<=1+32*np.finfo(float).eps))and np.all(np.diff(cdf,axis=1)>=-1e-12),'Continuous intrinsic quantile/CDF sanity')
    physical=np.exp(np.log(run.design.BOUNDS[:,0,None])+run.design.LOG_WIDTH[:,None]*q);need(np.allclose(physical,out['material_quantiles'],rtol=1e-13,atol=0),'Physical material quantiles derived from log prior')
    return out


def metrics(s,truth,parameters):
    error=(s['mean_m']-truth)**2;variance=s['SD_m']**2
    field=dict(MSE_m2=float(error.mean()),RMSE_m=float(np.sqrt(error.mean())),posterior_variance_m2=float(variance.mean()),posterior_RMS_SD_m=float(np.sqrt(variance.mean())),moment_band_2SD_inclusion_fraction=float(np.mean((truth>=s['mean_m']-2*s['SD_m'])&(truth<=s['mean_m']+2*s['SD_m']))),moment_band_2SD_mean_width_m=float(np.mean(4*s['SD_m'])),moment_bands_have_no_nominal_coverage_level=True)
    q=s['material_quantiles'];tu=(np.log(parameters)-np.log(run.design.BOUNDS[:,0]))/run.design.LOG_WIDTH
    material=dict(central90_inclusion=((parameters>=q[:,0])&(parameters<=q[:,1])).tolist(),central90_width=(q[:,1]-q[:,0]).tolist(),unit_squared_error=((s['unit_mean']-tu)**2).tolist(),interpretation='Fixed8interior design descriptive interval inclusion; not SBC/population90%calibration')
    return dict(field=field,material=material,layout_probability=s['layout_probability'].tolist())


def policies(before,after,adaptive,fixed):
    out={}
    for name,ids in(('adaptive',[adaptive]),('fixed',[fixed]),('uniform_random',list(range(8)))):
        r={k:float(np.mean([after[i][k]for i in ids]))for k in('MSE_m2','RMSE_m','posterior_variance_m2','posterior_RMS_SD_m','moment_band_2SD_inclusion_fraction','moment_band_2SD_mean_width_m')};r['RMSE_m']=float(np.sqrt(r['MSE_m2']));r['posterior_RMS_SD_m']=float(np.sqrt(r['posterior_variance_m2']));r.update(realized_MSE_improvement_m2=before['MSE_m2']-r['MSE_m2'],variance_down_error_up=bool(r['posterior_variance_m2']<before['posterior_variance_m2']and r['MSE_m2']>before['MSE_m2']),moment_bands_have_no_nominal_coverage_level=True);out[name]=r
    out['paired_adaptive_minus_fixed_MSE_m2']=out['adaptive']['MSE_m2']-out['fixed']['MSE_m2'];out['paired_adaptive_minus_random_MSE_m2']=out['adaptive']['MSE_m2']-out['uniform_random']['MSE_m2'];out['uniform_random_interpretation']='Exact mean8action losses for this frozenprefix+8independentpoke-noise realization; not noise-integrated or population expectation'
    return out


def material_policies(before,after,adaptive,fixed):
    out={}
    for name,ids in(('adaptive',[adaptive]),('fixed',[fixed]),('uniform_random',list(range(8)))):
        inclusion=np.mean([after[i]['central90_inclusion']for i in ids],axis=0);width=np.mean([after[i]['central90_width']for i in ids],axis=0);error=np.mean([after[i]['unit_squared_error']for i in ids],axis=0)
        out[name]=dict(central90_inclusion=inclusion.tolist(),central90_width=width.tolist(),unit_squared_error=error.tolist(),narrower_but_larger_parameter_error=((width<before['central90_width'])&(error>before['unit_squared_error'])).tolist(),fixedcase_descriptive_not_nominal_calibration=True)
    return out


def aggregate(rows):
    need(len(rows)==32 and sorted((r['index'],r['replicate'],r['object_index'])for r in rows)==[(v['index'],v['replicate'],v['object_index'])for v in run.design.trials()],'All32 exact prescribed identities')
    complete=all(r.get('qualified')is True for r in rows);out=dict(planned_objects=8,planned_noise_replicates=4,planned_paired_trials=32,qualified_trials=sum(r.get('qualified')is True for r in rows),all32_retained=True,comparison_blocked=not complete,population_or_nominal_calibration_claim=False)
    if complete:
        values=[{name:float(np.mean([r['policies'][name]for r in rows if r['object_index']==i]))for name in('paired_adaptive_minus_fixed_MSE_m2','paired_adaptive_minus_random_MSE_m2')}for i in range(8)]
        out.update(equal_object_mean_adaptive_minus_fixed_MSE_m2=float(np.mean([r['paired_adaptive_minus_fixed_MSE_m2']for r in values])),equal_object_mean_adaptive_minus_random_MSE_m2=float(np.mean([r['paired_adaptive_minus_random_MSE_m2']for r in values])),object_noise_averages=values,interpretation='All8equallyweighted fixedobjects; four withinobjectnoise repeats. Descriptive gridconditional/noise stress only, no population CI or adaptive-advantage generalization.')
    return out


def actual_truth(p,index,models):
    points=run.design.physics.measurement_points();specs=run.design.physics.measurement_specs();labels=[v['id']for v in run.design.physics.load_specs()];queries={};linear={};captures=[]
    for label in labels:
        rec=run.capture(p,index,label);captures.append(rec)
        with np.load(bound(rec),allow_pickle=False)as z:
            queries[label]=z['query_displacement_m'].copy()
            need(queries[label].shape==(16,3)and np.isfinite(queries[label]).all(),'All finite physical query16 snapshots')
            if label in labels[9:]:linear[label]=run.physics_stage.geometry.field_queries(12,z['linear_displacement_m'],points)
    values=[];warm=[]
    for spec in specs:
        k=next(k for k,x in enumerate(points)if np.array_equal(x,spec['point_m']));values.append(float(queries[spec['load']][k]@np.asarray(spec['projection'])))
        if spec['role']=='target':warm.append(float(linear[spec['load']][k]@np.asarray(spec['projection'])))
    parameters=np.array([p['objects'][index]['E_Pa'],p['objects'][index]['K_N_m']]);unit=(np.log(parameters)-np.log(run.design.BOUNDS[:,0]))/run.design.LOG_WIDTH;layout=next(k for k,v in enumerate(run.design.physics.LAYOUTS)if v['id']==p['objects'][index]['layout']);oracle=models[layout].select([v['id']for v in specs[12:]])(unit[None])[0,:,0]
    return dict(values=np.array(values),warm=np.array(warm),oracle=oracle,parameters=parameters,captures=captures)


def audit():
    start=time.monotonic();guard(start);target=run.LOCAL/'independent-terminal-review.json';need(not target.exists(),'No independent audit overwrite')
    p,digest,phases,seals,inputs,entries=metadata_barrier();run.continuation_gate(digest)
    # Actual arrays and target-derived scores first become accessible HERE,
    # after every phase closed and every32 choice seal verified.
    with np.load(bound(inputs['poke_noise']),allow_pickle=False)as z:noise=z['poke_noise'].copy()
    need(noise.shape==(4,8,8)and np.isfinite(noise).all(),'Exact finite frozen poke-noise archive')
    models=run.base.load_models(dict(models=p['origin']['models']));truth_cache={};rows=[];reference_count=bank_count=joint_count=prefix_count=utility_count=0;bindings=[]
    names=[s['id']for s in run.design.physics.measurement_specs()];target_names=names[12:]
    for entry in entries:
        guard(start);index=entry['index'];trial=p['trials'][index];i,r=trial['object_index'],trial['replicate'];seal=run.read(bound(seals[index]['seal']));prefix=run.read(bound(seal['prefix_result']));input_record=inputs['prefixes'][index]
        if i not in truth_cache:truth_cache[i]=actual_truth(p,i,models)
        actual=truth_cache[i];truth=actual['values'][12:];parameters=actual['parameters']
        with np.load(bound(input_record['record']),allow_pickle=False)as z:
            need(set(z.files)=={'observed_prefix','prefix_noise'},'Exact acquired prefix schema');observed=z['observed_prefix'].copy();prefix_noise=z['prefix_noise'].copy()
        need(observed.shape==prefix_noise.shape==(4,)and np.isfinite(observed).all()and np.isfinite(prefix_noise).all()and np.array_equal(observed,actual['values'][:4]+prefix_noise)and run.design.fingerprint(observed)==input_record['observation_fingerprint'],'Acquired noisy prefix/physical source/fingerprint')
        observations=list(zip(names[:4],observed[:,None]));need(prefix['observations_fingerprint']==run.observations_fingerprint(observations),'Actual prefix context fingerprint')
        check=check_posterior(prefix,run.LOCAL/f'trial-{index:02d}'/'prefix',observations,p,prefix=True);reference_count+=check['references'];bank_count+=check['banks']
        ranking=prefix.get('ranking');rank_pass=False
        if ranking is not None:
            raw=dict(ranking);raw['qualified']=all(v['qualified']for v in raw['actions']);raw['unique_best_action']=raw['numerical_ties'][0]if raw['qualified']and len(raw['numerical_ties'])==1 else None
            utility_count+=len(prefix['utility_jobs']);c=actions_audit.check_rows(prefix['utility_jobs'],dict(outcome_relative=1e-4,latent_relative=1e-3),raw);rank_pass=c['qualified']
        qualified=bool(check['density_qualified']and rank_pass);need(prefix['qualified']is qualified and seal['qualified']is qualified,'Independent prefix density/ranking qualification');prefix_count+=qualified
        before=metrics(snapshot(bound(prefix['snapshot'])),truth,parameters)if qualified else None
        value=run.read(bound(entry['record']));need(value['index']==index and value['object_index']==i and value['replicate']==r and value['protocol_SHA256']==digest and value['choice_seal']==seals[index]['seal']and value['prefix_result']==seal['prefix_result']and value['source_captures']==actual['captures'],'Exact trial/input/seal/physical source lineage')
        with np.load(bound(value['truth']),allow_pickle=False)as z:
            need(set(z.files)=={'NH_target_m','n12_linear_target_m','n6_true_parameter_oracle_m','clean_candidate_m'},'Exact evaluator truth/oracle snapshot schema')
            for key,w in(('NH_target_m',truth),('n12_linear_target_m',actual['warm']),('n6_true_parameter_oracle_m',actual['oracle']),('clean_candidate_m',actual['values'][4:12])):need(np.array_equal(z[key],w),'Independent physical truth/linear oracle reproduction')
        oracle=dict(n12_linear_vs_n12_NH_MSE_m2=float(np.mean((actual['warm']-truth)**2)),n6_modal_vs_n12_linear_MSE_m2=float(np.mean((actual['oracle']-actual['warm'])**2)),n6_modal_vs_n12_NH_MSE_m2=float(np.mean((actual['oracle']-truth)**2)));need(equal(value['oracle_metrics'],oracle),'Independent model/discretization oracle errors')
        posts=value.get('posteriors',[]);need(len(posts)==8 and[v['action']for v in posts]==list(range(8)),'All8 one-poke counterfactuals retained')
        after=[]
        for action,post in enumerate(posts):
            directory=run.LOCAL/f'trial-{index:02d}'/'evaluation'/f'action-{action}';need(post['record']==run.record(directory/'result-LOCAL.json'),'Canonical one-poke posterior result');v=run.read(bound(post['record']));obs=observations+[(names[4+action],np.array([actual['values'][4+action]+noise[r,i,action]]))]
            need(v['index']==index and v['action']==action and v['protocol_SHA256']==digest and v['choice_seal']==seals[index]['seal']and v.get('prefix_plus_ONE_reset_poke')is True and v['observations_fingerprint']==run.observations_fingerprint(obs),'Actual single reset poke context/fingerprint')
            need(v['observations']==run.record(directory/'observations-LOCAL.npz'),'Canonical acquired observation path')
            with np.load(bound(v['observations']),allow_pickle=False)as z:need(set(z.files)=={'labels','values'}and tuple(str(x)for x in z['labels'])==tuple(x[0]for x in obs)and np.array_equal(z['values'],np.array([x[1]for x in obs])),'Exact prefix plus ONE reset observation, no accumulated pokes')
            c=check_posterior(v,directory,obs,p);reference_count+=c['references'];bank_count+=c['banks'];joint_count+=c['density_qualified'];need(post['qualified']is c['density_qualified'],'Joint record qualification journal')
            m=metrics(snapshot(bound(v['snapshot'])),truth,parameters)if c['density_qualified']else None
            if m is not None:need(equal(post['metrics'],m),'Independently reconstructed clean predictive/material metrics')
            after.append(m);guard(start)
        all_ok=bool(before is not None and qualified and all(m is not None for m in after));need(value['qualified']is all_ok and entry['qualified']is all_ok,'Whole fixed trial admission, no exclusions')
        row=dict(index=index,object_index=i,replicate=r,qualified=all_ok)
        if all_ok:
            need(equal(value['before'],before),'Independent before-poke metrics');a,f=int(seal['adaptive_action']),int(seal['fixed_action']);policy=policies(before['field'],[m['field']for m in after],a,f);material=material_policies(before['material'],[m['material']for m in after],a,f)
            need(equal(value['policies'],policy)and equal(value['material_policies'],material),'Independent paired policies/random mean-of-loss/material metrics');row.update(policies=policy,material_policies=material,before=before,adaptive_action=a,fixed_action=f)
        rows.append(row);bindings.append(entry['record'])
    computed=aggregate(rows);summary=run.read(run.LOCAL/'summary-LOCAL.json');need(equal(summary['summary'],computed)and sorted(summary['all32_trial_records'],key=lambda v:v['index'])==entries and summary['protocol_SHA256']==digest and summary['phases']==[run.record(run.LOCAL/f'execution-{name}.json')for name in('choices','checkpoint','remaining')],'Independent equal-object/four-noise aggregate')
    diagnostics={}
    if not computed['comparison_blocked']:
        for name in('adaptive','fixed','uniform_random'):
            diagnostics[name]=dict(mean_MSE_m2=float(np.mean([r['policies'][name]['MSE_m2']for r in rows])),mean_posterior_variance_m2=float(np.mean([r['policies'][name]['posterior_variance_m2']for r in rows])),mean_momentband_width_m=float(np.mean([r['policies'][name]['moment_band_2SD_mean_width_m']for r in rows])),mean_momentband_inclusion_fraction=float(np.mean([r['policies'][name]['moment_band_2SD_inclusion_fraction']for r in rows])),variance_down_error_up_trials=sum(r['policies'][name]['variance_down_error_up']for r in rows),material_central90_inclusion=np.mean([r['material_policies'][name]['central90_inclusion']for r in rows],axis=0).tolist(),material_central90_width=np.mean([r['material_policies'][name]['central90_width']for r in rows],axis=0).tolist(),descriptive_not_nominal_calibration=True)
    result=dict(audit_passed=True,planned_objects=8,planned_trials=32,planned_mixture_posteriors=288,planned_category_references=1152,completed_category_references=reference_count,planned_decision_banks=1440,completed_decision_banks=bank_count,planned_utility_jobs=1536,completed_utility_jobs=utility_count,qualified_prefixes=prefix_count,qualified_joint_posteriors=joint_count,all32_choice_seals_verified_before_any_actual_targets=True,summary=computed,descriptive_diagnostics=diagnostics,posterior_integrals_and_category_moments_are_bound_producer_witnesses=True,independent_oracle_refinement_scalar_flags_recomputed=True,source_SHA256={SOURCE:run.sha(run.ROOT/SOURCE),TEST:run.sha(run.ROOT/TEST)},source_physical_and_inference_SHA256=p['source_SHA256'],protocol=run.record(run.LOCAL/'protocol.json'),release=run.record(run.LOCAL/'release.json'),source_summary=run.record(run.LOCAL/'summary-LOCAL.json'),all32_trial_records=bindings,closed_phases={name:run.record(run.LOCAL/f'execution-{name}.json')for name in phases},audit_resources=LIMITS,audit_wall_s=time.monotonic()-start,audit_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope='Controlled8fixedobjects/fournoise repeats, matchingLINEAR model onNHtruth. No population CI/SBC/NHcalibration/generalBC NN or active advantage generalization. Mean±2SD are momentbands; variance-down/error-up alone is not miscalibration proof. Noise generator is source/seed/commitment-bound, not redrawn in audit. Raw arrays LOCAL.')
    guard(start);run.write(target,result);return result

if __name__=='__main__':print(json.dumps(audit()))
