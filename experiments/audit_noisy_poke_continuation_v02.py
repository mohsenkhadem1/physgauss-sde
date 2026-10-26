"""Independent mixed-lineage audit of the resource-only noisy continuation.

Original trial0, all32 choices and noise commitments are reused byte-for-byte.
New trials1..31 attest new execution sources/resources, not old execution.
All phase/source/context/hash barriers precede NPZ; no posterior/PDE/RNG replay.
"""
import argparse,json,os,resource,time
import numpy as np
from experiments import run_noisy_poke_continuation_v02 as run
from experiments import run_noisy_poke_benchmark_v01 as parent
from experiments import audit_noisy_poke_benchmark_v01 as legacy
ROOT=run.ROOT
SOURCE='experiments/audit_noisy_poke_continuation_v02.py'
TEST='tests/test_audit_noisy_poke_continuation_v02.py'
LIMITS=dict(CPU=16,threads=1,GPU=False,wall_s=3600,RSS_bytes=4*1024**3,external_owned_watchdog_required=True)
need=legacy.need
check_posterior,snapshot,metrics=legacy.check_posterior,legacy.snapshot,legacy.metrics
policies,material_policies,aggregate=legacy.policies,legacy.material_policies,legacy.aggregate
equal,actual_truth,actions_audit=legacy.equal,legacy.actual_truth,legacy.actions_audit

def sources():return {name:run.sha(ROOT/name)for name in(SOURCE,TEST)}
def guard(start):
    need(set(os.sched_getaffinity(0))=={16}and os.environ.get('CUDA_VISIBLE_DEVICES')==''and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'Audit CPU16/noGPU/one-thread')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=LIMITS['RSS_bytes']:raise MemoryError('Audit4GiB')
    if time.monotonic()-start>=LIMITS['wall_s']:raise TimeoutError('Audit3600s')
def bound(rec):
    path=ROOT/rec['path'];need(run.record(path)==rec,'Exact mixed artifact byte commitment');return path

def phase_records():
    return [run.record((parent.LOCAL if name!='remaining'else run.LOCAL)/f'execution-{name}.json')for name in('choices','checkpoint','remaining')]

def check_execution_context(value,index):
    context=run.execution_context()
    if index==0:
        need(all(k not in value for k in context),'Original trial0 bytes must not gain new execution attestations')
    else:
        need(all(value.get(k)==v for k,v in context.items()),'Every new trial/post attests NEW execution protocol/release/source')

def metadata_barrier(parent_sha):
    # run's barrier verifies BOTH original closed phases + NEW remaining and
    # exact1..31 schedules; it performs no model/array access.
    p,phases,seals,inputs,entries=run.metadata_barrier(parent_sha)
    new_protocol=run.read(run.LOCAL/'protocol.json')
    need(new_protocol.get('source_SHA256')==run.sources()and all(new_protocol['source_SHA256'].get(k)==v for k,v in sources().items()),'New independent auditor/test prospectively pinned')
    need(p['source_SHA256']==parent.sources(),'Original scientific and audit source attestations unchanged')
    need(set(phases)=={'choices','checkpoint','remaining'}and len(seals)==32 and[v['index']for v in seals]==list(range(32)),'Both old phases/newremaining + all32oldseals')
    need(sum(v['wall_s']for v in phases.values())<=run.RESOURCES['global_wall_s']and phases['remaining']['wall_s']<=run.RESOURCES['evaluation_wall_s'],'Documented revised wall budgets')
    for e in phases.values():
        need(e.get('peak_aggregate_RSS_bytes',0)<=run.RESOURCES['aggregate_RSS_bytes'],'Unchanged aggregate RSS')
        for w in e['workers']:need(w.get('peak_RSS_bytes',0)<=run.RESOURCES['worker_RSS_bytes']and w.get('peak_owned_VRAM_bytes',0)<=run.RESOURCES['owned_VRAM_bytes'],'Unchanged worker host/GPU caps')
    need([v['index']for v in entries]==list(range(32)),'All32 mixed old0/new1..31; no replacements')
    for entry in entries:
        index=entry['index'];directory=(parent.LOCAL if index==0 else run.LOCAL)/f'trial-{index:02d}'/'evaluation'
        need(run.evaluation_directory(index)==directory and entry['record']==run.record(directory/'result-LOCAL.json'),'Exact old0/new31 canonical path')
    # Hash every prefix and trial's transitive raw commitments before ANY NPZ.
    # Numerical payloads are not scored in this metadata-only pass.
    parent.verify(inputs)
    parent.verify(p['origin'])
    for entry,seal in zip(entries,seals):
        choice=run.read(bound(seal['seal']));prefix=run.read(bound(choice['prefix_result']));parent.verify(prefix)
        value=run.read(bound(entry['record']));index=entry['index'];check_execution_context(value,index);parent.verify(value)
        need((value.get('index'),value.get('object_index'),value.get('replicate'))==(index,index%8,index//8),'Every mixed trial retains original identity')
        posts=value.get('posteriors',[]);need(len(posts)==8 and[v['action']for v in posts]==list(range(8)),'All256 resetpost memberships before NPZ')
        for action,post in enumerate(posts):
            need(post['record']==run.record(run.evaluation_directory(index)/f'action-{action}'/'result-LOCAL.json'),'Canonical everymixed post path before NPZ')
            v=run.read(bound(post['record']));check_execution_context(v,index);parent.verify(v)
            need(v.get('index')==index and v.get('action')==action and v.get('protocol_SHA256')==parent_sha and v.get('prefix_plus_ONE_reset_poke')is True,'Every mixedpost original design/reset context')
    return p,phases,seals,inputs,entries

def audit(parent_sha):
    start=time.monotonic();guard(start);target=run.LOCAL/'independent-terminal-review.json';need(not target.exists(),'No independent audit overwrite')
    p,phases,seals,inputs,entries=metadata_barrier(parent_sha);digest=parent_sha
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
        check=check_posterior(prefix,parent.LOCAL/f'trial-{index:02d}'/'prefix',observations,p,prefix=True);reference_count+=check['references'];bank_count+=check['banks']
        ranking=prefix.get('ranking');rank_pass=False
        if ranking is not None:
            raw=dict(ranking);raw['qualified']=all(v['qualified']for v in raw['actions']);raw['unique_best_action']=raw['numerical_ties'][0]if raw['qualified']and len(raw['numerical_ties'])==1 else None
            utility_count+=len(prefix['utility_jobs']);c=actions_audit.check_rows(prefix['utility_jobs'],dict(outcome_relative=1e-4,latent_relative=1e-3),raw);rank_pass=c['qualified']
        qualified=bool(check['density_qualified']and rank_pass);need(prefix['qualified']is qualified and seal['qualified']is qualified,'Independent prefix density/ranking qualification');prefix_count+=qualified
        before=metrics(snapshot(bound(prefix['snapshot'])),truth,parameters)if qualified else None
        value=run.read(bound(entry['record']));check_execution_context(value,index);need(value['index']==index and value['object_index']==i and value['replicate']==r and value['protocol_SHA256']==digest and value['choice_seal']==seals[index]['seal']and value['prefix_result']==seal['prefix_result']and value['source_captures']==actual['captures'],'Exact trial/input/seal/physical source lineage')
        with np.load(bound(value['truth']),allow_pickle=False)as z:
            need(set(z.files)=={'NH_target_m','n12_linear_target_m','n6_true_parameter_oracle_m','clean_candidate_m'},'Exact evaluator truth/oracle snapshot schema')
            for key,w in(('NH_target_m',truth),('n12_linear_target_m',actual['warm']),('n6_true_parameter_oracle_m',actual['oracle']),('clean_candidate_m',actual['values'][4:12])):need(np.array_equal(z[key],w),'Independent physical truth/linear oracle reproduction')
        oracle=dict(n12_linear_vs_n12_NH_MSE_m2=float(np.mean((actual['warm']-truth)**2)),n6_modal_vs_n12_linear_MSE_m2=float(np.mean((actual['oracle']-actual['warm'])**2)),n6_modal_vs_n12_NH_MSE_m2=float(np.mean((actual['oracle']-truth)**2)));need(equal(value['oracle_metrics'],oracle),'Independent model/discretization oracle errors')
        posts=value.get('posteriors',[]);need(len(posts)==8 and[v['action']for v in posts]==list(range(8)),'All8 one-poke counterfactuals retained')
        after=[]
        for action,post in enumerate(posts):
            directory=run.evaluation_directory(index)/f'action-{action}';need(post['record']==run.record(directory/'result-LOCAL.json'),'Canonical one-poke posterior result');v=run.read(bound(post['record']));check_execution_context(v,index);obs=observations+[(names[4+action],np.array([actual['values'][4+action]+noise[r,i,action]]))]
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
    computed=aggregate(rows);summary=run.read(run.LOCAL/'summary-LOCAL.json');need(equal(summary['summary'],computed)and sorted(summary['all32_trial_records'],key=lambda v:v['index'])==entries and summary['protocol_SHA256']==digest and summary['phases']==phase_records()and summary.get('parent_protocol')==run.record(parent.LOCAL/'protocol.json')and all(summary.get(k)==v for k,v in run.execution_context().items()),'Independent equal-object/four-noise aggregate')
    diagnostics={}
    if not computed['comparison_blocked']:
        for name in('adaptive','fixed','uniform_random'):
            diagnostics[name]=dict(mean_MSE_m2=float(np.mean([r['policies'][name]['MSE_m2']for r in rows])),mean_posterior_variance_m2=float(np.mean([r['policies'][name]['posterior_variance_m2']for r in rows])),mean_momentband_width_m=float(np.mean([r['policies'][name]['moment_band_2SD_mean_width_m']for r in rows])),mean_momentband_inclusion_fraction=float(np.mean([r['policies'][name]['moment_band_2SD_inclusion_fraction']for r in rows])),variance_down_error_up_trials=sum(r['policies'][name]['variance_down_error_up']for r in rows),material_central90_inclusion=np.mean([r['material_policies'][name]['central90_inclusion']for r in rows],axis=0).tolist(),material_central90_width=np.mean([r['material_policies'][name]['central90_width']for r in rows],axis=0).tolist(),descriptive_not_nominal_calibration=True)
    result=dict(audit_passed=True,planned_objects=8,planned_trials=32,planned_mixture_posteriors=288,planned_category_references=1152,completed_category_references=reference_count,planned_decision_banks=1440,completed_decision_banks=bank_count,planned_utility_jobs=1536,completed_utility_jobs=utility_count,qualified_prefixes=prefix_count,qualified_joint_posteriors=joint_count,all32_choice_seals_verified_before_any_actual_targets=True,summary=computed,descriptive_diagnostics=diagnostics,posterior_integrals_and_category_moments_are_bound_producer_witnesses=True,independent_oracle_refinement_scalar_flags_recomputed=True,source_SHA256={SOURCE:run.sha(run.ROOT/SOURCE),TEST:run.sha(run.ROOT/TEST)},source_physical_and_inference_SHA256=p['source_SHA256'],parent_source_SHA256=p['source_SHA256'],execution_source_SHA256=run.sources(),parent_protocol=run.record(parent.LOCAL/'protocol.json'),parent_protocol_SHA256=parent_sha,reused_trial_indices=[0],new_trial_indices=list(range(1,32)),protocol=run.record(run.LOCAL/'protocol.json'),release=run.record(run.LOCAL/'release.json'),source_summary=run.record(run.LOCAL/'summary-LOCAL.json'),all32_trial_records=bindings,closed_phases=dict(zip(('choices','checkpoint','remaining'),phase_records())),audit_resources=LIMITS,audit_wall_s=time.monotonic()-start,audit_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope='Versioned resource-only continuation: original trial0 and new untouched1..31, same original choices/noise/inference. Controlled8fixedobjects/fournoise repeats, matchingLINEAR model onNHtruth. No population CI/SBC/NHcalibration/generalBC NN or active advantage generalization. Mean±2SD are momentbands; variance-down/error-up alone is not miscalibration proof. Noise generator is source/seed/commitment-bound, not redrawn in audit. Raw arrays LOCAL.')
    guard(start);run.write(target,result);return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sha',required=True);args=ap.parse_args();print(json.dumps(audit(args.sha)))
