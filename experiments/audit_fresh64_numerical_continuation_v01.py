"""Independent mixed-lineage terminal audit; new namespace, no inference replay.

Preparation/tests do not access real study evidence. All parent/new file hashes
and terminal closures precede any actual array access. Frozen old audit methods
verify unchanged physical evidence; accepted refined moments are checked under
explicit new schemas/plans. Original failures are never rewritten or dropped.
"""
import argparse,ast,hashlib,json,os,resource,signal,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_fresh64_numerical_continuation_v01 as run
from experiments import audit_fresh64_nonlinear_calibration_v01 as old
from experiments import campaign_fresh64_accepted_posterior_v01 as accepted
ROOT=run.ROOT;LOCAL=ROOT/'runs/operator-dev/fresh64-continuation-independent-audit-v01'
SOURCE='experiments/audit_fresh64_numerical_continuation_v01.py';TEST='tests/test_audit_fresh64_numerical_continuation_v01.py'
READER='experiments/campaign_fresh64_accepted_posterior_v01.py';READER_TEST='tests/test_fresh64_accepted_posterior_v01.py'
sha,record,write=run.sha,run.record,run.write
demand=old.demand


def sources():
    pending=[SOURCE,TEST,READER,READER_TEST];seen=set()
    while pending:
        f=pending.pop()
        if f in seen:continue
        seen.add(f)
        for node in ast.walk(ast.parse((ROOT/f).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or '']+[(node.module+'.'+v.name)for v in node.names if node.module]
            for name in modules:
                path=name.replace('.','/')+'.py'
                if(ROOT/path).is_file()and path not in seen:pending.append(path)
    return {f:sha(ROOT/f)for f in sorted(seen)}


def audit_predeclared_sources():return {name:sha(ROOT/name)for name in(SOURCE,TEST,READER,READER_TEST)}


def bound_archive(binding,manifest):
    path=Path(binding['path']);demand(not path.is_absolute()and'..'not in path.parts,'Canonical workspace relative evidence record')
    path=ROOT/path
    demand(str(path)in manifest and manifest[str(path)]==binding['sha256']and record(path)==binding,'Exact terminal-bound archive or metadata bytes')
    return path


def census_selection(entry,original):
    """Independently recover the numerical-only refinement choice, no labels."""
    if entry['decision']=='reuse_qualified':demand(original.get('qualified')is True,'Only qualified old contexts can be reused');return
    if entry['decision']=='blocked':return
    demand(entry['decision']=='refine'and original.get('qualified')is False,'Only failed original numerical contexts can refine')
    axes=[]
    for axis in('inner','outer'):
        rows=original.get('diagnostics',{}).get('successive',{}).get(axis,[])
        witness=original.get('strict_absolute_checks',{}).get('final_axis_witnesses',{}).get(axis,{})
        if not rows or rows[-1].get('passed')is not True or witness.get('passed')is not True:axes.append(axis)
    if not axes:axes=['inner','outer']
    plan={axis:[128,256,512]+([1024]if axis in axes else[])for axis in('inner','outer')};plan['moment_chunk']=1024
    demand(entry.get('failed_axes')==axes and entry.get('plan')==plan,'Refine every and only failed numerical axis; no outcome-based plan selection')


def summary_membership_checks(summary,execution,rows,entries,planned_workers):
    """Reject premature/tampered admission flags before any array is opened."""
    status=accepted.membership_status(rows,entries);counts={'original_five_levels':0,'refined_axis_extension':0}
    for row in rows:
        for kind in old.MODELS:
            for phase in old.KEYS:
                item=row['posterior_lineage'][kind][phase].get('accepted')
                if item is not None:
                    demand(item.get('source')in counts,'Unknown accepted-source lineage')
                    counts[item['source']]+=1
    demand(summary.get('qualified_posteriors')==status['accepted']and summary.get('unresolved')==status['unresolved']and summary.get('reused_qualified')==counts['original_five_levels']and summary.get('newly_qualified')==counts['refined_axis_extension'],'Exact continued acceptance/failure counters')
    demand(type(summary.get('comparison_blocked'))is bool and type(summary.get('completed'))is bool and type(execution.get('completed'))is bool,'Explicit completion/admission booleans')
    demand(not status['comparison_blocked']or summary['comparison_blocked'],'Unresolved896 membership cannot claim admission')
    if not summary['comparison_blocked']:
        demand(summary['completed']and execution['completed']and'failure'not in execution and len(execution['workers'])==planned_workers and all(w.get('reaped')is True and w.get('exit_code')==0 for w in execution['workers']),'Unblocked summary requires clean complete all-worker closure')
    return status


def terminal_metadata(digest,execution_sha,summary_sha):
    """All byte/closure barriers, no np.load; reusable by secondary reader.

    Returns p,census,execution,summary,merged_rows,original_rows,manifest,updates.
    Manifest keys are absolute paths, matching the frozen original audit ABI.
    """
    p,c=run.release_gate(digest)
    release=json.loads((run.LOCAL/'release.json').read_text())
    demand(release.get('audit_source_SHA256')==audit_predeclared_sources(),'Independent procedure prepinned before continuation execution')
    epath=run.LOCAL/'execution.json';spath=run.LOCAL/'summary-LOCAL.json'
    demand(sha(epath)==execution_sha and sha(spath)==summary_sha,'Exact continued terminal/summary seals')
    e=json.loads(epath.read_text());s=json.loads(spath.read_text())
    demand(e.get('terminal')is True and e.get('protocol_SHA256')==digest and e.get('release')==record(run.LOCAL/'release.json')and e.get('summary')==record(spath),'Continued terminal protocol/release/aggregate binding')
    demand(s.get('protocol_SHA256')==digest and s.get('objects')==64 and s.get('expected_posteriors')==896 and s.get('posterior_contexts_retained')==896 and s.get('all64_denominators_retained')is True,'All64/896 denominator/seal')
    demand(s.get('parent')==p['parent']and s.get('fine_control')==p['fine_control']and s.get('old_primary_evidence_unchanged')is True and s.get('no_new_draws_or_PDE')is True,'Original evidence/control lineage')
    workers=e.get('workers',[]);slots=[w.get('slot')for w in workers]
    demand(len(slots)<=p['resources']['workers']and len(set(slots))==len(slots)and all(type(v)is int and 0<=v<p['resources']['workers']for v in slots),'No replaced/extra worker slots')
    run.parent.full.prefix.absent(e['controller_pid'])
    pins=json.loads((ROOT/p['input_pins']['path']).read_text());run.verify_flat(pins)
    manifest={str(ROOT/path):h for path,h in pins.items()}
    def add(binding):
        path=Path(binding['path']);demand(not path.is_absolute()and'..'not in path.parts,'Relative in-workspace file record')
        path=ROOT/path;demand(record(path)==binding,'Changed continued file commitment');manifest[str(path)]=binding['sha256'];return path
    add(record(run.LOCAL/'protocol.json'));add(record(run.LOCAL/'release.json'));add(record(epath));add(record(spath))
    parent_protocol=json.loads(bound_archive(p['parent']['protocol'],manifest).read_text())
    parent_audit=json.loads(bound_archive(p['parent']['independent_audit'],manifest).read_text())
    demand(parent_audit.get('completed')is True and parent_audit.get('integrity_passed')is True and parent_audit.get('execution_clean')is True and parent_audit.get('comparison_blocked')is True,'Sealed original integrity PASS/numerical negative preserved')
    original=[]
    for i in range(64):
        path=run.parent.LOCAL/f'object-{i:02d}/summary-LOCAL.json'
        if path.exists():
            demand(str(path)in manifest and sha(path)==manifest[str(path)],'All original object manifests bound')
            row=json.loads(path.read_text());demand(row.get('index')==i and row.get('protocol_SHA256')==p['parent_protocol_SHA256'],'Original object context');original.append(row)
    original=old.planned_records(original)
    entries=c.get('entries',[]);expected={f'{i:02d}/{k}/{phase}'for i in range(64)for k in old.MODELS for phase in old.KEYS}
    demand(len(entries)==896 and {v.get('id')for v in entries}==expected,'Full original census exactly once')
    jobs=[v for v in entries if v['decision']=='refine'];byid={v['id']:v for v in entries}
    for entry in entries:
        r=original[entry['index']]['posteriors'][entry['kind']][entry['phase']];census_selection(entry,r)
        if entry.get('original'):
            demand(entry['original']['context']==r.get('context'),'Original numerical context after frozen census')
            path=bound_archive(entry['original']['json'],manifest)
            demand(path==run.parent.LOCAL/f"object-{entry['index']:02d}"/(entry['kind']+'-'+entry['phase']+'-posterior-LOCAL.json'),'Original posterior path membership')
            if entry['original'].get('raw'):bound_archive(entry['original']['raw'],manifest)
    worker_updates={}
    for w in workers:
        demand(w.get('reaped')is True and type(w.get('exit_code'))is int,'Every owned continued worker reaped, failures retained');run.parent.full.prefix.absent(w['pid'])
        if not w.get('summary'):demand(w['exit_code']!=0,'Successful worker cannot omit summary');continue
        path=add(w['summary']);demand(path==run.LOCAL/f"worker-{w['slot']}/summary-LOCAL.json",'Fixed worker-summary path')
        z=json.loads(path.read_text());demand(z.get('slot')==w['slot']and z.get('pid')==w['pid']and z.get('protocol_SHA256')==digest,'Worker identity')
        assigned={v['id']for v in jobs[w['slot']::p['resources']['workers']]};seen=set()
        for j in z.get('jobs',[]):
            identity=j['id'];demand(identity in assigned and identity not in seen and identity not in worker_updates,'No duplicated/reassigned/refiltered continued jobs');seen.add(identity)
            demand(j['plan']==byid[identity]['plan'],'Frozen numerical plan');entry=byid[identity]
            jpath=add(j['binding']['json']);expected_path=run.LOCAL/'jobs'/f"{entry['index']:02d}-{entry['kind']}-{entry['phase']}"/'posterior-LOCAL.json';demand(jpath==expected_path,'Fixed job output path')
            r=json.loads(jpath.read_text())
            if j['binding'].get('raw'):
                raw=add(j['binding']['raw']);demand(raw==jpath.with_suffix('.npz')and r.get('raw_SHA256')==j['binding']['raw']['sha256'],'Refined raw path/checksum')
            else:demand(not r.get('qualified')and not r.get('raw_SHA256'),'No missing claimed/qualified raw')
            worker_updates[identity]=dict(record=r,binding=j['binding'],plan=j['plan'])
        if z.get('completed'):demand(seen==assigned,'Completed worker cannot omit failed jobs')
    update_path=add(s['update_manifest']);u=json.loads(update_path.read_text())
    demand(set(u)==set(worker_updates),'Every committed worker result appears once in update manifest')
    for key,value in u.items():demand(value==dict(binding=worker_updates[key]['binding'],plan=worker_updates[key]['plan']),'Exact worker/update commitments')
    merged_path=add(s['merged_rows']);rows=json.loads(merged_path.read_text())
    demand(len(s.get('object_summaries',[]))==64,'All merged object summaries')
    object_rows=[]
    for i,binding in enumerate(s['object_summaries']):
        path=add(binding);demand(path==run.LOCAL/f'object-{i:02d}/summary-LOCAL.json','Fixed merged object order')
        row=json.loads(path.read_text());demand(row.get('index')==i and row.get('protocol_SHA256')==digest and row.get('parent_protocol_SHA256')==p['parent_protocol_SHA256'],'Merged object lineage')
        oldpath=run.parent.LOCAL/f'object-{i:02d}/summary-LOCAL.json'
        demand(row.get('parent_object_summary')==(record(oldpath)if oldpath.exists()else None),'Immutable original object hash')
        object_rows.append(row)
    demand(rows==object_rows,'Global merged rows equal all64 per-object commits');summary_membership_checks(s,e,rows,entries,p['resources']['workers'])
    add(s['raw_calibration'])
    # Parent protocol must remain an explicit source-bound root, not a self-anchor.
    demand(parent_protocol['cost_receipt']['memo']['model_fingerprint']==p['parent_configuration']['cost_receipt']['memo']['model_fingerprint'],'Same qualified original CPU modal identity')
    return p,c,e,s,rows,original,manifest,worker_updates


def parent_arrays(index,p,manifest):
    """Only after terminal_metadata. Return selected saved fields; no PDE."""
    directory=run.parent.LOCAL/f'object-{index:02d}';draws=directory/'draws-LOCAL.npz'
    demand(str(draws)in manifest and sha(draws)==manifest[str(draws)],'Bound original draw')
    with np.load(draws,allow_pickle=False)as z:unit=z['unit'].copy();prefix=z['prefix_noise'].copy();noise=z['probe_noise'].copy()
    responses=[]
    for label in old.design.TRUTH_LABELS:
        path=directory/(label+'-truth-LOCAL.npz');demand(str(path)in manifest and sha(path)==manifest[str(path)],'Bound original direct response')
        with np.load(path,allow_pickle=False)as z:responses.append(z['response_m'].copy())
    responses=np.asarray(responses);obs={}
    for key in old.KEYS:
        values=[(old.design.PREFIX,responses[0]+prefix)]
        if key!='prefix':
            family,reset=key.split('-r');f=('normal','axial').index(family);values.append((old.design.PROBES[f],responses[f]+noise[f,int(reset)]))
        obs[key]=run.parent.oldstudy.observation_fingerprint(values)
    return unit,responses[2:].ravel(),obs,sha(draws)


def audit(digest,execution_sha,summary_sha):
    demand(set(os.sched_getaffinity(0))=={18}and os.environ.get('CUDA_VISIBLE_DEVICES')==''and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'Independent CPU18/one-thread/noGPU audit')
    demand(not(LOCAL/'result.json').exists(),'No audit overwrite');start=time.monotonic();result=dict(completed=False,integrity_passed=False,comparison_blocked=True,source_SHA256=sources(),no_PDE_or_posterior_replay=True,no_parameter_or_noise_redraw=True)
    def stop(*_):raise TimeoutError('600s independent continued audit cap')
    previous=signal.signal(signal.SIGALRM,stop);signal.alarm(600)
    try:
        p,c,e,s,rows,original,manifest,updates=terminal_metadata(digest,execution_sha,summary_sha)
        parent_protocol=json.loads((ROOT/p['parent']['protocol']['path']).read_text());expected_modal=p['parent_configuration']['cost_receipt']['memo']['model_fingerprint'];entries={v['id']:v for v in c['entries']};audited=[];checks=[];counts=dict(original_five_levels=0,refined_axis_extension=0)
        for i,(row,orig)in enumerate(zip(rows,original)):
            base,physical_check=old.object_arrays(orig,parent_protocol,p['parent_protocol_SHA256'],manifest);answer=dict(row,posteriors={k:dict(v)for k,v in row['posteriors'].items()});slot_checks=[]
            if physical_check.get('missing')or physical_check.get('missing_draw')or physical_check.get('incomplete_truth'):
                demand(all(v.get('accepted')is None for group in row['posterior_lineage'].values()for v in group.values()),'No accepted forecast without complete original truth')
                audited.append(answer);checks.append(dict(index=i,physical_records=physical_check.get('truth_records',0),accepted=0,original_incomplete_preserved=True));continue
            unit,target,observations,draw_sha=parent_arrays(i,p,manifest)
            for kind in old.MODELS:
                for phase in old.KEYS:
                    r=row['posteriors'][kind][phase];lineage=row['posterior_lineage'][kind][phase];a=lineage.get('accepted');identity=f'{i:02d}/{kind}/{phase}';entry=entries[identity];o=orig['posteriors'][kind][phase]
                    if a is None:demand(r.get('qualified')is not True,'Unresolved slot admitted');continue
                    context=o['context'];demand(context['draws_SHA256']==draw_sha and context['observations_SHA256']==observations[phase]and context['true_unit_SHA256']==old.array_digest(unit),'Independent same-noise/parameter commitment')
                    check=accepted.verify_accepted_record(r,a,entry,o,continuation_sha=digest,expected_modal_fingerprint=expected_modal,targets=p['parent_configuration']['target_labels'],atlas_fingerprint=p['parent_configuration']['atlas_fingerprint']);slot_checks.append(check)
                    path=bound_archive(a['json'],manifest);saved=json.loads(path.read_text())
                    for key,value in saved.items():old.compare(value,r.get(key),'accepted JSON/'+identity+'/'+key)
                    raw=bound_archive(a['raw'],manifest)
                    if a['source']=='original_five_levels':answer['posteriors'][kind][phase]=base['posteriors'][kind][phase]
                    else:
                        demand(identity in updates and updates[identity]['binding']==dict(json=a['json'],raw=a['raw'])and updates[identity]['plan']==entry['plan'],'Accepted refinement belongs to exact committed update')
                        with np.load(raw,allow_pickle=False)as z:metrics=old.lean_metrics(z,unit,target)
                        for key,value in metrics.items():old.compare(value.tolist()if isinstance(value,np.ndarray)else value,r[key],identity+'/'+key)
                        answer['posteriors'][kind][phase]=dict(r,**metrics)
                    counts[a['source']]+=1
            old.compare(row['oracle_RMS_m'],base['oracle_RMS_m'],'Unchanged original oracle');answer['oracle_RMS_m']=base['oracle_RMS_m'];audited.append(answer);checks.append(dict(index=i,physical_records=physical_check.get('truth_records',0),accepted=len(slot_checks),lineage_checks=slot_checks))
            demand(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<4*1024**3,'Own4GiB RSS')
        cal,prediction,primary=old.aggregate(audited);old.compare(cal,s['primary_calibration'],'Continued primary statistics');old.compare(prediction,s['paired_metrics'],'Continued shared-object bootstrap')
        with np.load(ROOT/s['raw_calibration']['path'],allow_pickle=False)as z:
            demand(set(z.files)==set(primary),'Raw calibration memberships')
            for key,value in primary.items():demand(np.array_equal(z[key],value,equal_nan=True),'Same continued raw coverage/PIT')
        membership=accepted.membership_status(rows,c['entries']);blocked=membership['comparison_blocked']or cal['comparison_blocked']or prediction['comparison_blocked'];clean=bool(e.get('completed')is True and'failure'not in e and len(e['workers'])==p['resources']['workers']and all(w['exit_code']==0 for w in e['workers']))
        demand(s['qualified_posteriors']==membership['accepted']and s['reused_qualified']==counts['original_five_levels']and s['newly_qualified']==counts['refined_axis_extension']and s['unresolved']==membership['unresolved']and s['comparison_blocked']==blocked,'All admission/failure counts and denominator flags')
        result.update(completed=True,integrity_passed=True,comparison_blocked=bool(blocked or not clean),execution_clean=clean,objects=64,expected_posteriors=896,qualified_posteriors=membership['accepted'],accepted_original=counts['original_five_levels'],accepted_refined=counts['refined_axis_extension'],unresolved=membership['unresolved'],protocol=record(run.LOCAL/'protocol.json'),release=record(run.LOCAL/'release.json'),execution=record(run.LOCAL/'execution.json'),summary=record(run.LOCAL/'summary-LOCAL.json'),canonical_modal_fingerprint=expected_modal,aggregate_statistics_reproduced=True,closures_verified=True,source_bindings_verified=True,object_checks=checks,scope='Numerical/context/integrity audit; not a population coverage criterion, model-fidelity test or CDF/PDE replay.')
    except BaseException as error:result['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous);result.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol-sha',required=True);parser.add_argument('--execution-sha',required=True);parser.add_argument('--summary-sha',required=True);a=parser.parse_args();r=audit(a.protocol_sha,a.execution_sha,a.summary_sha);print(json.dumps(dict(completed=r['completed'],integrity_passed=r['integrity_passed'],comparison_blocked=r['comparison_blocked'])))
