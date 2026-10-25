"""Independent all264 saved-field audit; no PDE, factor, RNG or inference.

Shared frozen geometry/load/deformation validators reproduce arrays and
physical metrics. All phase closures and264 byte/context commitments precede
first NPZ. Sparse residual and tangent operators remain producer witnesses;
reference validity never filters low excitation or grants legacy eligibility.
"""
import json,os,resource,time
import numpy as np
from experiments import run_fixture_balanced_nh_catalog_v01 as stage
from experiments import audit_fixture_balanced_held_nh_refinement_v01 as field_audit
SOURCE='experiments/audit_fixture_balanced_nh_catalog_v01.py'
TEST='tests/test_audit_fixture_balanced_nh_catalog_v01.py'
need=field_audit.need
AUDIT_LIMITS=dict(CPU=16,GPU=False,threads=1,wall_s=3600,RSS_bytes=4*1024**3,external_owned_watchdog_required=True)

def guard(start):
    need(set(os.sched_getaffinity(0))=={AUDIT_LIMITS['CPU']}and os.environ.get('CUDA_VISIBLE_DEVICES')==''and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'Exact CPU16/noGPU/one-thread audit guard')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=AUDIT_LIMITS['RSS_bytes']:raise MemoryError('Audit4GiB RSS cap')
    if time.monotonic()-start>=AUDIT_LIMITS['wall_s']:raise TimeoutError('Audit3600s cap')


def read(path):return json.loads(path.read_text())


def phase_metadata(phase,p,digest):
    base=stage.LOCAL;path=base/f'execution-{phase}.json';e=read(path);closure=base/f'{phase}-root-closure.json';c=read(closure);release=stage.record(base/'release.json')
    need(e.get('terminal')is True and e.get('completed')is True and 'failure'not in e and e['phase']==phase and e['protocol_SHA256']==digest and e['release']==release,'Clean exact root-reaped phase')
    assigned=stage.assignments(phase);need(len(e['workers'])==len(assigned),'Exact phase worker count')
    ids=[e['controller_pid']]+[v['pid']for v in e['workers']]
    need(c.get('root_exit_code')==0 and c.get('root_reaped')is True and c.get('all_owned_PID_and_groups_absent')is True and len(c.get('owned_pids',[]))==len(ids)and set(c['owned_pids'])==set(ids),'External exact phase root closure')
    need(c.get('protocol_SHA256')==digest and c.get('source_SHA256')==p['source_SHA256']and c.get('execution_SHA256')==stage.sha(path),'Root closure source/protocol/execution')
    for pid in ids:stage.controls.prior.shared.old.absent(pid)
    need(field_audit.le(e['peak_aggregate_RSS_bytes'],stage.LIMITS['aggregate_RSS_bytes']),'Phase aggregate RSS')
    groups=[]
    for slot,w in enumerate(e['workers']):
        need(w['slot']==slot and w['groups']==assigned[slot]and w.get('exit_code')==0 and w.get('cleanup_exit_code')==0 and w.get('reaped')is True,'Exact worker assignments and reap')
        wp=base/f'worker-{phase}-{slot}.json';need(w['result']==stage.record(wp),'Worker byte commitment');v=read(wp)
        need(v.get('completed')is True and 'failure'not in v and v['pid']==w['pid']and v['slot']==slot and v['phase']==phase and v['protocol_SHA256']==digest and v['release']==release and[x['index']for x in v['groups']]==assigned[slot],'Exact complete worker lineage')
        need(field_audit.le(v['peak_RSS_bytes'],stage.LIMITS['worker_RSS_bytes']),'Worker RSS bound')
        for item in v['groups']:
            index=item['index'];gp=base/'groups'/f'group-{index:02d}'/'result-LOCAL.json';need(item['result']==stage.record(gp),'Group byte commitment');cache=stage.ROOT/item['external_cache']['path'];need(item['external_cache']==stage.record(cache)and cache==stage.ROOT/'runs/operator-dev/completed-metadata-cache-v01'/(item['result']['sha256']+'.json')and cache.read_bytes()==gp.read_bytes(),'Exact immutable completed group cache');g=read(gp)
            need(g.get('completed')is True and 'failure'not in g and g['group']==p['plan']['groups'][index]and g['protocol_SHA256']==digest and g['release']==release and g['worker_pid']==w['pid']and g['new_fields']==33 and g['reused_fields']==0,'Exact new group lineage')
            need(field_audit.le(g['wall_s'],stage.LIMITS['group_wall_s'])and field_audit.le(g['peak_RSS_bytes'],stage.LIMITS['worker_RSS_bytes']),'Group wall/RSS caps')
            need(len(g['rows'])==33 and{(r['n'],r['label'])for r in g['rows']}=={(n,l)for n in stage.LEVELS for l in stage.LOADS},'All33 unique fields per group')
            for row in g['rows']:
                need(row.get('completed')is True and row['kind']=='NEW'and row['group_index']==index and row['layout']==g['group']['layout']and(row['E_Pa'],row['K_N_m'])==(g['group']['E_Pa'],g['group']['K_N_m']),'Exact264 material/load contexts')
                cap=gp.parent/'raw-LOCAL'/f"n{row['n']}-{row['label']}.npz";need(row['capture']==stage.record(cap),'All264 capture bytes before NPZ')
            groups.append(dict(index=index,result=item['result'],value=g))
    return dict(execution=stage.record(path),root_closure=stage.record(closure),wall_s=e['wall_s']),groups


def metadata_barrier():
    base=stage.LOCAL;digest=stage.sha(base/'protocol.json');p=stage.validate(digest)
    need({SOURCE,TEST}<=set(p['source_SHA256']),'Prospectively frozen independent audit/test')
    a,left=phase_metadata('checkpoint',p,digest);b,right=phase_metadata('remaining',p,digest)
    need(field_audit.le(a['wall_s']+b['wall_s'],stage.LIMITS['global_wall_s']),'Combined global wall cap')
    stage.continuation_gate(p,digest)
    groups=sorted(left+right,key=lambda x:x['index']);need([x['index']for x in groups]==list(range(8)),'All8 fixed objects no replacement')
    s=read(base/'summary-LOCAL.json');phases=dict(checkpoint=a,remaining=b)
    need(s.get('completed')is True and s.get('all264_retained')is True and s.get('new_fields')==264 and s.get('reused_fields')==0 and s.get('group_load_references')==88 and s['protocol']==stage.record(base/'protocol.json')and s['release']==stage.record(base/'release.json')and s['source_SHA256']==p['source_SHA256']and s['phases']==phases,'Exact complete summary context/counts')
    expected=[dict(index=x['index'],group=x['value']['group'],result=x['result'],completed=True,reference_valid=x['value']['all11_reference_valid'],references=x['value']['references'],captures=[dict(n=r['n'],label=r['label'],capture=r['capture'])for r in x['value']['rows']])for x in groups]
    need(s['groups']==expected,'Exact8 summary/group/capture commitments')
    return p,s,phases,groups


def field_status(row,d):
    physical=bool(row['physical_passed']);excited=bool(d['nonrigid_RMS_m']>=100e-6 and d['nonrigid_fraction']>=.3)
    temporary=dict(row,physical_passed=physical,excitation_goal_met=excited,weak_prefix_control=False,load_eligible=bool(physical and excited),bank_admitted=False)
    checked=field_audit.scalar_status(temporary,d) # independent frozen residual/reaction/Ritz scalar predicates
    weak=row['label']=='prefix-center-down';legacy=bool(checked['physical_passed']and d['nonrigid_fraction']>=.3 and(weak or excited))
    result=dict(physical_passed=checked['physical_passed'],excitation_goal_met=excited,legacy_load_eligible=legacy,legacy_weak_prefix_control=weak,nonrigid_RMS_goal_m=100e-6,nonrigid_fraction_goal=.3)
    need(all(row[k]==v for k,v in result.items()),'Unchanged legacy eligibility versus new reference status')
    return result


def spatial_results(fields):
    out={};g=stage.controls.metrics.GATES
    for label in stage.LOADS:
        values=[(n,fields[n,label])for n in stage.LEVELS];dist=stage.geometry.common_rms(values,finest=12);a,b=float(dist[1,0]),float(dist[2,1]);need(field_audit.le(a,np.inf)and field_audit.le(b,np.inf),'Finite common-volume distances')
        order=float(np.log2(a/b))if a>0 and b>0 else None;fine=b/(2**order-1)if order is not None and order>=g['spatial_order_floor']else None;medium=2*b
        spatial=bool(fine is not None and fine<=g['maximum_conditional_fine_indicator_m']);adequate=bool(order is not None and order>=g['spatial_order_floor']and medium<=g['maximum_medium_indicator_m'])
        query=[stage.geometry.field_queries(n,u,stage.bank.measurement_points())for n,u in values];coarse=float(np.max(abs(query[1]-query[0])));last=float(np.max(abs(query[2]-query[1])))
        out[label]=dict(coarse_medium_RMS_m=a,medium_fine_RMS_m=b,observed_order=order,conditional_fine_indicator_m=fine,medium_indicator_m=medium,spatial_passed=spatial,medium_adequate=adequate,conditional_only=True,public_point_n3_n6_max_coordinate_m=coarse,public_point_n6_n12_max_coordinate_m=last,public_point20um_passed=last<=20e-6,spatial_development_passed=bool(spatial and adequate and last<=20e-6))
    return out


def surfaces_pass(surfaces):
    need(set(surfaces)=={str(n)for n in stage.LEVELS},'All3 surface quadrature witnesses')
    for s in surfaces.values():
        need(s.get('positive_quadrature_Gram_PSD')is True,'Positive quadrature Gram scope')
        ok=field_audit.le(s['support']['normalization_error'],1e-10)and field_audit.le(s['support']['first_moment_error_m'],1e-10)and field_audit.le(s['relative_support16_vs20'],1e-10)and field_audit.le(s['maximum_load16_vs20_N'],1e-12)
        need(set(s['refined_known_load_checks'])==set(stage.LOADS),'All11 known-load surface witnesses')
        for v in s['refined_known_load_checks'].values():ok=ok and all(field_audit.le(v[key],limit)for key,limit in(('net_force_error_N',1e-12),('moment_error_Nm',1e-13),('normalization_error',1e-10),('first_moment_error_m',1e-10)))
        need(s.get('passed')is bool(ok),'Independent surface quadrature predicate')
        if not ok:return False
    return True


def group_classification(rows,spatial,surfaces):
    complete=bool(len(rows)==33 and{(r['n'],r['label'])for r in rows}=={(n,l)for n in stage.LEVELS for l in stage.LOADS}and all(r.get('completed')is True for r in rows));surface=surfaces_pass(surfaces);references={}
    for label in stage.LOADS:
        members=[r for r in rows if r['label']==label];physical=bool(complete and surface and all(r['physical_passed']for r in members));space=bool(label in spatial and spatial[label]['spatial_development_passed'])
        references[label]=dict(physical_and_surface_passed=physical,spatial_passed=space,reference_valid=bool(physical and space),all_meshes_excitation_goal_met=bool(len(members)==3 and all(r['excitation_goal_met']for r in members)))
    return dict(all33_retained=complete,references=references,all11_reference_valid=bool(complete and all(v['reference_valid']for v in references.values())),all_fields_excitation_goal_met=bool(complete and all(r['excitation_goal_met']for r in rows)),bank_admitted=False)


def audit():
    start=time.monotonic();guard(start);base=stage.LOCAL;need(not(base/'independent-review.json').exists()and not(base/'compact-summary.json').exists(),'No audit overwrite/retry')
    p,s,phases,groups=metadata_barrier();compact_groups=[];field_count=reference_count=legacy_eligible=excited=0
    for item in groups:
        group=item['value'];fields={};rows=[]
        for row in group['rows']:
            # Pure same-version geometry/load validator; no old local data read.
            guard(start);u,d=field_audit.check_capture(row);guard(start);status=field_status(row,d);fields[row['n'],row['label']]=u
            rows.append(dict(n=row['n'],label=row['label'],completed=True,**status,minimum_sampled_J=d['minimum_J'],maximum_symmetric_strain=d['maximum_absolute_symmetric_strain_eigenvalue'],nonrigid_coordinate_RMS_m=d['nonrigid_RMS_m'],nonrigid_fraction=d['nonrigid_fraction'],capture=row['capture']))
        spatial=spatial_results(fields);guard(start);need(spatial==group['spatial'],'Independently recomputed all11 saved-field spatial metrics')
        status=group_classification(rows,spatial,{n:v['surface']for n,v in group['meshes'].items()});need(all(group[k]==v for k,v in status.items()),'Independent reference validity separate from excitation')
        field_count+=len(rows);reference_count+=sum(v['reference_valid']for v in status['references'].values());legacy_eligible+=sum(r['legacy_load_eligible']for r in rows);excited+=sum(r['excitation_goal_met']for r in rows)
        compact_groups.append(dict(index=item['index'],group=group['group'],result=item['result'],**status,rows=rows,spatial=spatial))
    all_valid=reference_count==88;need(s['all88_reference_valid']is all_valid and s['all_fields_excitation_goal_met']is(excited==264),'Honest complete reference/excitation summary flags')
    compact=dict(completed=True,all264_retained=field_count==264,new_fields=264,reused_fields=0,reference_valid_count=reference_count,all88_reference_valid=all_valid,legacy_load_eligible_field_count=legacy_eligible,excitation_goal_met_field_count=excited,all_fields_excitation_goal_met=excited==264,groups=compact_groups,bank_admitted=False,scope='Eight fixed interior DEVELOPMENT states/all11 loads. Weak excitation retained and legacy flags unchanged; new physical+surface+spatial reference validity is not old eligibility, population calibration or policy admission. J sampled; conditional spatial/Ritz numerical evidence only; raw fields LOCAL.')
    guard(start);stage.write(base/'compact-summary.json',compact)
    stage.write(base/'independent-review.json',dict(audit_passed=True,audit_resources=AUDIT_LIMITS,audit_wall_s=time.monotonic()-start,audit_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,all264_retained=field_count==264,all88_reference_valid=all_valid,source_SHA256=p['source_SHA256'],source_summary=stage.record(base/'summary-LOCAL.json'),audit_source=stage.record(stage.ROOT/SOURCE),audit_test=stage.record(stage.ROOT/TEST),compact_summary=stage.record(base/'compact-summary.json'),phases=phases,checks=dict(all264_hash_contexts_before_NPZ=True,all_root_workers_reaped_and_closed=True,known_force_geometry_query_wrench_and_sampled_deformation_reproduced=True,all88_spatial_and_reference_flags_reproduced=True,legacy_eligibility_and_excitation_preserved=True,no_PDE_factor_or_RNG_replay=True,sparse_residual_tangent_and_surface_Gram_witnesses_only=True,all_cases_actions_and_failures_retained=True)));return compact

if __name__=='__main__':print(json.dumps(audit()))
