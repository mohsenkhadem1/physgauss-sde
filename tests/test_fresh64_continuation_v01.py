"""Pure synthetic contexts only; no filesystem/cohort access or RNG."""
from copy import deepcopy
import pytest
from experiments import campaign_fresh64_continuation_v01 as m

OLD='a'*64;NEW='c'*64


def toy():
    records=[];bindings={}
    for i in range(64):
        row=dict(index=i,completed=True,physical_valid=True,posteriors={kind:{}for kind in m.MODELS})
        for kind in m.MODELS:
            for phase in m.PHASES:
                identity=m.context_id(i,kind,phase)
                r=dict(kind=kind,qualified=True,physical_valid=True,numerical_backend=m.OLD_BACKENDS[kind],raw_SHA256='b'*64,context=dict(protocol_SHA256=OLD,index=i,phase=phase,draws_SHA256='d'*64,observations_SHA256='e'*64,model_fingerprint='f'*64,true_unit_SHA256='0'*64),diagnostics=dict(schema=m.OLD_SCHEMAS[kind],levels=[dict(outer=a,inner=b,status='computed')for a,b in m.OLD_LEVELS],successive={axis:[dict(passed=True,coarse=256,fine=512)]for axis in('inner','outer')}),strict_absolute_checks=dict(final_axis_witnesses={axis:dict(passed=True)for axis in('inner','outer')}),PIT=[.2,.7],covered=[True,False],MAP_prediction=['unaltered-raw-reference'])
                row['posteriors'][kind][phase]=r;bindings[identity]=dict(json=dict(path='parent/'+identity+'.json',sha256='1'*64),raw=dict(path='parent/'+identity+'.npz',sha256='b'*64))
        records.append(row)
    return records,bindings


def negative(records,i,kind,phase,axis=None):
    r=records[i]['posteriors'][kind][phase];r['qualified']=False
    if axis:r['diagnostics']['successive'][axis][-1]['passed']=False
    return r


def refined(entry,*,qualified=True):
    old=entry['original']['context'];context=dict(protocol_SHA256=NEW,index=entry['index'],phase=entry['phase'],parent_protocol_SHA256=OLD,parent_record_SHA256=entry['original']['json']['sha256'],draws_SHA256=old['draws_SHA256'],observations_SHA256=old['observations_SHA256'],model_fingerprint=old['model_fingerprint'],true_unit_SHA256=old['true_unit_SHA256'],refinement_plan=entry['plan'])
    r=dict(kind=entry['kind'],physical_valid=True,qualified=qualified,raw_SHA256='2'*64,context=context,diagnostics=dict(levels=entry['levels']),PIT=[.3,.6],covered=[True,True])
    return dict(record=r,plan=entry['plan'],binding=dict(json=dict(path='continued/'+entry['id']+'.json',sha256='3'*64),raw=dict(path='continued/'+entry['id']+'.npz',sha256='2'*64)))


def test_all896_original_qualified_reused_bitwise_records_and_no_jobs():
    records,bindings=toy();before=deepcopy(records);c=m.census(records,bindings)
    assert c['counts']==dict(reuse_qualified=896,refine=0,blocked=0)
    merged=m.merge_accepted(records,c,{},protocol_SHA256=NEW)
    assert merged['reused_qualified']==896 and merged['newly_qualified']==0 and not merged['comparison_blocked']
    for i in range(64):assert merged['rows'][i]['posteriors']==records[i]['posteriors']
    assert records==before
    path=m.accepted_posterior(merged,7,'linear','prefix');assert path['source']=='original_five_levels'and path['raw']==bindings['07/linear/prefix']['raw']


def test_all_failed_both_models_per_axis_and_fallback_no_exclusion():
    records,bindings=toy()
    negative(records,5,'nonlinear','normal-r0','inner');negative(records,1,'nonlinear','axial-r0','outer');negative(records,1,'linear','axial-r0','outer');negative(records,2,'linear','prefix')
    c=m.census(records,bindings);jobs={e['id']:e for e in c['entries']if e['decision']=='refine'}
    assert len(jobs)==4 and c['counts']['reuse_qualified']==892
    assert jobs['05/nonlinear/normal-r0']['plan']['inner'][-1]==1024 and jobs['05/nonlinear/normal-r0']['plan']['outer'][-1]==512
    assert jobs['01/linear/axial-r0']['backend_family']=='frozen_linear_V02_outer_extension'
    assert jobs['02/linear/prefix']['backend_family']=='extended_linear_inner'
    assert [len(jobs[v]['levels'])for v in('05/nonlinear/normal-r0','01/linear/axial-r0','02/linear/prefix')]==[6,6,7]
    assert jobs['02/linear/prefix']['failed_axes']==['inner','outer']


def test_absolute_witness_selects_axis_even_when_relative_gate_passed():
    records,bindings=toy();r=negative(records,3,'linear','normal-r1');r['strict_absolute_checks']['final_axis_witnesses']['inner']['passed']=False
    e=next(e for e in m.census(records,bindings)['entries']if e['id']=='03/linear/normal-r1')
    assert e['failed_axes']==['inner']


def test_mixed_new_six_seven_levels_lineage_preserves_all_old_records():
    records,bindings=toy();negative(records,5,'nonlinear','normal-r0','inner');negative(records,1,'linear','axial-r0','outer');negative(records,2,'nonlinear','prefix')
    before=deepcopy(records);c=m.census(records,bindings);jobs=[e for e in c['entries']if e['decision']=='refine'];updates={e['id']:refined(e)for e in jobs}
    merged=m.merge_accepted(records,c,updates,protocol_SHA256=NEW)
    assert records==before and merged['reused_qualified']==893 and merged['newly_qualified']==3 and not merged['comparison_blocked']
    new=m.accepted_posterior(merged,5,'nonlinear','normal-r0');assert new['source']=='refined_axis_extension'and len(new['levels'])==6
    assert len(m.accepted_posterior(merged,2,'nonlinear','prefix')['levels'])==7
    assert merged['rows'][5]['posteriors']['linear']['prefix']==records[5]['posteriors']['linear']['prefix']


def test_missing_invalid_failed_refinements_keep64_and896_denominator():
    records,bindings=toy();negative(records,5,'nonlinear','normal-r0','inner');records[3]['physical_valid']=False;del records[4]['posteriors']['linear']['prefix'];records.pop()
    c=m.census(records,bindings);job=next(e for e in c['entries']if e['decision']=='refine')
    merged=m.merge_accepted(records,c,{job['id']:refined(job,qualified=False)},protocol_SHA256=NEW)
    assert merged['objects']==64 and merged['contexts']==896 and merged['unresolved']==30 and merged['comparison_blocked']
    assert merged['rows'][3]['posteriors']['linear']['prefix']['qualified']is False
    with pytest.raises(ValueError,match='Unresolved'):m.accepted_posterior(merged,5,'nonlinear','normal-r0')


def test_no_qualified_original_recomputation_or_changed_observation_binding():
    records,bindings=toy();negative(records,5,'nonlinear','normal-r0','inner');c=m.census(records,bindings);e=next(e for e in c['entries']if e['decision']=='refine')
    packet=refined(e);packet['record']['context']['observations_SHA256']='4'*64
    with pytest.raises(ValueError,match='Same parent observations'):m.merge_accepted(records,c,{e['id']:packet},protocol_SHA256=NEW)
    with pytest.raises(ValueError,match='No recomputation'):m.merge_accepted(records,c,{'00/linear/prefix':refined(e)},protocol_SHA256=NEW)


def test_original_and_refined_missing_claimed_raw_are_not_accepted():
    records,bindings=toy();bindings['00/linear/prefix']['raw']=None;c=m.census(records,bindings)
    assert c['counts']['blocked']==1
    negative(records,5,'nonlinear','normal-r0','inner');c=m.census(records,bindings);e=next(e for e in c['entries']if e['decision']=='refine');packet=refined(e);packet['binding']['raw']['sha256']='9'*64
    with pytest.raises(ValueError,match='archive commitment'):m.merge_accepted(records,c,{e['id']:packet},protocol_SHA256=NEW)


def test_duplicate_objects_or_unplanned_bindings_refused():
    records,bindings=toy()
    with pytest.raises(ValueError,match='duplicated'):m.census(records+[records[0]],bindings)
    bindings['64/linear/prefix']={}
    with pytest.raises(ValueError,match='Unexpected'):m.census(records,bindings)


def test_census_and_accepted_lineage_JSON_roundtrip():
    import json
    records,bindings=toy();negative(records,5,'nonlinear','normal-r0','inner')
    c=json.loads(json.dumps(m.census(records,bindings)))
    e=next(v for v in c['entries']if v['decision']=='refine')
    merged=m.merge_accepted(records,c,{e['id']:refined(e)},protocol_SHA256=NEW)
    roundtrip=json.loads(json.dumps(merged))
    assert m.accepted_posterior(roundtrip,0,'linear','prefix')==m.accepted_posterior(merged,0,'linear','prefix')
    assert m.accepted_posterior(roundtrip,5,'nonlinear','normal-r0')==m.accepted_posterior(merged,5,'nonlinear','normal-r0')
