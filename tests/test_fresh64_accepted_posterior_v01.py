"""Meaningful synthetic mixed-lineage/tamper controls; no study reads."""
from copy import deepcopy
import pytest
from experiments import campaign_fresh64_accepted_posterior_v01 as m


def diagnostics(kind,plan=None):
    inner,outer=(m.OLD_ORDERS,m.OLD_ORDERS)if plan is None else(plan['inner'],plan['outer'])
    levels=m.OLD_LEVELS if plan is None else m.extended_levels(plan)
    metric=dict(passed=True,log_evidence_difference=0.,maximum_CDF_difference=0.,maximum_unit_quantile_difference=0.,maximum_predictive_mean_difference_m=0.,maximum_predictive_SD_difference_m=0.,maximum_unit_mean_difference=0.)
    schema=('conditional-amplitude-v02'if kind=='linear'else'nonlinear-conditional-reference-v03')if plan is None else m.NEW_SCHEMAS[kind]
    return dict(schema=schema,qualified=True,gates=m.GATES.copy(),independent_adaptive_per_posterior_required=True,levels=[dict(outer=o,inner=i,status='computed',log_evidence=1.,weighted_hypotheses=3*o*i*(2 if kind=='linear'else 4))for o,i in levels],successive={a:[dict(coarse=c,fine=f,**metric)for c,f in zip(orders,orders[1:])]for a,orders in[('inner',inner),('outer',outer)]},independent_adaptive=dict(passed=True,epsrel=[1e-6,1e-8],CDF_checks_per_epsrel=6,log_evidence_difference=0.,maximum_CDF_error=0.,adaptive_refinement_log_evidence_difference=0.,adaptive_refinement_maximum_CDF_difference=0.,maximum_QUADPACK_relative_error_estimates={'1e-06':0.,'1e-08':0.}))


def record(kind,plan=None):
    d=diagnostics(kind,plan)
    backend=('conditional_amplitude_PWL_v02'if kind=='linear'else'nonlinear_conditional_reference_v03_MIS')if plan is None else m.NEW_BACKENDS[kind]
    r=dict(qualified=True,physical_valid=True,kind=kind,numerical_backend=backend,diagnostics=d,raw_SHA256='e'*64)
    r['strict_absolute_checks']=m.old.fresh.cost.absolute_engine_witnesses(dict(qualified=True,diagnostics=d))
    return r


@pytest.mark.parametrize('kind',['linear','nonlinear'])
@pytest.mark.parametrize('axis',['inner','outer','both'])
def test_6_or_7_level_extensions_preserve_coarse_negatives_and_strict_gates(kind,axis):
    plan=dict(inner=m.OLD_ORDERS+([1024]if axis in('inner','both')else[]),outer=m.OLD_ORDERS+([1024]if axis in('outer','both')else[]),moment_chunk=1024)
    r=record(kind,plan)
    r['diagnostics']['successive']['inner'][0].update(passed=False,maximum_CDF_difference=.2)
    assert m.numerical_record_checks(r,kind,source='refined_axis_extension',plan=plan)
    assert len(r['diagnostics']['levels'])==(7 if axis=='both'else 6)
    r['diagnostics']['successive']['inner'][-1]['maximum_CDF_difference']=.005000001
    with pytest.raises(ValueError):m.numerical_record_checks(r,kind,source='refined_axis_extension',plan=plan)


def packet(kind='nonlinear',refined=True):
    parent_context=dict(protocol_SHA256='a'*64,index=5,phase='normal-r0',draws_SHA256='b'*64,observations_SHA256='c'*64,model_fingerprint='d'*64,true_unit_SHA256='f'*64,recipe_SHA256='0'*64)
    original=record(kind);original['context']=parent_context
    binding=dict(json=dict(path='original.json',sha256='1'*64),raw=dict(path='original.npz',sha256='e'*64),context=deepcopy(parent_context))
    entry=dict(index=5,kind=kind,phase='normal-r0',id=f'05/{kind}/normal-r0',original=binding,decision='reuse_qualified')
    if not refined:return original,dict(source='original_five_levels',json=binding['json'],raw=binding['raw'],levels=[dict(outer=o,inner=i)for o,i in m.OLD_LEVELS]),entry,deepcopy(original)
    original['qualified']=False;original['diagnostics']['qualified']=False
    plan=dict(inner=m.OLD_ORDERS+[1024],outer=m.OLD_ORDERS,moment_chunk=1024);entry.update(decision='refine',plan=plan)
    r=record(kind,plan);r['context']=dict(protocol_SHA256='9'*64,index=5,phase='normal-r0',parent_protocol_SHA256=parent_context['protocol_SHA256'],parent_record_SHA256=binding['json']['sha256'],draws_SHA256=parent_context['draws_SHA256'],observations_SHA256=parent_context['observations_SHA256'],model_fingerprint=parent_context['model_fingerprint'],true_unit_SHA256=parent_context['true_unit_SHA256'],refinement_plan=plan,recipe_SHA256=m.refined_recipe(kind,plan,['target'],'atlas',binding['json']['sha256']))
    accepted=dict(source='refined_axis_extension',json=dict(path='new.json',sha256='2'*64),raw=dict(path='new.npz',sha256=r['raw_SHA256']),levels=[dict(outer=o,inner=i)for o,i in m.extended_levels(plan)],plan=plan)
    return r,accepted,entry,original


def verify(r,a,e,o):return m.verify_accepted_record(r,a,e,o,continuation_sha='9'*64,expected_modal_fingerprint='d'*64,targets=['target'],atlas_fingerprint='atlas')


def test_refined_same_observation_parent_and_recipe_bindings_detect_tamper():
    r,a,e,o=packet();assert verify(r,a,e,o)['same_draws_observations_and_model']
    for key in('observations_SHA256','draws_SHA256','true_unit_SHA256','model_fingerprint','recipe_SHA256','parent_record_SHA256','protocol_SHA256'):
        bad=deepcopy(r);bad['context'][key]='3'*64
        with pytest.raises(ValueError):verify(bad,a,e,o)
    bad=deepcopy(r);bad['numerical_backend']='nonlinear_conditional_reference_v03_MIS'
    with pytest.raises(ValueError):verify(bad,a,e,o)
    bad=deepcopy(r);bad['diagnostics']['levels'][-1]['inner']=512
    with pytest.raises(ValueError):verify(bad,a,e,o)


def test_qualified_original_reuse_is_exact_and_no_finer_relabelling():
    r,a,e,o=packet(refined=False);assert verify(r,a,e,o)['levels']==5
    bad=deepcopy(r);bad['raw_SHA256']='3'*64
    with pytest.raises(ValueError):verify(bad,a,e,o)
    bad=deepcopy(a);bad['json']['sha256']='3'*64
    with pytest.raises(ValueError):verify(r,bad,e,o)
    with pytest.raises(ValueError):m.numerical_record_checks(r,'nonlinear',source='refined_axis_extension',plan=dict(inner=m.OLD_ORDERS+[1024],outer=m.OLD_ORDERS,moment_chunk=1024))


def test_no_gate_relaxation_or_missing_final_level():
    r,a,e,o=packet();r['diagnostics']['gates']['CDF']=.006
    with pytest.raises(ValueError):verify(r,a,e,o)
    r,a,e,o=packet();r['diagnostics']['levels'][-1].update(status='unresolved',reason='failure')
    with pytest.raises(ValueError):verify(r,a,e,o)


def test_missing_comparisons_keep_all896_and64_denominators():
    rows=[];entries=[]
    for i in range(64):
        rows.append(dict(index=i,posteriors={k:{p:dict(qualified=False,physical_valid=True)for p in m.old.KEYS}for k in('linear','nonlinear')},posterior_lineage={k:{p:dict(accepted=None)for p in m.old.KEYS}for k in('linear','nonlinear')}))
        entries.extend(dict(id=f'{i:02d}/{k}/{p}')for k in('linear','nonlinear')for p in m.old.KEYS)
    result=m.membership_status(rows,entries)
    assert result==dict(objects=64,contexts=896,accepted=0,unresolved=896,comparison_blocked=True,all_original_denominators_retained=True)
    with pytest.raises(ValueError):m.membership_status(rows[:-1],entries)
    with pytest.raises(ValueError):m.membership_status(rows,entries[:-1]+entries[:1])
