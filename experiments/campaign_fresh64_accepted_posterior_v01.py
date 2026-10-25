"""Pure mixed-lineage checks for a separate numerical continuation audit.

No I/O, arrays, RNG, models or inference. This accepts neither a cohort nor a
file by itself: the caller first verifies all terminal/hash commitments, then
uses frozen independent truth/metric checks on the accepted saved archives.
Original five-level results remain immutable; refinements have explicit new
six/seven-level records and unchanged likelihood/prior/admission thresholds.
"""
import hashlib,json
import numpy as np
from experiments import audit_fresh64_nonlinear_calibration_v01 as old

OLD_ORDERS=[128,256,512]
OLD_LEVELS=[(512,128),(512,256),(512,512),(128,512),(256,512)]
NEW_BACKENDS=dict(linear='linear_amplitude_extended_v01_axis_extension',nonlinear='nonlinear_conditional_reference_v04_MIS_axis_extension')
NEW_SCHEMAS=dict(linear='conditional-amplitude-linear-extended-v01',nonlinear='nonlinear-conditional-reference-v04')
GATES=dict(logZ=.01,CDF=.005,quantile=.002,unit_mean=.001,target_absolute_m=2e-6,target_relative_SD=.01)
demand=old.demand


def extended_levels(plan):
    demand(set(plan)=={'inner','outer','moment_chunk'}and plan['moment_chunk']==1024,'Exact extended numerical plan fields/chunk')
    for axis in('inner','outer'):demand(plan[axis]in(OLD_ORDERS,OLD_ORDERS+[1024]),'Original orders or appended1024 only')
    demand(any(plan[a][-1]==1024 for a in('inner','outer')),'Refinement cannot relabel original512')
    inner,outer=plan['inner'],plan['outer']
    return list(dict.fromkeys([(outer[-1],n)for n in inner]+[(n,inner[-1])for n in outer]))


def numerical_record_checks(record,kind,*,source,plan=None):
    demand(kind in('linear','nonlinear')and record.get('kind')==kind and record.get('qualified')is True and record.get('physical_valid')is True,'Same physically valid qualified model family')
    if source=='original_five_levels':
        demand(plan is None,'No refined plan grafted onto immutable original')
        demand(record.get('numerical_backend')==('conditional_amplitude_PWL_v02'if kind=='linear'else'nonlinear_conditional_reference_v03_MIS'),'Exact frozen original backend')
        return old.numerical_record_checks(record,kind)
    demand(source=='refined_axis_extension'and plan is not None,'Explicit refined lineage and plan required')
    expected=extended_levels(plan);d=record['diagnostics']
    demand(record.get('numerical_backend')==NEW_BACKENDS[kind]and d.get('schema')==NEW_SCHEMAS[kind],'Explicit new version/schema, never masquerade as old512')
    demand(d.get('qualified')is True and d.get('independent_adaptive_per_posterior_required')is True and d.get('gates')==GATES,'Original exact rules and adaptive requirement')
    levels=d.get('levels',[])
    demand([(v.get('outer'),v.get('inner'))for v in levels]==expected,'Exact six/seven integration levels in declared order')
    last_i,last_o=plan['inner'][-1],plan['outer'][-1]
    needed={(last_o,plan['inner'][-2]),(last_o,last_i),(plan['outer'][-2],last_i)}
    for v in levels:
        identity=(v['outer'],v['inner'])
        if v.get('status')=='computed':
            demand(np.isfinite(v.get('log_evidence',np.nan))and type(v.get('weighted_hypotheses'))is int and v['weighted_hypotheses']>0,'Resolved finite positive-hypothesis level')
        else:demand(identity not in needed and v.get('status')=='unresolved'and isinstance(v.get('reason'),str)and bool(v['reason']),'Only coarse unresolved levels, with retained reason')
    demand(set(d.get('successive',{}))=={'inner','outer'},'Both refinement axes retained')
    errors=('log_evidence_difference','maximum_CDF_difference','maximum_unit_quantile_difference','maximum_predictive_mean_difference_m','maximum_predictive_SD_difference_m','maximum_unit_mean_difference')
    for axis in('inner','outer'):
        history=d['successive'][axis];orders=plan[axis]
        demand([(v.get('coarse'),v.get('fine'))for v in history]==list(zip(orders,orders[1:])),'Exact original and extra order pairs')
        for v in history:
            demand(type(v.get('passed'))is bool,'Preserve every comparison admission flag')
            if 'reason'in v:demand(v['passed']is False and isinstance(v['reason'],str)and bool(v['reason']),'Unresolved coarse comparison reason preserved')
            else:demand(all(isinstance(v.get(k),(float,int))and np.isfinite(v[k])and v[k]>=0 for k in errors),'Finite nonnegative numerical diagnostics')
    result=dict(qualified=True,diagnostics=d)
    demand(old.fresh.cost.engine_gates(result)['passed'],'All original/adaptive/finalaxis/actual2um gates, no relaxation')
    old.compare(record['strict_absolute_checks'],old.fresh.cost.absolute_engine_witnesses(result),'Refined strict2um witnesses')
    return True


def refined_recipe(kind,plan,targets,atlas_fingerprint,parent_record_sha):
    demand(kind in NEW_BACKENDS,'Known continued numerical backend')
    payload=dict(kind=kind,numerical_backend=NEW_BACKENDS[kind],plan=plan,targets=targets,atlas=atlas_fingerprint if kind=='nonlinear'else None,parent_record_SHA256=parent_record_sha)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def verify_accepted_record(record,accepted,entry,original,*,continuation_sha,expected_modal_fingerprint,targets,atlas_fingerprint):
    """Check one accepted JSON against its immutable parent and frozen census.

    File byte/hash checks are mandatory in the terminal reader BEFORE calling
    this pure verifier. Old record comparison is exact canonical JSON equality.
    Refinement observations/draws/true-unit identities must remain the same.
    """
    i,kind,phase=entry['index'],entry['kind'],entry['phase']
    demand(type(i)is int and 0<=i<64 and kind in('linear','nonlinear')and phase in old.KEYS and entry['id']==f'{i:02d}/{kind}/{phase}','Fixed64×14 context membership')
    parent=entry['original'];context=original['context']
    demand(context==parent['context']and context['index']==i and context['phase']==phase and context['model_fingerprint']==expected_modal_fingerprint,'Original hash-bound context and canonical CPU-modal identity')
    demand(accepted is not None,'Unresolved contexts have no accepted archive; denominator retained elsewhere')
    source=accepted.get('source')
    if source=='original_five_levels':
        demand(entry['decision']=='reuse_qualified'and original.get('qualified')is True,'Reuse only original qualified records')
        demand(accepted['json']==parent['json']and accepted['raw']==parent['raw']and [(v['outer'],v['inner'])for v in accepted['levels']]==OLD_LEVELS,'Exact original JSON/raw/five-level lineage')
        demand(json.dumps(record,sort_keys=True,allow_nan=False)==json.dumps(original,sort_keys=True,allow_nan=False),'Qualified original content cannot be recomputed or replaced')
        numerical_record_checks(record,kind,source=source)
    else:
        demand(source=='refined_axis_extension'and entry['decision']=='refine'and original.get('qualified')is False,'Only failed originals eligible for refinement')
        plan=entry['plan'];levels=extended_levels(plan)
        demand(accepted.get('plan')==plan and [(v['outer'],v['inner'])for v in accepted['levels']]==levels,'Explicit six/seven-level accepted plan')
        recipe=refined_recipe(kind,plan,targets,atlas_fingerprint,parent['json']['sha256'])
        expected=dict(protocol_SHA256=continuation_sha,index=i,phase=phase,parent_protocol_SHA256=context['protocol_SHA256'],parent_record_SHA256=parent['json']['sha256'],draws_SHA256=context['draws_SHA256'],observations_SHA256=context['observations_SHA256'],model_fingerprint=expected_modal_fingerprint,true_unit_SHA256=context['true_unit_SHA256'],refinement_plan=plan,recipe_SHA256=recipe)
        demand(record.get('context')==expected,'Exact continued protocol/recipe/parent/same-observation/noise/parameter identity')
        demand(accepted['raw']['sha256']==record['raw_SHA256'],'Accepted posterior raw checksum identity')
        numerical_record_checks(record,kind,source=source,plan=plan)
    return dict(id=entry['id'],source=source,kind=kind,levels=len(accepted['levels']),parent_preserved=True,same_draws_observations_and_model=True,numerical_gates_verified=True)


def membership_status(rows,entries):
    """Retain all64/896 slots, including unresolved entries; never filter them."""
    expected={f'{i:02d}/{m}/{p}'for i in range(64)for m in('linear','nonlinear')for p in old.KEYS}
    demand(len(rows)==64 and all(type(r.get('index'))is int for r in rows)and [r.get('index')for r in rows]==list(range(64)),'All64 ordered object denominators')
    demand(len(entries)==896 and {e.get('id')for e in entries}==expected,'All896 original context memberships exactly once')
    accepted=0
    for row in rows:
        for field in('posteriors','posterior_lineage'):
            demand(set(row.get(field,{}))=={'linear','nonlinear'}and all(set(row[field][m])==set(old.KEYS)for m in('linear','nonlinear')),'Bothmodels/allseven contexts; no failed-entry omission')
        for m in('linear','nonlinear'):
            for p in old.KEYS:
                a=row['posterior_lineage'][m][p].get('accepted');r=row['posteriors'][m][p]
                if a is None:demand(r.get('qualified')is not True,'Unresolved path cannot claim admission')
                else:demand(r.get('qualified')is True and r.get('physical_valid')is True,'Accepted path requires physical/numerical admission');accepted+=1
    return dict(objects=64,contexts=896,accepted=accepted,unresolved=896-accepted,comparison_blocked=accepted!=896,all_original_denominators_retained=True)
