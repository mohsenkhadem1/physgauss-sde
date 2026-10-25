"""Pure numerical census/lineage merge; no I/O, RNG, PDE or inference.

All64×14 original slots survive. Resolution plans depend ONLY on numerical
admission statuses, never coverage, PIT, truth, target error or action scores.
Qualified original posterior records and archives are reused unchanged.
"""
from copy import deepcopy

MODELS=('linear','nonlinear')
PHASES=('prefix','normal-r0','normal-r1','normal-r2','axial-r0','axial-r1','axial-r2')
OLD_ORDERS=[128,256,512]
OLD_LEVELS=[(512,128),(512,256),(512,512),(128,512),(256,512)]
OLD_BACKENDS=dict(linear='conditional_amplitude_PWL_v02',nonlinear='nonlinear_conditional_reference_v03_MIS')
OLD_SCHEMAS=dict(linear='conditional-amplitude-v02',nonlinear='nonlinear-conditional-reference-v03')


def context_id(index,kind,phase):
    if type(index)is not int or not 0<=index<64 or kind not in MODELS or phase not in PHASES:raise ValueError('Original64/two-model/seven-context membership')
    return f'{index:02d}/{kind}/{phase}'


def _bound_record(value):
    return isinstance(value,dict)and isinstance(value.get('path'),str)and bool(value['path'])and isinstance(value.get('sha256'),str)and len(value['sha256'])==64 and all(c in'0123456789abcdef'for c in value['sha256'])


def integration_levels(plan):
    inner=plan['inner'];outer=plan['outer']
    for values in(inner,outer):
        if values not in(OLD_ORDERS,OLD_ORDERS+[1024]):raise ValueError('Only fixed original orders or appended1024')
    if inner==OLD_ORDERS and outer==OLD_ORDERS:raise ValueError('Continuation must extend at leastone axis')
    return [dict(outer=a,inner=b)for a,b in dict.fromkeys([(outer[-1],n)for n in inner]+[(n,inner[-1])for n in outer])]


def resolution_plan(record):
    """Deterministic extension from last numerical/absolute axis statuses only."""
    if record.get('qualified')is not False:raise ValueError('Only an original numerical negative is eligible')
    diagnostics=record.get('diagnostics',{});successive=diagnostics.get('successive',{})
    witnesses=record.get('strict_absolute_checks',{}).get('final_axis_witnesses',{})
    failed=[]
    for axis in('inner','outer'):
        rows=successive.get(axis,[]);last=rows[-1]if rows else{}
        if last.get('passed')is not True or witnesses.get(axis,{}).get('passed')is not True:failed.append(axis)
    fallback=not failed
    if fallback:failed=['inner','outer']
    plan=dict(inner=OLD_ORDERS+([1024]if'inner'in failed else[]),outer=OLD_ORDERS+([1024]if'outer'in failed else[]),moment_chunk=1024)
    return dict(plan=plan,failed_axes=failed,axis_selection='Bothaxes for independent/admission failure with no failed final axis'if fallback else'Original final successive or strict-absolute axis status',levels=integration_levels(plan))


def backend_family(kind,plan):
    if kind=='nonlinear':return 'nonlinear_V04_MIS'
    if kind=='linear':return 'extended_linear_inner'if plan['inner'][-1]==1024 else'frozen_linear_V02_outer_extension'
    raise ValueError('Both original model families, no unsupported-model exclusion')


def _rows_by_index(records):
    by={}
    for row in records:
        i=row.get('index')
        if type(i)is not int or not 0<=i<64 or i in by:raise ValueError('No duplicated/replaced/out-of-plan objects')
        by[i]=row
    return by


def census(records,bindings):
    """Bindings map context_id to {json:record,raw:record|None} after parent audit.

    A deliberately absent raw archive for an unqualified no-final-state record
    is distinct from a missing archive that the parent claims by checksum.
    Missing/invalid commitments become explicit hardblocked slots, never drops.
    """
    by=_rows_by_index(records);entries=[]
    expected={context_id(i,m,p)for i in range(64)for m in MODELS for p in PHASES}
    if set(bindings)-expected:raise ValueError('Unexpected posterior binding membership')
    for i in range(64):
        parent=by.get(i)
        if parent is not None:
            post=parent.get('posteriors',{})
            if set(post)-set(MODELS)or any(set(post.get(m,{}))-set(PHASES)for m in MODELS):raise ValueError('Unexpected original model/context membership')
        for kind in MODELS:
            for phase in PHASES:
                identity=context_id(i,kind,phase);r=None if parent is None else parent.get('posteriors',{}).get(kind,{}).get(phase)
                entry=dict(id=identity,index=i,kind=kind,phase=phase,decision='blocked',hard_block=True)
                supplied=bindings.get(identity,{})
                if r is not None and _bound_record(supplied.get('json')):entry['original']=dict(json=deepcopy(supplied['json']),raw=deepcopy(supplied.get('raw')),context=deepcopy(r.get('context',{})),backend=r.get('numerical_backend'))
                reason=None
                if parent is None or r is None:reason='Missing original object or context'
                elif parent.get('physical_valid')is not True or r.get('physical_valid')is not True:reason='Original physical-invalid context retained'
                elif r.get('kind')!=kind or r.get('numerical_backend')!=OLD_BACKENDS[kind]or r.get('diagnostics',{}).get('schema')!=OLD_SCHEMAS[kind]:reason='Unrecognized original numerical/source schema'
                elif type(r.get('qualified'))is not bool:reason='Missing original admission status'
                elif r.get('context',{}).get('index')!=i or r.get('context',{}).get('phase')!=phase:reason='Mismatched original context identity'
                else:
                    binding=bindings.get(identity,{})
                    if not _bound_record(binding.get('json')):reason='Missing original posterior JSON commitment'
                    elif r.get('raw_SHA256')and(not _bound_record(binding.get('raw'))or binding['raw']['sha256']!=r['raw_SHA256']):reason='Missing or mismatched claimed original posterior archive'
                    elif r.get('qualified')is True and not _bound_record(binding.get('raw')):reason='Qualified original lacks bound raw archive'
                    else:
                        original=dict(json=deepcopy(binding['json']),raw=deepcopy(binding.get('raw')),context=deepcopy(r.get('context',{})),backend=r['numerical_backend'],levels=[dict(outer=v.get('outer'),inner=v.get('inner'),status=v.get('status'))for v in r['diagnostics'].get('levels',[])])
                        if [(v['outer'],v['inner'])for v in original['levels']]!=OLD_LEVELS:reason='Original five-level lineage does not match frozen plan'
                        else:
                            entry.update(original=original,hard_block=False)
                            if r['qualified']:entry.update(decision='reuse_qualified',original_qualified=True)
                            else:
                                selected=resolution_plan(r);entry.update(decision='refine',original_qualified=False,**selected);entry['backend_family']=backend_family(kind,selected['plan'])
                if reason:entry['reason']=reason
                entries.append(entry)
    counts={decision:sum(v['decision']==decision for v in entries)for decision in('reuse_qualified','refine','blocked')}
    return dict(objects=64,contexts=896,entries=entries,counts=counts,comparison_blocked=bool(counts['blocked']or counts['refine']),numerical_selection_only=True,no_replacements=True)


def _validate_update(entry,update,protocol_SHA256):
    if update.get('plan')!=entry['plan']:raise ValueError('Exact predeclared per-axis refinement plan')
    r=update.get('record',{});binding=update.get('binding',{});c=r.get('context',{});old=entry['original']['context']
    if not _bound_record(binding.get('json'))or r.get('kind')!=entry['kind']or r.get('physical_valid')is not True:raise ValueError('New record identity/physical admission/commitment')
    expected=dict(protocol_SHA256=protocol_SHA256,index=entry['index'],phase=entry['phase'],parent_protocol_SHA256=old['protocol_SHA256'],parent_record_SHA256=entry['original']['json']['sha256'],draws_SHA256=old['draws_SHA256'],observations_SHA256=old['observations_SHA256'],model_fingerprint=old['model_fingerprint'],refinement_plan=entry['plan'])
    if 'true_unit_SHA256'in old:expected['true_unit_SHA256']=old['true_unit_SHA256']
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Same parent observations/draws/model and explicit new lineage')
    if r.get('raw_SHA256')and(not _bound_record(binding.get('raw'))or binding['raw']['sha256']!=r['raw_SHA256']):raise ValueError('New archive commitment')
    if r.get('qualified')is True:
        if not _bound_record(binding.get('raw')):raise ValueError('Qualified refinement needs committed raw')
        got=[dict(outer=v.get('outer'),inner=v.get('inner'))for v in r.get('diagnostics',{}).get('levels',[])]
        if got!=entry['levels']:raise ValueError('Exact six/seven-level refined lineage, never relabel as512')
    return r,binding


def merge_accepted(original_records,classification,updates,*,protocol_SHA256):
    """Merge accepted paths without editing any parent posterior record/context.

    Caller independently verifies every numerical gate/PIT before marking a new
    record qualified. Continued auditor rechecks those rules before scoring.
    """
    by=_rows_by_index(original_records);entries=classification['entries'];expected={context_id(i,m,p)for i in range(64)for m in MODELS for p in PHASES}
    if len(entries)!=896 or {e['id']for e in entries}!=expected:raise ValueError('ALL896 census entries required')
    jobs={e['id']for e in entries if e['decision']=='refine'}
    if set(updates)-jobs:raise ValueError('No recomputation/replacement of qualified originals or extra contexts')
    rows=[];accepted_old=accepted_new=unresolved=0
    for i in range(64):
        parent=by.get(i,dict(index=i,completed=False,physical_valid=False));row=deepcopy(parent);row['posteriors']={m:{}for m in MODELS};row['posterior_lineage']={m:{}for m in MODELS};row['parent_completed']=parent.get('completed',False);row['continuation_protocol_SHA256']=protocol_SHA256;row['completed']=bool(parent.get('completed',False))
        for e in entries:
            if e['index']!=i:continue
            kind,phase=e['kind'],e['phase'];old=parent.get('posteriors',{}).get(kind,{}).get(phase,dict(qualified=False,physical_valid=False,attempted=False));accepted=deepcopy(old)
            lineage=dict(original=deepcopy(e.get('original')),decision=e['decision'],accepted=None)
            if e['decision']=='reuse_qualified':
                if old.get('qualified')is not True:raise ValueError('Changed original admission status after census')
                lineage['accepted']=dict(source='original_five_levels',json=e['original']['json'],raw=e['original']['raw'],levels=[dict(outer=a,inner=b)for a,b in OLD_LEVELS]);accepted_old+=1
            elif e['decision']=='refine'and e['id']in updates:
                candidate,binding=_validate_update(e,updates[e['id']],protocol_SHA256);lineage['refinement_attempt']=dict(binding=deepcopy(binding),plan=deepcopy(e['plan']))
                if candidate.get('qualified')is True:
                    accepted=deepcopy(candidate);lineage['accepted']=dict(source='refined_axis_extension',json=binding['json'],raw=binding['raw'],levels=e['levels'],plan=e['plan']);accepted_new+=1
                else:unresolved+=1
            else:
                unresolved+=1
                if e['decision']=='refine':row['completed']=False
                if e['decision']=='blocked':accepted=dict(qualified=False,physical_valid=False,attempted=False,reason=e.get('reason','Original context blocked'))
            row['posteriors'][kind][phase]=accepted;row['posterior_lineage'][kind][phase]=lineage
        rows.append(row)
    return dict(rows=rows,objects=64,contexts=896,reused_qualified=accepted_old,newly_qualified=accepted_new,unresolved=unresolved,comparison_blocked=unresolved>0,all_original_denominators_retained=True)


def accepted_posterior(merged,index,kind,phase):
    context_id(index,kind,phase);row=merged['rows'][index]
    if row.get('index')!=index:raise ValueError('Fixed object order')
    path=row['posterior_lineage'][kind][phase]['accepted']
    if path is None:raise ValueError('Unresolved context has no accepted posterior path')
    return deepcopy(path)
