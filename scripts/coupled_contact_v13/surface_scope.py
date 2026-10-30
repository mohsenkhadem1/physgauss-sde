"""Runtime authorization for prospectively fixed primary/stress scenario scopes."""
import hashlib,json
from pathlib import Path


def partition(p):
    full=p['forecast_scenarios'];primary=p['primary_forecast_scenarios'];stress=p['stress_forecast_scenarios']
    if len(full)!=4 or len(set(full))!=4 or not primary or len(set(primary))!=len(primary) or len(set(stress))!=len(stress) or set(primary)&set(stress) or set(primary)|set(stress)!=set(full):raise RuntimeError('Invalid fixed primary/stress partition')
    core={'slide','curved','moving_sphere'}
    if not core<=set(full) or set(full)-core not in ({'edge_drop'},{'rounded_support_drop'}):raise RuntimeError('Unrecognized scenario matrix')
    if stress and (set(primary)!=core or set(stress)!={'edge_drop'}):raise RuntimeError('Only sharp edge_drop is authorized as stress cohort')
    return primary,stress


def artifact(root,ref):
    relative=Path(ref['path'])
    if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Evidence path must be repository relative')
    path=root/relative
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise RuntimeError('Evidence hash mismatch: '+str(relative))
    return json.loads(path.read_text())


def validate_adoption(p,q,root):
    primary,stress=partition(p)
    scope='primary_with_stress' if stress else 'all_forecast_primary'
    if p.get('adopted_scope')!=scope or q.get('adopted_scope')!=scope:raise RuntimeError('Explicit matching adopted scope required')
    for key in ('primary_forecast_scenarios','stress_forecast_scenarios'):
        if q.get(key)!=p[key]:raise RuntimeError('Qualification partition differs from protocol')
    if q.get('primary_gates_pass') is not True:raise RuntimeError('Primary development qualification has not passed')
    if q.get('all_gates_pass') is not (not bool(stress)):raise RuntimeError('Overall qualification must retain stress limitation')
    decision=artifact(root,q['decision'])
    if decision.get('status')!='adopted' or decision.get('adopted_scope')!=scope:raise RuntimeError('Explicit adoption decision required')
    for key in ('primary_forecast_scenarios','stress_forecast_scenarios'):
        if decision.get(key)!=p[key]:raise RuntimeError('Adoption decision scenario mismatch')
    evidence=q['primary_evidence']
    if set(evidence)!=set(primary)|{'squeeze'}:raise RuntimeError('Evidence must cover calibration and every fixed primary scenario')
    for scene,refs in evidence.items():
        if not isinstance(refs,list) or not refs:raise RuntimeError('Missing qualified primary evidence: '+scene)
        specific=False
        for ref in refs:
            document=artifact(root,ref);value=document;keys=ref.get('passed_key')
            if not isinstance(keys,list) or not keys:raise RuntimeError('Explicit pass outcome path required')
            for key in keys:value=value[key]
            if value is not True:raise RuntimeError('Referenced primary evidence did not pass: '+scene)
            if keys==['scenarios',scene,'qualified']:
                record=document['scenarios'][scene];settings=p['reference_settings_by_scenario'][scene]
                for field,expected in [('primary_cells',settings['cells']),('companion_cells',settings['companion_cells']),('dt',settings['dt'])]:
                    actual=record[field]
                    if isinstance(expected,list) and isinstance(actual,int):actual=[actual]*3
                    if actual!=expected:raise RuntimeError('Primary evidence discretization mismatch: '+scene+'/'+field)
                specific=True
        if not specific:raise RuntimeError('Scene-specific qualification evidence required: '+scene)
    return scope


def check_qualification(report,root):
    p=json.loads((report/'protocol.json').read_text());freeze=json.loads((report/'method-freeze.json').read_text())
    path=report/'physics-development-qualification.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=freeze['physics_qualification_sha256']:raise RuntimeError('Qualification is not frozen exactly')
    return validate_adoption(p,json.loads(path.read_text()),root)
