"""Reuse every frozen base audit, adapting only its two reset-baseline keys.

The base validator assumes a candidate named poke_0; numerical captures instead
name that zero-force candidate zero_force. Source adaptation is explicit,
bounded to exactly two dictionary accesses, and saved in full with provenance.
Original execution errors, captures, base files and tolerances stay intact.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np


BASE_SHA256='116d4717bff0c7780925bd4e6d85f5b2f1615d33f4993e356a8284199829a821'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapt_reset_reference(source, candidate_id):
    old="saved['poke_0']"
    if source.count(old)!=2:raise ValueError('Expected exactly two fixed reset-reference accesses')
    new='saved['+repr(candidate_id)+']'
    adapted=source.replace(old,new)
    if adapted.replace(new,old)!=source:raise ValueError('Unbounded validator adaptation')
    return adapted


def validate(output):
    import isaac_calibrated_probes_v03 as base
    from campaign_numerical_convergence import analyze
    output=Path(output)
    if sha(base.__file__)!=BASE_SHA256:raise ValueError('Frozen base validator changed')
    ledger=json.loads((output/'execution.json').read_text())
    if [r['physics_hz'] for r in ledger['cases']] != [480,960,1920] or any('exit_code' not in r for r in ledger['cases']):
        raise ValueError('All three simulation processes must finish before post-validation')
    if any(r['exit_code'] for r in ledger['cases']):raise ValueError('Failed simulator exit cannot be repaired by validator adaptation')
    provenance=output/'numerical-validation-provenance';provenance.mkdir(exist_ok=False)
    source=inspect.getsource(base.validate_case)
    adapted=adapt_reset_reference(source,'zero_force')
    path=provenance/'adapted_validate_case.py';path.write_text(adapted)
    namespace=dict(vars(base));exec(compile(adapted,str(path),'exec'),namespace)
    function=namespace['validate_case']
    report={'base_source_sha256':BASE_SHA256,'base_function_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'adapted_function_sha256':sha(path),'adapter_source_sha256':sha(__file__),
        'changes':["Two saved['poke_0'] reset references become saved['zero_force']"],
        'unchanged':'All force, timestep, mesh, support, clock, source, capture-hash, sensor ray, camera and replay tolerances/checks',
        'original_execution_sha256':sha(output/'execution.json'),'rates':[]}
    for hz in [480,960,1920]:
        root=output/f'hz{hz}'/'captures'
        menu=json.loads((root/'public/t0001/action-menu.json').read_text())
        if [a['action_id'] for a in menu['candidates']]!=['zero_force'] or np.any(menu['candidates'][0]['peak_force_N']):
            raise ValueError('Unexpected numerical candidate/menu; do not adapt assumptions')
        result=function(root,'t0001')
        result['numerical_validator_adaptation_sha256']=report['adapted_function_sha256']
        result['base_validator_assumption_failure_preserved']='Original execution.json retains KeyError(poke_0)'
        target=root/'private/t0001/validation.json'
        with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        report['rates'].append({'physics_hz':hz,'passed':result['passed'],'validation_sha256':sha(target)})
    with (output/'post-validation.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    result=analyze(output,[480,960,1920])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/gaussian-campaign-v01/numerical-timestep-v01')
    r=validate(**vars(p.parse_args()));print(json.dumps({'rates':len(r['rates']),'pairs':r['refinement_pairs']},indent=2))
