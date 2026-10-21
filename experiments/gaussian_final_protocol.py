"""Seal checkpoint identities before predictive TEST access; never fit here."""
import argparse
import hashlib
import json
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def seal(design, roots, models, output):
    """Only public plan metadata and frozen checkpoint bytes are inspected."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Final protocol seals are immutable')
    payload = dict(design_path=str(Path(design).resolve()), design_sha256=file_sha(design),
        design=json.loads(Path(design).read_text()), datasets=[], models=models,
        qualification='Prediction checkpoint identities frozen before predictive TEST evaluation')
    for root in map(Path, roots):
        plan = json.loads((root / 'plan.json').read_text())
        cases = [dict(episode_id=c['episode_id'], split=c['split'], group_id=c['group_id'], recipe_sha256=c['recipe_sha256'])
                 for c in plan['episodes'] if c['split'] == 'test']
        if not cases:
            raise ValueError('No TEST cases declared in public plan')
        payload['datasets'].append(dict(root=str(root.resolve()), plan_sha256=file_sha(root / 'plan.json'), cases=cases))
    if not models:
        raise ValueError('At least one frozen model pair required')
    for model in models:
        for key in ['inverse', 'forward']:
            model[key + '_sha256'] = file_sha(model[key])
    payload['protocol_sha256'] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n')
    return payload


def load_protocol(path):
    payload = json.loads(Path(path).read_text())
    expected = payload.pop('protocol_sha256')
    if digest(payload) != expected:
        raise ValueError('Final TEST protocol seal changed')
    if file_sha(payload['design_path']) != payload['design_sha256']:
        raise ValueError('Final TEST design changed after sealing')
    payload['protocol_sha256'] = expected
    return payload


def require_model(protocol, inverse, forward):
    identity = (file_sha(inverse), file_sha(forward))
    if not any((m['inverse_sha256'], m['forward_sha256']) == identity for m in protocol['models']):
        raise ValueError('Checkpoint pair is not in the sealed final TEST protocol')


def collect_test_cases(roots, protocol):
    """Validate public metadata first, then infrastructure completion audits."""
    allowed = {d['root']: d for d in protocol['datasets']}
    records = []
    for root in map(Path, roots):
        declared = allowed.get(str(root.resolve()))
        if declared is None or file_sha(root / 'plan.json') != declared['plan_sha256']:
            raise ValueError('Dataset absent from final protocol or public plan changed')
        plan = json.loads((root / 'plan.json').read_text())
        recipes = {c['episode_id']: c for c in plan['episodes']}
        for frozen in declared['cases']:
            case = recipes[frozen['episode_id']]
            if any(case[k] != v for k, v in frozen.items()) or case['split'] != 'test':
                raise ValueError('TEST recipe differs from sealed public metadata')
            episode = root / 'episodes' / case['episode_id']
            if (episode / 'failure.json').exists() or not json.loads((episode / 'validation.json').read_text()).get('passed'):
                raise ValueError('Final TEST episode failed infrastructure audit')
            if not json.loads((episode / 'summary.json').read_text()).get('completed'):
                raise ValueError('Final TEST generation incomplete')
            records.append(dict(path=episode, recipe=case, dataset=plan['dataset'], key=plan['dataset']+'/'+case['episode_id']))
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', required=True)
    p.add_argument('--roots', nargs='+', required=True)
    p.add_argument('--models', required=True, help='JSON list of named inverse/forward checkpoint paths')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = seal(a.design, a.roots, json.loads(Path(a.models).read_text()), a.output)
    print(json.dumps(dict(protocol_sha256=result['protocol_sha256'], models=len(result['models']), no_test_truth_opened=True)))
