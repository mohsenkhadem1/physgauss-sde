"""Additive phaseB provenance: frozen180 regression files plus new40 strict inputs."""
from pathlib import Path
from scripts.benchmark_improvement_v14.common import ROOT,BASE,REPORT,read,sha
BASE = BASE / "robust"


def declared(case):
    p=read(REPORT/'protocol-b.json')
    matches=[c for c in p['regression_cases']+p['fresh_softvt_cases'] if c['name']==case.get('name')]
    if len(matches)!=1 or matches[0]!=case:
        raise ValueError('Case differs from exact frozen cohort identity')
    return matches[0]


def _checked(paths):
    return {str(Path(p).absolute()):sha(p) for p in paths}


def snapshot_sources_b(case):
    """Pin every permitted input/cache; fresh fits must already be complete."""
    declared(case)
    if case['cohort']!='prospective_softvt_phase_b':
        inventory=read(REPORT/'inventory-b.json')
        if inventory['protocol_sha256']!=sha(REPORT/'protocol-b.json'):
            raise ValueError('Historical inventory protocol mismatch')
        entry=inventory['regression'][case['name']]
        if entry['case']!=case:raise ValueError('Historical inventory identity mismatch')
        expected={str((ROOT/p).absolute()):h for p,h in entry['sha256'].items()}
        actual=_checked(expected)
        if actual!=expected:raise ValueError('Historical input/baseline changed: '+case['name'])
        return actual
    folder=BASE/'inputs'/case['name'];export=read(folder/'export.json')
    required={'full.npz','full.json','short.npz','short.json','scene.npz','scene.json'}
    if export['spec']!=case['spec'] or set(export['files'])!=required:
        raise ValueError('Fresh export identity or inventory mismatch')
    if any(sha(folder/p)!=h for p,h in export['files'].items()):
        raise ValueError('Fresh export changed')
    paths=[folder/'export.json',*(folder/p for p in sorted(required))]
    for tag in ['full','short']:
        fit=BASE/'fresh-base/current'/tag/case['name'];record=read(fit/'complete.json')
        if record['name']!=case['name'] or record['tag']!=tag or record['field'] is not True:
            raise ValueError('Fresh fit identity mismatch')
        contract=record['contract'];settings=dict(field=True,contact=True,nonconvex=False,materials=1)
        if contract['input_sha256']!=sha(folder/(tag+'.npz')) or contract['metadata_sha256']!=sha(folder/(tag+'.json')):
            raise ValueError('Fresh fit input mismatch')
        if contract['settings']!=settings or contract['maximum_iterations']!=100 or contract['frames']!=36:
            raise ValueError('Fresh fit settings mismatch')
        required_core={f'scripts/{p}' for p in ['contact_benchmark_v12/base_runner.py','external_benchmark/fit.py','external_benchmark/fem.py','external_benchmark/dynamic_fem.py']}
        if set(contract['core_source_sha256'])!=required_core or any(sha(ROOT/p)!=h for p,h in contract['core_source_sha256'].items()):
            raise ValueError('Fresh fit core code changed')
        if record['prediction_sha256']!=sha(fit/'prediction.npz'):
            raise ValueError('Fresh fit prediction changed')
        if record['fit_end']!=read(folder/(tag+'.json'))['train_end']:
            raise ValueError('Fresh fit prefix mismatch')
        paths.extend([fit/'complete.json',fit/'prediction.npz',fit/'prediction.json',fit/'provenance.json'])
        if record.get('continuation'):paths.append(fit/'continuation.json')
    return _checked(paths)


def validate_record_b(case,record,out):
    """Caller separately validates freeze SHA; this binds identity and all inputs."""
    declared(case);out=Path(out)
    expected=BASE/'evaluation'/case['cohort']/case['name']
    if out.absolute()!=expected.absolute() or record.get('case')!=case:
        raise ValueError('Forecast output/case identity mismatch')
    if record.get('sources')!=snapshot_sources_b(case):
        raise ValueError('Forecast input provenance changed')
    if record.get('prediction_sha256')!=sha(out/'predictions.npz') or record.get('metadata_sha256')!=sha(out/'metadata.json'):
        raise ValueError('Forecast artifact hash mismatch')
    return True

# Same public interface as phaseA; explicit suffixed names retained for audits.
snapshot_sources = snapshot_sources_b
validate_record = validate_record_b
