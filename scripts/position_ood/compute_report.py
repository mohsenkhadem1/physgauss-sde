"""Report frozen-run completion, optimizer termination and observed wall time.

Metadata only: never restarts or changes an inference run. Concurrent shared-machine
wall times are operational costs, not isolated hardware speed benchmarks.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE, OUT, ROOT
from scripts.position_ood.seal import verify, digest
from scripts.position_ood.campaign_jobs import jobs
from scripts.phystwin.sequential_reference import write


def summarize(values):
    if not values:
        return None
    return dict(count=len(values), mean=float(np.mean(values)),
                median=float(np.median(values)), minimum=float(min(values)),
                maximum=float(max(values)), total=float(sum(values)))


def acquisition_compute():
    """Account for declared probe fits once, even when policies share an action."""
    rows = []
    missing_choices = []
    sources = {}
    for split, count in [('transfer_calibration', 12), ('transfer_test', 9)]:
        for i in range(count):
            choice_path = BASE / f'active-jobs/{split}-{i}-choices.json'
            if not choice_path.exists():
                missing_choices.append(dict(split=split, object=i))
                continue
            sources[str(choice_path.relative_to(ROOT))] = digest(choice_path)
            choices = json.loads(choice_path.read_text())['actions']
            tasks = [(method, mesh, action,
                      sorted(policy for policy, selected in choices.items() if selected == action))
                     for action in sorted(set(choices.values()))
                     for method, mesh in [('joint', 2), ('extrapolated', 3)]]
            if split == 'transfer_test':
                tasks += [(method, mesh, choices['adaptive'], ['matched_frame_ablation'])
                          for method, mesh in [('material_frame', 2),
                                               ('extrapolated_material_frame', 3)]]
            for method, mesh, action, policies in tasks:
                path = BASE / f'inverse/{split}-{i}/{method}-n{mesh}-p4-0-1-2-{action}.json'
                row = dict(split=split, object=i, method=method, mesh=mesh,
                           action=action, policies=policies, complete=path.exists())
                if path.exists():
                    result = json.loads(path.read_text())
                    fits = result['rows']
                    row.update(device=result['device'], layouts=len(fits),
                               fit_wall_seconds=sum(f['elapsed_seconds'] for f in fits),
                               evaluations=sum(f['evaluations'] for f in fits),
                               optimizer_successes=sum(bool(f['optimizer_success']) for f in fits),
                               termination_messages=dict(Counter(f['optimizer_message'] for f in fits)))
                    sources[str(path.relative_to(ROOT))] = digest(path)
                rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row['method'], row.get('device'))].append(row)
    summaries = []
    for (split, method, device), records in sorted(groups.items(), key=lambda x: str(x[0])):
        ready = [r for r in records if r['complete']]
        summaries.append(dict(split=split, method=method, device=device,
                              fits=len(records), completed=len(ready),
                              fit_wall_seconds=summarize([r['fit_wall_seconds'] for r in ready])))
    return dict(complete=not missing_choices and all(r['complete'] for r in rows),
                missing_choices=missing_choices, rows=rows, groups=summaries,
                source_hashes=sources,
                interpretation='Each distinct acquired action is counted once per object and model resolution. Policies sharing an action reuse the same fit; the policies field records that reuse. Fit wall time sums recorded optimizer times across screened layouts, excluding lock waits, prediction, uncertainty calculation, prefix preparation and simulator reference generation. Coarse and fine stages are separate. Concurrent CPU and GPU timings are grouped by device and are not isolated hardware benchmarks.')


def main():
    seal = verify()
    rows = []
    groups = defaultdict(list)
    for job in jobs(seal['settings']):
        output = Path(job['result'])
        status = BASE / 'jobs' / (job['name'] + '.json')
        record = json.loads(status.read_text()) if status.exists() else {}
        row = dict(name=job['name'], script=job['script'],
                   state='complete' if output.exists() else record.get('state', 'pending'))
        if 'elapsed_seconds' in record:
            row['elapsed_seconds'] = record['elapsed_seconds']
        row['execution_kind'] = record.get('kind')
        if output.exists() and job['script'] in ('run_inverse.py', 'fit_mpm.py', 'phystwin.py'):
            result = json.loads(output.read_text())
            best = result.get('best', {})
            row['optimizer_success'] = best.get('optimizer_success')
            row['optimizer_message'] = best.get('optimizer_message', best.get('message'))
            row['iterations'] = best.get('iterations')
            row['evaluations'] = best.get('evaluations')
            row['fixed_epochs'] = result.get('epochs')
            row['latent_coordinates'] = best.get('z')
        rows.append(row)
        method = next((job['args'][k + 1] for k, arg in enumerate(job['args'])
                       if arg == '--method'), job['script'].removesuffix('.py'))
        groups[(job['script'], method, row['execution_kind'])].append(row)
    summaries = []
    for (script, method, device), records in sorted(groups.items(), key=lambda x: str(x[0])):
        completed = [r for r in records if r['state'] == 'complete']
        flags = [r['optimizer_success'] for r in completed if r.get('optimizer_success') is not None]
        summaries.append(dict(script=script, method=method, execution_kind=device, declared=len(records),
            completed=len(completed), states=dict(Counter(r['state'] for r in records)),
            wall_seconds=summarize([r['elapsed_seconds'] for r in completed if 'elapsed_seconds' in r]),
            optimizer_flags=dict(success=sum(flags), other_termination=len(flags)-sum(flags)),
            termination_messages=dict(Counter(r['optimizer_message'] for r in completed if r.get('optimizer_message')))))
    training = []
    for path in sorted((BASE/'neural').glob('*/summary.json')):
        record = json.loads(path.read_text())
        training.append(dict(model=path.parent.name, objects=record['count'],
                             updates=record['epochs'], training_actions=record['training_actions'],
                             loop_wall_seconds=record['elapsed_seconds']))
    report = dict(declared_jobs=len(rows), states=dict(Counter(r['state'] for r in rows)),
                  groups=summaries, rows=rows,
                  neural_training=training,
                  acquisition=acquisition_compute(),
                  neural_training_scope='Training-loop wall times exclude bank generation, data loading and spatial-basis construction. The shared 512-object bank contains twelve paired-resolution positional queries per object. Both candidate architectures and both action-training variants are reported; selected models are identified in neural-selection.json.',
                  interpretation='Shared-workstation wall time includes concurrent contention and process overhead. Fine inverse jobs exclude their separately reported coarse initialization. Optimizer termination flags describe the fixed computational budget; completion and prediction accuracy are separate quantities. No cases are filtered by these flags.')
    write(OUT / 'compute-report.json', report)
    print(json.dumps(dict(declared_jobs=len(rows), states=report['states'])))


if __name__ == '__main__':
    main()
