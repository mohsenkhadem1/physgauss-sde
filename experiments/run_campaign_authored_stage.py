"""Bounded, no-retry CPU orchestration of a sealed authored-force stage.

Preparation is pure manifest construction. Launch requires a fetched published
commit containing the exact local seal, manifest, resources and sources. A
failed worker stops new launches; already running independent cells finish.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
THREAD_ENV = {name: '2' for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Atomic status replacement; a reader never sees half a JSON record."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def relative_path(path):
    raw = str(path); path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or '..' in path.parts or str(path) == '.' or raw != str(path) or '\\' in raw:
        raise ValueError('Canonical nonempty repository-relative path required')
    return str(path)


def build_initial_manifest(protocol, *, protocol_sha256, source_sha256,
                           authored_auditor_source_sha256, output_root):
    """Resolve all 51 exact initial rows; no physics or response files opened."""
    output_root = relative_path(output_root)
    rows = []
    for template in protocol['initial_cell_plan']:
        if template['stage'] != 'initial_temporal':
            continue
        row = dict(template)
        row['template_id'] = template['id']
        row['output'] = output_root+'/'+relative_path(template['id'])
        rows.append(row)
    if len(rows) != 51 or len({r['id'] for r in rows}) != 51:
        raise ValueError('Complete unique 51-cell initial temporal plan required')
    # Deterministic interleaving covers all BCs/families early; no conditions change.
    scenarios = {s['id']: s for s in protocol['scenarios']}
    family_order = {name: i for i, name in enumerate(['push_pulse', 'pull_hold', 'shear_oscillation', 'twist_couple'])}
    def priority(row):
        if row['response'] == 'zero_external_force':
            return (-1, row['hz'], row['case'])
        rates = scenarios[row['scenario_id']]['initial_rates_hz']
        return (rates.index(row['hz']), family_order[row['response']], row['case'])
    rows.sort(key=priority)
    first_id = 't0001__push_pulse__temporal__hz120'
    first = next(row for row in rows if row['id']==first_id)
    rows.remove(first); rows.insert(0, first)
    return {'schema': 'actual-authored-force-resolved-stage-v0.1',
        'status': 'FROZEN_PUBLISHED_RESOLVED_EXECUTION_MANIFEST',
        'stage': 'initial_temporal', 'protocol_sha256': protocol_sha256,
        'source_sha256': dict(source_sha256),
        'authored_auditor_source_sha256': authored_auditor_source_sha256,
        'ordering': 'First fixed-patch authored push120, then shared moving zeros and rate-index/family/layout interleaving; no changed row settings.',
        'cells': rows}


def require_published_files(commit, paths):
    """Check exact bytes in a locally fetched commit, not a publication flag."""
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Exact fetched publication commit required')
    subprocess.run(['git', 'cat-file', '-e', commit+'^{commit}'], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for relative in sorted(set(paths)):
        relative = relative_path(relative)
        published = subprocess.check_output(['git', 'show', commit+':'+relative], cwd=ROOT)
        if hashlib.sha256(published).hexdigest() != digest(ROOT/relative):
            raise ValueError('Published/local bytes differ: '+relative)


def worker_command(cell, seal_path):
    return [str(ROOT/'.venv/bin/python'), '-u', str(ROOT/'experiments/campaign_authored_force_case.py'),
            '--prelaunch-seal', relative_path(seal_path), '--cell-id', cell['id'], '--output', cell['output']]


def independent_cell_audit(seal_path, cell_id):
    try:
        import campaign_authored_force_audit as auditor
    except ImportError:
        from experiments import campaign_authored_force_audit as auditor
    return auditor.audit_cell(seal_path, cell_id)


def launch(seal_path, published_commit, *, poll_seconds=.5):
    try:
        import campaign_authored_force_case as runner
    except ImportError:
        from experiments import campaign_authored_force_case as runner
    seal_path = relative_path(seal_path)
    seal = read(ROOT/seal_path)
    manifest_path = relative_path(seal['resolved_manifest']['path'])
    manifest = read(ROOT/manifest_path)
    cells = manifest['cells']
    if not cells:
        raise ValueError('Nonempty reviewed stage required')
    _, verified_manifest, _, pins = runner.validate_seal(seal_path, cells[0]['id'])
    if verified_manifest != manifest:
        raise ValueError('Manifest changed during preflight')
    resource_path = relative_path(seal['resource_declaration']['path'])
    resource = read(ROOT/resource_path)
    maximum = resource['max_parallel_workers']
    if type(maximum) is not int or not 1 <= maximum <= 8:
        raise ValueError('Explicit integer CPU worker cap in [1,8] required')
    first_gate = resource.get('first_cell_independent_audit_before_parallel_launch', False)
    if type(first_gate) is not bool:
        raise ValueError('Explicit boolean first-cell audit policy required')
    if first_gate and resource.get('first_cell_id') != cells[0]['id']:
        raise ValueError('Declared first physical cell must match frozen manifest order')
    driver_record = seal['stage_driver_source']
    if relative_path(driver_record['path']) != SOURCE.relative_to(ROOT).as_posix() or digest(SOURCE) != driver_record['sha256']:
        raise ValueError('Exact reviewed stage driver pin required')
    published_paths = [relative_path(path) for path in pins]
    published_paths += [seal_path, manifest_path, resource_path, SOURCE.relative_to(ROOT).as_posix()]
    require_published_files(published_commit, published_paths)
    stage = ROOT/manifest_path
    execution = stage.parent/'execution.json'
    first_audit_path = stage.parent/'first-cell-audit-v01.json'
    if execution.exists():
        raise FileExistsError('Stage already has an execution ledger; inspect it, never restart automatically')
    if first_gate and first_audit_path.exists():
        raise FileExistsError('First-cell audit already exists; preserve it and inspect the stage')
    for cell in cells:
        relative_path(cell['output'])
        if (ROOT/cell['output']).exists():
            raise FileExistsError('Stage output already exists: '+cell['output'])
    logs = stage.parent/'logs'; logs.mkdir(exist_ok=False)
    environment = {**os.environ, **THREAD_ENV}
    ledger = {'driver_pid': os.getpid(), 'publication_commit': published_commit,
        'started_unix_s': time.time(), 'terminal': False, 'completed': False,
        'max_parallel_workers': maximum, 'threads_per_worker': 2,
        'thread_environment': THREAD_ENV, 'prelaunch_seal_sha256': digest(ROOT/seal_path),
        'resolved_manifest_sha256': digest(ROOT/manifest_path), 'jobs': [],
        'no_automatic_retries': True, 'stop_new_launches_after_first_failure': True,
        'first_cell_audit_required': first_gate, 'first_cell_audit_passed': False}
    for cell in cells:
        ledger['jobs'].append({'id': cell['id'], 'output': cell['output'],
            'command': worker_command(cell, seal_path), 'state': 'queued'})
    write(execution, ledger)
    running = {}; next_index = 0; failed = False; last_report = 0.
    def unchanged():
        if digest(SOURCE) != driver_record['sha256']:
            raise ValueError('Stage driver changed during execution')
        for name, expected in pins.items():
            if digest(ROOT/relative_path(name)) != expected:
                raise ValueError('Sealed stage dependency changed: '+name)
    try:
        while running or (next_index < len(cells) and not failed):
            unchanged()
            capacity = 1 if first_gate and not ledger['first_cell_audit_passed'] else maximum
            while not failed and len(running) < capacity and next_index < len(cells):
                job = ledger['jobs'][next_index]
                stream = (logs/(job['id']+'.log')).open('x')
                try:
                    process = subprocess.Popen(job['command'], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                finally:
                    stream.close()
                job.update(state='running', pid=process.pid, started_unix_s=time.time())
                running[next_index] = process; next_index += 1
                write(execution, ledger)
            for index, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                job = ledger['jobs'][index]; output = ROOT/job['output']
                state = {}; summary = {}; validation_error = None
                try:
                    state = read(output/'execution.json') if (output/'execution.json').exists() else {}
                    summary = read(output/'summary.json') if (output/'summary.json').exists() else {}
                    if not isinstance(state, dict) or not isinstance(summary, dict):
                        raise ValueError('Worker completion records must be JSON objects')
                    passed = (code == 0 and state.get('terminal') is True and state.get('completed') is True
                        and state.get('pid') == process.pid and summary.get('completed') is True
                        and summary.get('cell_id') == job['id'] and summary.get('frames') == 61
                        and type(state.get('planned_steps')) is int and state['planned_steps'] == 3*cells[index]['hz']
                        and state.get('completed_steps') == state['planned_steps'])
                    if passed:
                        for filename, key in [('trajectory.npz','trajectory_sha256'), ('steps.jsonl','steps_sha256')]:
                            if summary.get(key) != digest(output/filename):
                                passed = False
                    if passed and first_gate and index == 0:
                        audit = independent_cell_audit(seal_path, job['id'])
                        if not isinstance(audit, dict):
                            raise ValueError('Independent cell audit must return a JSON report')
                        write(first_audit_path, audit)
                        passed = audit.get('passed') is True
                        ledger['first_cell_audit_passed'] = passed
                        ledger['first_cell_audit_sha256'] = digest(first_audit_path)
                except (OSError, ValueError, TypeError) as error:
                    passed = False; validation_error = type(error).__name__+': '+str(error)
                    if not isinstance(state, dict): state = {}
                    if not isinstance(summary, dict): summary = {}
                job.update(state='completed' if passed else 'failed', exit_code=code,
                           finished_unix_s=time.time(), worker_terminal=state.get('terminal'),
                           worker_completed=state.get('completed'), completion_validation_error=validation_error)
                failed |= not passed
                del running[index]; write(execution, ledger)
            if time.monotonic()-last_report >= 30:
                print(json.dumps({'driver_pid': os.getpid(), 'counts': dict(Counter(j['state'] for j in ledger['jobs']))}), flush=True)
                last_report = time.monotonic()
            if running:
                time.sleep(poll_seconds)
        unchanged()
        ledger.update(terminal=True, completed=all(j['state']=='completed' for j in ledger['jobs']), finished_unix_s=time.time())
        write(execution, ledger)
        return ledger
    except BaseException as error:
        # Preserve in-flight processes and their exact PIDs for diagnosis; no restart.
        ledger.update(terminal=True, completed=False, driver_error=type(error).__name__+': '+str(error),
                      running_workers_may_remain=[p.pid for p in running.values()], finished_unix_s=time.time())
        write(execution, ledger)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prelaunch-seal', required=True)
    parser.add_argument('--published-commit', required=True)
    args = parser.parse_args()
    result = launch(args.prelaunch_seal, args.published_commit)
    print(json.dumps({'terminal': result['terminal'], 'completed': result['completed']}), flush=True)
    raise SystemExit(0 if result['completed'] else 1)
