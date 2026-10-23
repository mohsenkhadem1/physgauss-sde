"""Exclusive driver-parity then ONE fine256 trajectory; source-only until release.

start detaches an owned controller; inspect reads durable progress. recover writes
a hash-bound interruption manifest ONLY: no restart/resume is implemented. An
infrastructure continuation requires a later independently reviewed segment.
"""
import argparse
import ast
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from experiments.campaign_long_trajectory_runtime_v03 import (
    atomic_json, load_checkpoint, record, sha, verify_pins,
)
from experiments.campaign_long_refinement_worker_v04 import JOB, GPUS, restore_signals
from experiments.campaign_owned_gpu_budget_v02 import run_owned_command

ROOT = Path(__file__).resolve().parents[1]
OP = ROOT/'runs/operator-dev/long-smooth-refinement-v04'
OUT = ROOT/'runs/gaussian-campaign-v01/long-smooth-refinement-v04'
OLD = ROOT/'runs/gaussian-campaign-v01/long-smooth-trajectory-v03'
OLDOP = ROOT/'runs/operator-dev/long-smooth-trajectory-v03'
ENV_SITE = ROOT/'runs/operator-dev/newton-runtime-v01/env/lib/python3.12/site-packages'
SOURCE = 'experiments/run_long_smooth_refinement_v04.py'


def proc_identity(pid):
    """ESRCH-safe ownership read; absence is a value, never a stale stat read."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, process_start_ticks=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def driver():
    return subprocess.check_output(['nvidia-smi', '-i', GPUS[0],
                                    '--query-gpu=uuid,driver_version', '--format=csv,noheader'],
                                   text=True, timeout=10).strip()


def source_paths():
    pending = [SOURCE, 'experiments/campaign_long_refinement_parity_v04.py',
               'tests/test_long_midpoint_dispersion_v01.py',
               'tests/test_long_smooth_refinement_v04.py']
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('experiments.'):
                other = 'experiments/'+node.module.split('.')[1]+'.py'
                if (ROOT/other).is_file():
                    pending.append(other)
    return sorted(seen)


def prepare():
    if OP.exists() or OUT.exists():
        raise FileExistsError('Exclusive new preparation/output namespace required')
    old = json.loads((OLDOP/'protocol.json').read_text())
    cap = json.loads((ROOT/'runs/operator-dev/long-midpoint-capacity-v01/protocol.json').read_text())
    pins = dict(old['source_pins'])
    pins.update({str(ROOT/name): sha(ROOT/name) for name in source_paths()})
    for name in ('protocol.json', 'convergence-summary.json', 'terminal-proof.json'):
        pins[str(OLDOP/name)] = sha(OLDOP/name)
    for job in ('fine40_h64', 'fine40_h128', 'coarse32_h128'):
        result = OLD/job/'result.json'
        r = json.loads(result.read_text())
        if not r['completed'] or not r['passed']:
            raise ValueError('All exact existing trajectory prerequisites required')
        pins[str(result)] = sha(result)
        for item in [r['geometry'], r['accepted_ledger'], *[x['capture'] for x in r['clocks']]]:
            pins[item['path']] = item['sha256']
    controls = ROOT/'runs/operator-dev/long-smooth-dispersion-diagnosis-v01/production-controls.json'
    c = json.loads(controls.read_text())
    if c['passed'] != 2 or c['failed'] != 0 or not c['no_FE_or_GPU_calls']:
        raise ValueError('Reviewed production-kernel oscillator controls required')
    pins[str(controls)] = sha(controls)
    current = driver()
    if current != GPUS[0]+', 595.99.02':
        raise ValueError('Expected current driver595.99.02 onGPU0; review changed provenance')
    verify_pins(pins)
    p = dict(schema='one-fine256-h128-reference-refinement-v04',
             status='FROZEN_PENDING_ROOT_FULL_SOURCE_REVIEW_NO_PHYSICS', source_pins=pins,
             job=JOB, GPU_UUID=GPUS[0], CPU_cores=[[0, 1]], CPU_affinity_cores=[0, 1],
             inputs=old['inputs'], physics=old['physics'], pulse=old['pulse'],
             clocks_s=old['clocks_s'], original_gates=old['gates'],
             driver_provenance=dict(previous='595.91', current='595.99.02', GPU0_query=current,
                                    agents_changed_environment=False),
             benchmark=cap['benchmark'], taskset=old['taskset'],
             installed_warp_API_sha256=old['installed_warp_API_sha256'],
             output=str(OUT.relative_to(ROOT)), owned_metadata_directory=str(OP.relative_to(ROOT)),
             resources=dict(physical_GPU_UUID=GPUS[0], physical_GPU_index=0,
                            RSS_cap_bytes=12*1024**3, disk_free_reserve_bytes=30*1024**3,
                            projected_remaining_writes_reserve_bytes=3*1024**3,
                            GPU_allocated_cap_bytes=8*1024**3,
                            parity_worker_s=600, parity_controller_s=700,
                            fine_worker_s=24000, controller_global_s=24300, per_step_s=600),
             qualification=dict(target='EXISTING fine40_h128', levels=[64, 128, 256],
                                quantities=['q', 'v'], representations=['allnode', '63query'],
                                temporal_order_minimum=1., h128_temporal_indicator_factor=2.,
                                spatial='Original SAMEh128 coarse32/fine40 query difference/(h32/h40-1); conditionalp>=1',
                                query_combined_thresholds=dict(q=20e-6, v=100e-6),
                                premise='Observed threelevel contraction and continuingp>=1 are conditional, not a certified tail bound'),
             recovery='inspect is read-only; recover exports verified transaction hashes after all ownedPIDs absent. No automatic restart. Later reviewed segment must use committed absolute time and immutable protocol/source/geometry/clocks, retain original failure and globalwork budgets. Physical/resource failure never auto-resumes.',
             failure_policy='Parity fail blocks fine launch. Exactly one fine256; no retries/overwrite. Preserve all accepted checkpoints, clocks and failed candidates.',
             scope='Nine-clock h128 query reference with conditional spatial order only; no h256-reference/fullfield spatial/continuous-time/stress/realmaterial certification',
             admission=False, raw_fields_LOCAL=True)
    OP.mkdir(parents=True)
    atomic_json(OP/'protocol.json', p)
    return sha(OP/'protocol.json')


def validate(digest, check_driver=True):
    p = json.loads((OP/'protocol.json').read_text())
    if sha(OP/'protocol.json') != digest or p['job'] != JOB or p['GPU_UUID'] != GPUS[0]:
        raise ValueError('Exact frozen protocol/job/GPU0 required')
    verify_pins(p['source_pins'])
    verify_pins({p['taskset']['path']: p['taskset']['sha256']})
    verify_pins({str(ENV_SITE/name): value for name, value in p['installed_warp_API_sha256'].items()})
    if check_driver and driver() != p['driver_provenance']['GPU0_query']:
        raise ValueError('Driver changed since freeze; no environment repair/retry')
    return p


def aggregate(p, digest):
    """Only the existing h128 target is assessed; no mismatched-time spatial pair."""
    fine = OUT/JOB['id']
    r = json.loads((fine/'result.json').read_text())
    if not r['completed'] or not r['passed'] or r['accepted_steps'] != 256 or r['protocol_sha256'] != digest:
        raise ValueError('Complete original physical trajectory prerequisite')
    verify_pins({r[k]['path']: r[k]['sha256'] for k in ('geometry', 'accepted_ledger', 'final_checkpoint_commit')})
    rows = [json.loads(line) for line in Path(r['accepted_ledger']['path']).read_text().splitlines()]
    if len(rows) != 256 or any(not x['passed'] or x['step'] != i+1 or x['time_s'] != (i+1)/256 for i, x in enumerate(rows)):
        raise ValueError('Complete original accepted ledger required')
    _, _, m = load_checkpoint(fine/'checkpoints', digest, JOB['id'])
    if m['step'] != 256 or m['time_s'] != 1.:
        raise ValueError('Final committed state required')
    all_arrays = []
    geometry = None
    for folder in (OLD/'fine40_h64', OLD/'fine40_h128', fine, OLD/'coarse32_h128'):
        result = json.loads((folder/'result.json').read_text())
        if [c['time_s'] for c in result['clocks']] != p['clocks_s']:
            raise ValueError('Nine exact accepted clocks required')
        values = {k: [] for k in ('q', 'v', 'probe_q', 'probe_v')}
        for clock in result['clocks']:
            item = clock['capture']
            verify_pins({item['path']: item['sha256']})
            with np.load(item['path'], allow_pickle=False) as f:
                for key in values:
                    values[key].append(f[key].copy())
        all_arrays.append({k: np.array(v) for k, v in values.items()})
        if folder != OLD/'coarse32_h128':
            with np.load(result['geometry']['path'], allow_pickle=False) as f:
                current = {k: f[k].copy() for k in f.files}
            if geometry is not None and any(not np.array_equal(geometry[k], current[k]) for k in geometry):
                raise ValueError('Exact fine spatial geometry/query identity required')
            geometry = current
    a, b, c, s = all_arrays
    if any(not np.array_equal(a[k][0], b[k][0]) or not np.array_equal(b[k][0], c[k][0]) for k in a):
        raise ValueError('Bit-identical initial fine fields required')
    hcoarse = json.loads((OLD/'coarse32_h128/result.json').read_text())['hmax_m']
    hfine = json.loads((OLD/'fine40_h128/result.json').read_text())['hmax_m']
    result = dict(target='EXISTING fine40_h128', physical_completed=True, quantities={}, admission=False,
                  conditional_order_premise=p['qualification']['premise'], raw_fields_LOCAL=True)
    for key, threshold in (('q', 20e-6), ('v', 100e-6)):
        quantity = {}
        for representation, field in (('allnode', key), ('63query', 'probe_'+key)):
            d0 = float(np.linalg.norm(a[field]-b[field], axis=2).max())
            d1 = float(np.linalg.norm(b[field]-c[field], axis=2).max())
            quantity[representation] = dict(delta64_128=d0, delta128_256=d1,
                                             observed_order=float(np.log2(d0/d1)) if d0*d1 else None,
                                             temporal_passed=bool(d0 >= 2*d1 and 2*d1 <= threshold),
                                             temporal_contraction_passed=bool(d0 >= 2*d1),
                                             temporal_accuracy_passed=bool(2*d1 <= threshold),
                                             h128_temporal_indicator=2*d1)
        spatial = float(np.linalg.norm(s['probe_'+key]-b['probe_'+key], axis=2).max())/(hcoarse/hfine-1)
        combined = quantity['63query']['h128_temporal_indicator']+spatial
        quantity.update(same_h128_spatial_indicator=spatial, combined_query_indicator=combined,
                        threshold=threshold, passed=bool(all(quantity[x]['temporal_passed'] for x in ('allnode', '63query')) and combined <= threshold))
        result['quantities'][key] = quantity
    result['conditional_qualification_passed'] = all(x['passed'] for x in result['quantities'].values())
    atomic_json(OP/'convergence-summary.json', result)
    return result


def controller(digest):
    restore_signals()
    p = validate(digest)
    if OUT.exists() or (OP/'execution.json').exists():
        raise FileExistsError('No retry/overwrite')
    apps = subprocess.check_output(['nvidia-smi', '-i', GPUS[0], '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True, timeout=10)
    if GPUS[0] in apps or shutil.disk_usage(ROOT).free-3*1024**3 < 30*1024**3:
        raise RuntimeError('GPU0 idle and30+3GiB reserve required')
    begin = time.monotonic()
    state = dict(terminal=False, protocol_sha256=digest, controller_pid=os.getpid(), phases=[])
    def save():
        state['elapsed_s'] = time.monotonic()-begin
        atomic_json(OP/'execution.json', state)
    save()
    try:
        for phase, timeout in (('parity', 700), ('fine', 24000)):
            validate(digest)
            command = [p['taskset']['path'], '-c', '0,1', str(ROOT/'.venv/bin/python'), '-u', '-m',
                       'experiments.run_long_smooth_refinement_v04', 'worker', '--phase', phase, '--sha', digest]
            def spawned(pid):
                identity = proc_identity(pid)
                if identity is None:
                    raise RuntimeError('Spawned worker identity unavailable; drain owned child, no next phase')
                atomic_json(OP/(phase+'-launch.json'), dict(**identity, command=command, protocol_sha256=digest))
                state['active_phase'] = phase
                state['worker_pid'] = pid
                save()
            remaining = 24300-(time.monotonic()-begin)
            if remaining <= 0:
                raise TimeoutError('Global24300s limit')
            with (OP/(phase+'.log')).open('x') as log:
                execution = run_owned_command(command, environment={**os.environ, 'CUDA_VISIBLE_DEVICES': GPUS[0],
                    'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                    log=log, timeout=min(timeout, remaining), on_child=spawned)
            state['phases'].append(dict(phase=phase, **execution))
            state.pop('worker_pid', None)
            save()
            folder = OUT/'parity' if phase == 'parity' else OUT/JOB['id']
            outcome = json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
            if execution['exit_code'] != 0 or not outcome.get('completed') or not outcome.get('passed'):
                raise RuntimeError(phase+' failed; no next phase or retry')
        aggregate(p, digest)
        state['completed'] = True
    except BaseException as error:
        state['failure'] = dict(type=type(error).__name__, message=str(error))
    finally:
        state['terminal'] = True
        save()
    return state.get('completed', False)


def start(digest):
    validate(digest)
    if OUT.exists():
        raise FileExistsError('No restart fromzero')
    # Exclusive durable lock precedes spawn; a spawn interruption needs review.
    with (OP/'STARTED').open('x') as f:
        f.write(digest+'\n')
        f.flush()
        os.fsync(f.fileno())
    with (OP/'controller.log').open('x') as log:
        child = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-u', '-m',
            'experiments.run_long_smooth_refinement_v04', 'controller', '--sha', digest],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS': '2',
                'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    identity = proc_identity(child.pid)
    if identity is None:
        raise RuntimeError('Detached controller already absent; preserve lock/log, inspect before any reviewed recovery')
    atomic_json(OP/'controller-start.json', dict(**identity, controller_pid=child.pid,
                protocol_sha256=digest, UI_independent=True))
    return child.pid


def inspect(digest):
    validate(digest, check_driver=False)
    report = dict(protocol_sha256=digest, owned_live=[], owned_groups_present=[], launched=[], recovery_action='NONE_AUTOMATIC')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for name in ('controller-start.json', 'parity-launch.json', 'fine-launch.json'):
        path = OP/name
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        pid = r.get('pid', r.get('controller_pid'))
        current_identity = proc_identity(pid)
        live = current_identity is not None and current_identity['boot_id'] == r['boot_id'] and current_identity['process_start_ticks'] == r['process_start_ticks']
        group_present = False
        if r['boot_id'] == boot:
            try:
                os.killpg(pid, 0)
                group_present = True
            except ProcessLookupError:
                pass
        if group_present:
            report['owned_groups_present'].append(pid)
        report['launched'].append(dict(receipt=record(path), pid=pid, same_identity_live=live))
        if live:
            report['owned_live'].append(pid)
    for name in ('execution.json', 'convergence-summary.json'):
        if (OP/name).exists():
            report[name] = json.loads((OP/name).read_text())
    folder = OUT/JOB['id']
    if (folder/'checkpoints/COMMIT.json').exists():
        _, _, metadata = load_checkpoint(folder/'checkpoints', digest, JOB['id'])
        report['last_accepted'] = metadata
        report['checkpoint_commit'] = record(folder/'checkpoints/COMMIT.json')
    if (folder/'result.json').exists():
        r = json.loads((folder/'result.json').read_text())
        report['fine_result'] = r
        for c in r.get('clocks', []):
            verify_pins({c['capture']['path']: c['capture']['sha256']})
        if 'geometry' in r:
            verify_pins({r['geometry']['path']: r['geometry']['sha256']})
    return report


def recover(digest):
    r = inspect(digest)
    if r['owned_live'] or r['owned_groups_present']:
        raise RuntimeError('Owned original process identities AND groups must be absent before recovery manifest')
    r['status'] = 'REVIEW_ONLY_NO_RESUME_RELEASED'
    r['warning'] = 'Resource/physical failures remain terminal. Infrastructure interruption needs new reviewed segment, exact committed absolute time, provenance and original total budgets; never silently restart.'
    target = OP/('recovery-manifest-'+str(time.time_ns())+'.json')
    atomic_json(target, r)
    return record(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'start', 'controller', 'worker', 'inspect', 'recover'])
    parser.add_argument('--sha')
    parser.add_argument('--phase', choices=['parity', 'fine'])
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare())
    elif args.action == 'start':
        print(start(args.sha))
    elif args.action == 'controller':
        raise SystemExit(0 if controller(args.sha) else 1)
    elif args.action == 'worker':
        restore_signals()
        protocol = validate(args.sha)
        if args.phase == 'parity':
            from experiments.campaign_long_refinement_parity_v04 import worker
            worker(protocol, OP/'protocol.json')
        elif args.phase == 'fine':
            parity = json.loads((OUT/'parity/result.json').read_text())
            if not parity.get('completed') or not parity.get('passed') or parity['protocol']['sha256'] != args.sha:
                raise ValueError('Matching completed new driver parity required before any fine FE calls')
            from experiments.campaign_long_refinement_worker_v04 import worker
            worker(protocol, JOB, args.sha)
        else:
            raise ValueError('Explicit worker phase required')
    else:
        print(json.dumps(inspect(args.sha) if args.action == 'inspect' else recover(args.sha), indent=2))
