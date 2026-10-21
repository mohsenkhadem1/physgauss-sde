"""Prepare two post-hoc elastic-excitation controls on CALIBRATION object15.

No inference/test evaluation, new truth draw, finite execution or old source
change. The independent reserved-view experiment remains unchanged. These
two amplitudes are development diagnostics, not a population comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
try:
    import campaign_hypothesis_dynamics as worker
    from campaign_balanced_probe import balanced_probe_clock, wrench_diagnostics
except ImportError:
    from experiments import campaign_hypothesis_dynamics as worker
    from experiments.campaign_balanced_probe import balanced_probe_clock, wrench_diagnostics


def prepare(calibration, digest, output):
    marker = json.loads(worker.checked(dict(path=calibration, sha256=digest)).read_text())
    if not marker.get('completed') or marker.get('calibration_object_indices') != list(range(16)):
        raise ValueError('Completed independent calibration freeze required')
    execution = json.loads(worker.checked(marker['execution_protocol']).read_text())
    plan = json.loads(worker.checked(execution['cohort_plan']).read_text())
    objects = [o for o in plan['objects'] if o['index'] == 15]
    if len(objects) != 1 or objects[0]['role'] != 'CALIBRATION':
        raise ValueError('Only previously diagnosed CALIBRATION object15 permitted')
    obj = objects[0]
    # Only hash the already-known calibration hypothesis, never reserved values.
    worker.checked(obj['realized_hypothesis'])
    public = json.loads(worker.checked(plan['public_hardware']).read_text())
    origins = np.asarray([h['origin_world_m'] for h in public['handles']])
    out = worker.resolve(output); out.mkdir(parents=True, exist_ok=False)
    record = lambda p: dict(path=p.relative_to(worker.ROOT).as_posix(), sha256=worker.sha(p))
    pins = worker.runtime_sources()
    pins.update({f'experiments/{n}': worker.sha(worker.ROOT/'experiments'/n)
        for n in ['campaign_balanced_probe.py', 'prepare_balanced_probe.py']})
    clocks = []
    for label, amplitude in [('020N', .02), ('200N', .2)]:
        clock = balanced_probe_clock(origins, amplitude_N=amplitude)
        p = out/(label+'-commands.npz')
        np.savez_compressed(p, **{n: getattr(clock, n) for n in ['endpoint_times_s',
            'output_times_s', 'handle_forces_world_N', 'handle_torques_world_Nm',
            'nodal_forces_world_N', 'body_acceleration_world_m_s2']})
        clocks.append(dict(label=label, amplitude_N=amplitude, commands=record(p),
            public_command_manifest=clock.public_manifest(), reference_wrench=wrench_diagnostics(clock, origins)))
    design = dict(schema='balanced-reference-probe-development-v0.1',
        calibration_freeze=dict(path=calibration, sha256=digest),
        role='DEVELOPMENT_EXCITATION_DIAGNOSTIC', object_index=15,
        selection='Post-hoc diagnosis of near-rigid calibration motion; both amplitudes fixed before execution',
        public_hardware=plan['public_hardware'], realized_hypothesis=obj['realized_hypothesis'],
        source_sha256=pins, clocks=clocks, finite_runs_per_amplitude=1,
        inference_or_reserved_comparison=False, retries_or_response_selection=False,
        limitations=['Open-loop force is central at rest only; measure actual moment.',
            'Same previously seen calibration object, no generalization or uncertainty claim.',
            'Independent engineering finite mechanics, not native Isaac or continuum truth.'])
    design_path = out/'design.json'; worker.write(design_path, design)
    cells = []
    for row in clocks:
        protocol = dict(protocol='independent-hypothesis-fixed-clock-smoke-v0.1',
            status='FROZEN_REVIEWED_SINGLE_TRAIN_SMOKE', split='TRAIN',
            cell_id='balanced_'+row['label'], evaluation_role=design['role'], input_design=record(design_path),
            settings=worker.SETTINGS, information_roles=worker.INFORMATION_ROLES, integration_hz=120,
            output='runs/gaussian-campaign-v01/balanced-probe-v01/'+row['label'],
            public_hardware=plan['public_hardware'], realized_hypothesis=obj['realized_hypothesis'],
            known_commands=row['commands'], program_id='balanced_reference_tension_'+row['label']+'_v01',
            public_command_manifest=row['public_command_manifest'], source_sha256=worker.runtime_sources())
        p = out/(row['label']+'-protocol.json'); worker.write(p, protocol)
        worker.validate_protocol(record(p)['path'], worker.sha(p))
        cells.append(dict(protocol=record(p), output=protocol['output']))
    result = dict(design=record(design_path), cells=cells, source_sha256=pins, physical_execution=False)
    worker.write(out/'prepared.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', required=True); p.add_argument('--calibration-sha256', required=True)
    p.add_argument('--output', required=True); a = p.parse_args()
    print(json.dumps(prepare(a.calibration, a.calibration_sha256, a.output)))
