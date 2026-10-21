"""Authorize only the two prospectively declared conditional 960 Hz cells.

This guard changes no physical model, tolerance or initial evidence. Passing a
request is not reference qualification and does not authorize 1920 Hz runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PROTOCOL_PATH = Path('configs/gaussian_campaign_dynamic_temporal_extension_v01.json')
PROTOCOL_SHA256 = 'e6c9efb31c34ea1b6ffe476da192edecad3da5697c0cebaf185ddc5d8813df6f'
TRIGGER_SHA256 = '3733850713231a5d750a74e8d454475fe30f579f6b4e638699a53e6268872f43'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_temporal_extension(protocol_path, *, episode, hz, response,
                                relative_tolerance, absolute_tolerance_N,
                                maximum_iterations, translation, inputs_json,
                                core_path, gate_path):
    """Validate before output creation or a physical solve; preserve failure.

    Returns metadata and the pinned trigger path for the runner's provenance.
    Source publication and baseline-terminal process checks are orchestration
    requirements, not inferred from the presence of this local guard.
    """
    protocol_path = Path(protocol_path)
    if digest(protocol_path) != PROTOCOL_SHA256:
        raise ValueError('Conditional temporal protocol hash differs from declaration')
    protocol = json.loads(protocol_path.read_text())
    if digest(gate_path) != protocol['parent_gate_sha256']:
        raise ValueError('Original independent dynamic gate hash differs')
    gate = json.loads(Path(gate_path).read_text())
    if digest(core_path) != protocol['core_sha256']:
        raise ValueError('Conditional temporal study requires the frozen physical core')
    if inputs_json is None or digest(inputs_json) != protocol['static_fixture_sha256']:
        raise ValueError('Conditional temporal study requires the exact public TRAIN fixture')
    trigger_path = Path(protocol['trigger_audit_path'])
    if protocol['trigger_audit_sha256'] != TRIGGER_SHA256 or digest(trigger_path) != TRIGGER_SHA256:
        raise ValueError('Initial failed temporal trigger evidence differs')
    trigger = json.loads(trigger_path.read_text())
    retained = [comparison for comparison in trigger['comparisons']
        if comparison['case'] == protocol['episode_id'] and comparison['kind'] == 'temporal']
    if retained != protocol['trigger_comparisons'] or len(retained) != 2 or any(c['passed'] for c in retained):
        raise ValueError('Both initial failed temporal comparisons must be retained')
    if (episode != protocol['episode_id'] or type(hz) is not int
            or hz != protocol['new_physics_hz']
            or hz not in gate['optional_further_refinement_hz_if_gate_fails']
            or response not in protocol['responses']
            or relative_tolerance != protocol['relative_residual_tolerance']
            or absolute_tolerance_N != protocol['absolute_residual_floor_N']
            or type(maximum_iterations) is not int
            or maximum_iterations != protocol['nonlinear_iteration_cap']):
        raise ValueError('Only the two predeclared moving-case 960 Hz requests are allowed')
    shift = np.asarray(translation, dtype=float)
    if shift.shape != (3,) or not np.array_equal(shift, protocol['translation_m']):
        raise ValueError('Conditional temporal study requires original world coordinates')
    if gate['seconds'] != protocol['seconds'] or gate['output_hz'] != protocol['output_hz']:
        raise ValueError('Conditional temporal clocks differ from the parent gate')
    return {
        'protocol_path': str(protocol_path),
        'protocol_sha256': PROTOCOL_SHA256,
        'parent_gate_sha256': protocol['parent_gate_sha256'],
        'trigger_audit_path': str(trigger_path),
        'trigger_audit_sha256': TRIGGER_SHA256,
        'episode_id': episode,
        'physics_hz': hz,
        'response': response,
        'initial_temporal_failure_preserved': True,
        'physical_model_unchanged': True,
        'purpose': 'Temporal candidate only; matched 960 Hz translation, nonlinear accuracy and fresh replay remain required before adoption.',
        'reference_qualification_implied': False,
    }
