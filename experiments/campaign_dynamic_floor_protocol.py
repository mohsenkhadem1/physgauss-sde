"""Validate the separately predeclared moving-support accuracy experiment.

This module changes no dynamics or acceptance threshold. It authorizes only
the two new TRAIN cells declared in the supplemental protocol; the original
1e-10 N baseline is reused, rather than regenerated as new evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


PROTOCOL_PATH = Path('configs/gaussian_campaign_dynamic_floor_refinement_v01.json')
PROTOCOL_SHA256 = 'ab4d663af320b1e1863d07239b473982b340992b2b6e6ff7554c9a9350fc358e'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_supplemental_floor(protocol_path, *, episode, hz, response,
                                relative_tolerance, absolute_tolerance_N,
                                maximum_iterations, translation,
                                inputs_json, core_path, gate_path):
    """Return frozen metadata after validating the full supplemental request.

    Call before creating an output directory or starting a physical solve.
    No fallback to a different floor, input, prior or physical model is allowed.
    """
    protocol_path = Path(protocol_path)
    if digest(protocol_path) != PROTOCOL_SHA256:
        raise ValueError('Supplemental floor protocol hash differs from declaration')
    protocol = json.loads(protocol_path.read_text())
    if digest(gate_path) != protocol['parent_gate_sha256']:
        raise ValueError('Original independent dynamic gate hash differs')
    if digest(core_path) != protocol['physical_model_source_sha256']:
        raise ValueError('Supplemental accuracy study requires the frozen physical core')
    if inputs_json is None or digest(inputs_json) != protocol['physical_input_fixture_sha256']:
        raise ValueError('Supplemental accuracy study requires the exact public TRAIN fixture')
    if (episode != protocol['episode_id'] or hz != protocol['physics_hz']
            or response != protocol['response']
            or relative_tolerance != protocol['relative_residual_tolerance']
            or maximum_iterations != protocol['nonlinear_iteration_cap']):
        raise ValueError('Only the predeclared moving-zero accuracy request is allowed')
    shift = np.asarray(translation, dtype=float)
    if shift.shape != (3,) or not np.array_equal(shift, protocol['translation_m']):
        raise ValueError('Supplemental floor study requires original world coordinates')
    # The middle level already exists. Never generate another baseline and
    # mistake it for an independent accuracy level or a replacement success.
    floors = protocol['absolute_residual_floors_N']
    if (not math.isfinite(absolute_tolerance_N)
            or absolute_tolerance_N not in [floors[0], floors[-1]]):
        raise ValueError('Only the new coarse and tight floors are allowed; reuse baseline')
    return {
        'protocol_path': str(protocol_path),
        'protocol_sha256': PROTOCOL_SHA256,
        'parent_gate_sha256': protocol['parent_gate_sha256'],
        'absolute_residual_floor_N': float(absolute_tolerance_N),
        'baseline_absolute_residual_floor_N_reused': float(floors[1]),
        'purpose': 'Additional effective nonlinear accuracy evidence, not a new physical model.',
        'physical_model_unchanged': True,
    }
