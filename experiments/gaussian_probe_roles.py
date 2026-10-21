"""Frozen control-response roles and explicit TRAIN-only known-load labels.

These helpers are for training/calibration data collection, never passive
inference. Role eligibility must be checked BEFORE opening trajectory labels.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from gaussian_operator import physical_vector,category_labels
except ImportError:
    from experiments.gaussian_operator import physical_vector,category_labels


def validate_role_manifest(manifest):
    teacher=manifest['teacher_fit_ids'];calibration=manifest['calibration_only_ids']
    if not teacher or not calibration or set(teacher)&set(calibration):
        raise ValueError('Disjoint nonempty teacher/calibration roles required')
    all_ids=teacher+calibration
    if len(all_ids)!=len(set(all_ids)):raise ValueError('Duplicate role IDs')
    if any(manifest['response_splits'].get(eid)!='train' for eid in all_ids):
        raise ValueError('Only TRAIN split calibrated responses are eligible')
    actions=set(manifest['eligible_control_ids'])
    if actions!={'poke_0','poke_1','poke_2','poke_3'}:
        raise ValueError('Only four predeclared candidate controls are eligible')
    if actions&set(manifest['excluded_teacher_control_ids']):raise ValueError('Contradictory action roles')
    if not {'evaluation','reset_replay'}<=set(manifest['excluded_teacher_control_ids']):
        raise ValueError('Evaluation and replay must be excluded')
    for role in [teacher,calibration]:
        if {manifest['boundary_types'][eid] for eid in role}!={'point_pin','fixed_patch','two_patches','moving_patch'}:
            raise ValueError('Each role must cover all four boundary types')
    return manifest


def load_role_manifest(path, expected_sha256=None):
    content=Path(path).read_bytes()
    if expected_sha256 and hashlib.sha256(content).hexdigest()!=expected_sha256:
        raise ValueError('Frozen response-role manifest fingerprint differs')
    return validate_role_manifest(json.loads(content))


def control_role(manifest, episode_id, action_id, source_split):
    validate_role_manifest(manifest)
    if source_split!='train':raise ValueError('VAL/TEST response prohibited by frozen development roles')
    if episode_id not in set(manifest['teacher_fit_ids']+manifest['calibration_only_ids']):
        raise ValueError('Episode absent from frozen role manifest')
    if action_id=='reset_replay':return 'audit_only'
    if action_id=='evaluation':return 'evaluation_only'
    if action_id not in manifest['eligible_control_ids']:raise ValueError('Unknown control')
    return 'teacher_fit' if episode_id in manifest['teacher_fit_ids'] else 'calibration_only'


def require_teacher_control(manifest, episode_id, action_id, source_split):
    if control_role(manifest,episode_id,action_id,source_split)!='teacher_fit':
        raise ValueError('Response labels are not eligible for teacher fitting')


def known_control_label(case,spec,points,action):
    """TRAIN supervision: retain material/BC/hardware, replace ALL old forces."""
    if case['split']!='train':raise ValueError('Known-control teacher labels permit TRAIN only')
    if action['action_id'] not in {'poke_0','poke_1','poke_2','poke_3'}:
        raise ValueError('Evaluation/replay response not a teacher label')
    physical=physical_vector(case,spec,points)
    physical[16:22]=0.
    vector=np.asarray(action['peak_force_N'],dtype=np.float32)
    if vector.shape!=(3,) or not np.isfinite(vector).all():raise ValueError('Finite world force3-vector required')
    physical[16:19]=vector
    boundary,_=category_labels(case)
    family=int(action['force_type_index'])
    if family not in (0,2):raise ValueError('Unscheduled control family')
    return physical,boundary,family


def teacher_control_label(manifest,case,spec,points,action):
    """Guard the case/action role BEFORE constructing any supervision label."""
    require_teacher_control(manifest,case['episode_id'],action['action_id'],case['split'])
    return known_control_label(case,spec,points,action)
