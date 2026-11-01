"""Public development design; no reserved truth construction or response access.

These settings are a proposal until the end-to-end development gate and final
protocol/source freeze. Predictions for design actions are not test responses.
"""
import numpy as np

GRADIENTS = (-.8, -.4, 0., .4, .8)
SCALES = (.75, 1., 1.5)
RATIOS = (.75, 1.25)
LAYOUTS = ('parallel_lower', 'crossed')


def initial_pose():
    pose = np.zeros((2, 6))
    pose[0, 0], pose[1, 0] = -.0005, .0005
    return pose


def development_actions():
    actions = {'initial': initial_pose()}
    for side in (0, 1):
        for axis, label in ((0, 'twist'), (2, 'bend')):
            pose = initial_pose()
            pose[side, 3 + axis] = .12
            actions[f'probe_{"left" if side == 0 else "right"}_{label}'] = pose
    pose = initial_pose()
    pose[0, 3], pose[1, 3] = -.15, .15
    actions['design_opposed_twist'] = pose
    pose = initial_pose()
    pose[1, 5] = .15
    actions['design_right_bend'] = pose
    pose = initial_pose()
    pose[0, 1], pose[1, 1] = -.006, .006
    actions['design_opposed_translation'] = pose
    return actions


def base_cases():
    # Exact common stiffness scaling is qualified separately. Only these 20
    # distinct geometrical responses need solving for the 60 prior hypotheses.
    cases = []
    for layout in LAYOUTS:
        for j, gradient in enumerate(GRADIENTS):
            for k, ratio in enumerate(RATIOS):
                cases.append(dict(name=f'{layout}-g{j}-r{k}', E=12000.,
                                  Kleft=6000.*ratio, Kright=6000.*ratio,
                                  layout=layout, material_gradient=gradient))
    return cases


def hypotheses():
    rows = []
    for case in base_cases():
        for scale in SCALES:
            row = dict(case)
            row['base_name'] = row.pop('name')
            row['scale'] = scale
            for key in ('E', 'Kleft', 'Kright'):
                row[key] *= scale
            rows.append(row)
    return rows
