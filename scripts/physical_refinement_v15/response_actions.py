"""Prospective response actions excluded from acquisition/design utility.

These pose inputs may be used to commit hypothesis predictions. No actual
reserved-world responses may enter policy selection or development tuning.
"""
import numpy as np
from scripts.physical_refinement_v15.study import initial_pose,development_actions


def response_actions():
    actions={}
    p=initial_pose();p[0,4]=-.10;p[1,4]=.10
    actions['response_opposed_y_bend']=p
    p=initial_pose();p[0,3]=.10;p[1,5]=-.10
    actions['response_cross_axis']=p
    p=initial_pose();p[1,3]=-.12
    actions['response_reverse_right_twist']=p
    p=initial_pose();p[0,0]=.001;p[1,0]=-.001;p[0,1]=-.004;p[1,1]=.004
    actions['response_shear_compression']=p
    for name,pose in actions.items():
        if any(np.array_equal(pose,known) for known in development_actions().values()):
            raise ValueError('Response pose duplicates an acquisition/design pose: '+name)
    return actions


def solve_actions():
    # Every response initializes independently from the same initial equilibrium.
    return {'initial':initial_pose(),**response_actions()}
