"""Synthetic scenes with known truth. Truth is generated on the fine mesh (truth.py)."""
import math
import numpy as np
from scripts.patch_roles_v01 import model as M
from scripts.patch_roles_v01.truth import draw_history

LOGE, M1 = math.log(16000.), .3
CMD_HISTORY = (.3 * (.25 + .75 * np.sin(np.pi * M.TIMES)))[None]     # known command, N
CMD_SPEC = M.gspec('top', .035, .03, (0, 0, -1), name='known command top -z x=+35mm')

SCENES = ['i_published', 'ii_two_forces', 'ii_h_scale', 'iii_a_support_truth', 'iii_b_force_truth',
          'iv_decoy', 'v_a_tophat_short', 'v_b_nonlinear_support']


def get(name):
    """Return (truth spec, run settings)."""
    rng = np.random.default_rng(7000 + SCENES.index(name))
    gen = lambda face, x, s, **kw: dict(kind='gauss', x=x, phi=M.phi_of(face, s), **kw)
    settings = dict(K=3, hier=False, probe=False, family='general')
    if name == 'i_published':
        spec = dict(family='published', logE=LOGE, m1=.4, commands=[], cmd=np.zeros((0, 8)), patches=[
            dict(role='force', kind='pub', x=-.02, history=draw_history(rng, 2)),
            dict(role='support', kind='pub', x=.0125, k=8000.)])
        settings.update(family='published', regression=True)
    elif name in ('ii_two_forces', 'ii_h_scale'):
        scale = 1. if name == 'ii_two_forces' else 1.8
        spec = dict(family='general', logE=LOGE, m1=M1, commands=[], cmd=np.zeros((0, 8)), patches=[
            dict(role='force', **gen('top', -.03, .03, width=.01), history=draw_history(rng, 3, scale)),
            dict(role='force', **gen('front', .03, .025, width=.01), history=draw_history(rng, 3, scale)),
            dict(role='support', **gen('bottom', 0., .03, width=.02), k=8000.)], tau=scale)
        if name == 'ii_h_scale': settings.update(hier=True, probe=True)
    elif name == 'iii_a_support_truth':
        spec = dict(family='general', logE=LOGE, m1=M1, commands=[CMD_SPEC], cmd=CMD_HISTORY, patches=[
            dict(role='support', **gen('bottom', -.02, .03, width=.02), k=8000.),
            dict(role='support', **gen('bottom', .035, .03, width=.012), k=3000.)])
        settings.update(probe=True, fixed_probe=2)      # P top -z x=+40mm, above the confound
    elif name == 'iii_b_force_truth':
        h = np.zeros((8, 3)); h[:, 2] = .45 * CMD_HISTORY[0]                  # upward, synchronous with command
        spec = dict(family='general', logE=LOGE, m1=M1, commands=[CMD_SPEC], cmd=CMD_HISTORY, patches=[
            dict(role='support', **gen('bottom', -.02, .03, width=.02), k=8000.),
            dict(role='force', **gen('bottom', .035, .03, width=.012), history=h)])
        settings.update(probe=True, fixed_probe=2)
    elif name == 'iv_decoy':
        spec = dict(family='general', logE=LOGE, m1=M1, commands=[], cmd=np.zeros((0, 8)), patches=[
            dict(role='force', **gen('top', .02, .03, width=.01), history=draw_history(rng, 3)),
            dict(role='support', **gen('bottom', -.01, .03, width=.02), k=6000.)])
        settings.update(K=4)
    elif name == 'v_a_tophat_short':
        h = np.zeros((8, 3)); h[:, 2] = -np.abs(draw_history(rng, 1, 2.0, length=.10)[:, 0])   # pressure only, -z
        spec = dict(family='general', logE=LOGE, m1=M1, commands=[], cmd=np.zeros((0, 8)), patches=[
            dict(role='force', kind='tophat', x=.02, phi=M.phi_of('top', .03), radius=.015, history=h),
            dict(role='support', **gen('bottom', -.01, .03, width=.02), k=6000.)])
    elif name == 'v_b_nonlinear_support':
        spec = dict(family='general', logE=LOGE, m1=M1, commands=[], cmd=np.zeros((0, 8)), patches=[
            dict(role='force', **gen('top', .02, .03, width=.01), history=draw_history(rng, 3)),
            dict(role='support', **gen('bottom', -.01, .03, width=.02), k=6000., u0=1.5e-5)])
        settings.update(probe=True)
    else:
        raise KeyError(name)
    return spec, settings


def true_config(spec):
    return (sum(p['role'] == 'force' for p in spec['patches']), sum(p['role'] == 'support' for p in spec['patches']))
