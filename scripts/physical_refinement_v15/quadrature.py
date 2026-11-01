"""Continuous-prior quadrature; refining common scale requires no new mechanics.

The original 60 categorical hypotheses remain a separate coarse ablation.
Scale invariance applies only with zero external load and common material/support
scaling; it does not apply to gravity or independently changed support stiffness.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scripts.physical_refinement_v15.study import base_cases, LAYOUTS, RATIOS


def scale_rule(count=25):
    if count not in (25, 49, 97):
        raise ValueError('Only prospectively declared scale refinements admitted')
    nodes, weights = leggauss(count)
    lower, upper = np.log(.75), np.log(1.5)
    return np.exp(lower + (nodes + 1) * (upper - lower) / 2), weights / 2


def gradient_rule(count=5):
    if count not in (5, 9, 17):
        raise ValueError('Only prospectively declared gradient refinements admitted')
    nodes = np.linspace(-.8, .8, count)
    weights = np.ones(count)
    weights[[0, -1]] = .5
    return nodes, weights / weights.sum()


def geometry_cases(count=5):
    existing = base_cases()
    rows = []
    for layout in LAYOUTS:
        for gradient in gradient_rule(count)[0]:
            for k, ratio in enumerate(RATIOS):
                match = next((row for row in existing if row['layout'] == layout and
                              abs(row['material_gradient'] - gradient) < 1e-12 and
                              row['Kleft'] == 6000*ratio), None)
                if match is not None:
                    rows.append(dict(match))
                else:
                    tag = ('m' if gradient < 0 else 'p') + f'{round(abs(gradient)*1e6):06d}'
                    rows.append(dict(name=f'{layout}-gref{tag}-r{k}', E=12000.,
                                     Kleft=6000.*ratio, Kright=6000.*ratio,
                                     layout=layout, material_gradient=float(gradient)))
    return rows


def hypotheses(gradient_count=5, scale_count=25):
    gradients, gradient_weights = gradient_rule(gradient_count)
    scales, scale_weights = scale_rule(scale_count)
    rows, weights = [], []
    for case in geometry_cases(gradient_count):
        j = int(np.argmin(abs(gradients-case['material_gradient'])))
        for scale, scale_weight in zip(scales, scale_weights):
            row = dict(case)
            row['base_name'] = row.pop('name')
            row['scale'] = float(scale)
            for key in ('E', 'Kleft', 'Kright'):
                row[key] *= float(scale)
            rows.append(row)
            weights.append(gradient_weights[j]*scale_weight/(len(LAYOUTS)*len(RATIOS)))
    weights = np.asarray(weights)
    if not np.isclose(weights.sum(), 1.):
        raise ValueError('Prior quadrature failed normalization')
    return rows, weights
