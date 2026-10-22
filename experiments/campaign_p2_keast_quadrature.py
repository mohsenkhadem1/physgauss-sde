"""Positive 24-point degree-six bulk integration; old80 J guards retained.

Published numerical rule: P. Keast, CMAME55(3),339–348(1986),
https://doi.org/10.1016/0045-7825(86)90059-9
Coordinates/normalized weights checked against Burkardt's KEAST7 dataset:
https://people.math.sc.edu/burkardt/datasets/quadrature_rules_tet/keast7_x.txt
https://people.math.sc.edu/burkardt/datasets/quadrature_rules_tet/keast7_w.txt
This implementation expands symmetric barycentric orbits independently.

Scope: constant-material polynomial stable NH, affine-reference T10 only.
No exactness claim for logarithmic NH, contact activation or Gaussian material.
"""
from itertools import permutations
import numpy as np
from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
from experiments.campaign_static_p2_block import shapes


def positive_keast24():
    bary, weights = [], []
    for b,w in [(.2146028712591517,.0399227502581679),
                (.0406739585346113,.0100772110553207),
                (.3223378901422757,.0553571815436544)]:
        orbit=sorted(set(permutations((1-3*b,b,b,b))))
        bary.extend(orbit);weights.extend([w]*len(orbit))
    a,b=.0636610018750175,.2696723314583159
    orbit=sorted(set(permutations((a,a,b,1-2*a-b))))
    bary.extend(orbit);weights.extend([.0482142857142857]*len(orbit))
    points=np.asarray(bary,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    if points.shape!=(24,4) or np.any(points<=0) or np.any(w<=0):
        raise ValueError('Positive interior 24-point rule required')
    return points,w


class KeastBulkP2Geometry(ExactBulkP2Geometry):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Preserve all old80 quadrature guard locations in addition to the
        # original vertices/edge nodes. Changing the integration rule never
        # silently removes an existing positivity check.
        self.check_bary=np.concatenate([self.check_bary,self.bary])
        self.bary,self.weights=positive_keast24()
        self.N=shapes(self.bary)
        # The exact80-consistent body integral/load is intentionally retained
        # byte-for-byte; both rules integrate its degree-two shape functions.
        self.bulk_rule='Keast24_degree6_positive_with_old80_J_guards'
