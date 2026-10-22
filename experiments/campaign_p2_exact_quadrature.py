"""Positive 80-point rule exact for degree-six tetrahedron polynomials.

Compatible affine-reference T10 makes F linear and the campaign's polynomial
stable-NH energy degree at most six. Under the Duffy map, weighted integrands
have coordinate degrees at most (8,7,6): Gauss orders (5,4,4) suffice. This is
not an exactness claim for logarithmic NH or clipped-contact surface integrals.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from experiments.campaign_p2_geometry import P2Geometry
from experiments.campaign_static_p2_block import shapes


def exact_degree6_rule():
    rules=[((x+1)/2,w/2) for x,w in [leggauss(n) for n in (5,4,4)]];b=[];weights=[]
    for i,a in enumerate(rules[0][0]):
        for j,c in enumerate(rules[1][0]):
            for k,d in enumerate(rules[2][0]):
                b.append([(1-a)*(1-c)*(1-d),a,(1-a)*c,(1-a)*(1-c)*d])
                weights.append(6*rules[0][1][i]*rules[1][1][j]*rules[2][1][k]*(1-a)**2*(1-c))
    return np.array(b),np.array(weights)


class ExactBulkP2Geometry(P2Geometry):
    def __init__(self,*args,**kwargs):
        gravity=kwargs.get('gravity',0.)
        super().__init__(*args,**kwargs)
        previous=self.body_integral_weights.copy();self.bary,self.weights=exact_degree6_rule();self.N=shapes(self.bary);self.order='positive_Duffy_5x4x4_degree6_exact'
        local=self.volumes[:,None]*np.einsum('q,qa->a',self.weights,self.N);self.body_integral_weights=np.zeros(len(self.rest));np.add.at(self.body_integral_weights,self.elements,local)
        self.load[:,2]+=gravity*self.density*(previous-self.body_integral_weights)
