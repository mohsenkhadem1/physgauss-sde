"""Linear candidate model for PUBLIC observation inference, no private E/BC.

Constructed from all 28 previously sealed supplied-physics query fields. Grid
labels identify hypotheses, not the unknown observed object. The executor must
verify those capture hashes before supplying rows. No file or observation I/O.
"""
import numpy as np
from experiments.campaign_c38_candidate_point_forward import CandidatePointForward
from experiments.campaign_c38_table_validation import POINTS
from experiments.run_campaign_c38_static_prequalification import cases


class LinearCandidatePointForward(CandidatePointForward):
    def __init__(self, rows, *, cells=(40,20,20)):
        expected = {c['id']: c for c in cases()}
        if len(rows) != 28 or tuple(cells) not in ((32,16,16),(40,20,20)):
            raise ValueError('All 28 sealed linear query fields and declared mesh required')
        joined = {}
        for row in rows:
            c = row['case']; n = row['cells']
            if c['id'] not in expected or any(c[k] != expected[c['id']][k] for k in ('young_Pa','support','Fz_N')) or tuple(n) not in ((32,16,16),(40,20,20)):
                raise ValueError('Exact sealed case/material/support/force/mesh required')
            key = (c['support'],c['Fz_N'],c['young_Pa'],n[0])
            q = np.asarray(row['probe_positions_m'])
            if key in joined or not np.array_equal(row['material_probes_m'],POINTS) or q.shape != (63,3) or not np.isfinite(q).all():
                raise ValueError('Unique finite exactly bound 63-point linear fields required')
            joined[key] = q - POINTS
        self.cells = tuple(cells)
        self.young_nodes = np.array([75000.,np.sqrt(75000.*150000.),150000.])
        self.young_nodes.setflags(write=False)
        self.categories = tuple(sorted({c['support'] for c in expected.values()}))
        self.forces_N = (-2.,0.,1.,2.)
        self.coefficients = {}
        tol = 64*np.finfo(float).eps*max(float(np.ptp(POINTS,axis=0).max()),float(abs(POINTS).max()))
        for b in self.categories:
            for nx in (32,40):
                gravity = joined[(b,0.,75000.,nx)]
                unit = (joined[(b,2.,75000.,nx)]-joined[(b,-2.,75000.,nx)])/4
                for c in expected.values():
                    if c['support'] == b:
                        actual = joined[(b,c['Fz_N'],c['young_Pa'],nx)]
                        predicted = (75000./c['young_Pa'])*(gravity+c['Fz_N']*unit)
                        if np.max(abs(actual-predicted)) > tol:
                            raise ValueError('Sealed point fields violate linear E/force scaling beyond roundoff')
                if nx == self.cells[0]:
                    for F in self.forces_N:
                        coef = np.zeros((3,63,3));coef[1] = gravity+F*unit
                        coef.setflags(write=False);self.coefficients[(b,F)] = coef
        self.scope = 'Linear working-model candidate for unknown-physics inference; not an oracle label input or nonlinear equilibrium'
