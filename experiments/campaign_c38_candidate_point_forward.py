"""Candidate-only static point forward library; never opens files or truth.

Stored grid labels describe hypothetical candidate physics, not an observed
object's material/support. Actual execution must separately verify the sealed
archive and finite forward-validation evidence. No continuous error certificate.
"""
import numpy as np
from experiments.campaign_c38_nonlinear_table_catalog import catalog
from experiments.campaign_c38_table_validation import POINTS

ARCHIVE_KEYS = {'source_index', 'case_ids', 'material_use', 'information_role',
                'cells', 'support', 'young_Pa', 'Fz_N', 'material_probes_m', 'probe_positions_m'}


class CandidatePointForward:
    def __init__(self, archive, *, cells=(40, 20, 20)):
        if set(archive) != ARCHIVE_KEYS or tuple(cells) not in ((32,16,16), (40,20,20)):
            raise ValueError('Exact candidate-only archive schema and declared mesh required')
        a = {k: np.asarray(v) for k, v in archive.items()}
        if a['information_role'].shape != () or str(a['information_role']) != 'CANDIDATE_POINT_FORWARD_TABLE_NOT_ADMITTED':
            raise ValueError('Direct withheld truth cannot become the candidate model')
        if not np.array_equal(a['material_probes_m'], POINTS) or a['probe_positions_m'].shape != (48,63,3) or not np.isfinite(a['probe_positions_m']).all():
            raise ValueError('Exact finite 63-point candidate fields required')
        for k in ('source_index', 'case_ids', 'material_use', 'support', 'young_Pa', 'Fz_N'):
            if a[k].shape != (48,):
                raise ValueError('Exact 48 table-node fields required')
        if a['cells'].shape != (48,3) or a['source_index'].dtype.kind not in 'iu':
            raise ValueError('Exact mesh and source-index arrays required')
        cases = catalog(); expected = {}
        for index, c in enumerate(cases):
            if c['material_use'] == 'TABLE_NODE':
                for m, n in enumerate(((32,16,16), (40,20,20))):
                    expected[2*index+m] = (c, n)
        if set(a['source_index'].tolist()) != set(expected) or len(np.unique(a['source_index'])) != 48:
            raise ValueError('Exact unique predeclared table indices required; no validation promotion')
        self.cells = tuple(cells)
        self.coefficients = {}
        joined = {}
        for i, source_index in enumerate(a['source_index']):
            c, n = expected[int(source_index)]
            if str(a['case_ids'][i]) != c['id'] or str(a['material_use'][i]) != 'TABLE_NODE' or str(a['support'][i]) != c['support'] or a['young_Pa'][i] != c['young_Pa'] or a['Fz_N'][i] != c['Fz_N'] or tuple(a['cells'][i]) != n:
                raise ValueError('Candidate source-index/material/support/force/mesh identity mismatch')
            if n == self.cells:
                joined[(c['support'],c['Fz_N'],c['young_Pa'])] = a['probe_positions_m'][i] - POINTS
        self.young_nodes = np.array(sorted({c['young_Pa'] for c in cases if c['material_use'] == 'TABLE_NODE'}))
        self.categories = tuple(sorted({c['support'] for c in cases}))
        self.forces_N = tuple(sorted({c['Fz_N'] for c in cases}))
        x = self.young_nodes[0] / self.young_nodes
        V = np.stack([np.ones(3), x, x*x], axis=1)
        for b in self.categories:
            for F in self.forces_N:
                u = np.array([joined[(b,F,E)] for E in self.young_nodes])
                coef = np.linalg.solve(V, u.reshape(3,-1)).reshape(3,63,3)
                coef.setflags(write=False)
                self.coefficients[(b,F)] = coef
        self.young_nodes.setflags(write=False)

    def observation_coefficients(self, category, Fz_N, point_ids):
        ids = np.asarray(point_ids)
        if category not in self.categories or not np.isfinite(Fz_N) or Fz_N not in self.forces_N:
            raise ValueError('Declared candidate support and discrete known force required')
        if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= 63):
            raise ValueError('Explicit unique qualified material-point IDs required')
        # Returned coefficients describe displacement, so known rest geometry
        # is subtracted from acquired positions before building a likelihood.
        return self.coefficients[(category,Fz_N)][:,ids,:].reshape(3,-1).copy()

    def predict_displacement(self, category, Fz_N, young_Pa, point_ids):
        if not np.isfinite(young_Pa) or not self.young_nodes[0] <= young_Pa <= self.young_nodes[-1]:
            raise ValueError('No material extrapolation')
        c = self.observation_coefficients(category, Fz_N, point_ids)
        x = self.young_nodes[0] / young_Pa
        return ((c[2]*x+c[1])*x+c[0]).reshape(-1,3)
