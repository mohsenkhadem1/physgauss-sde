"""Evaluate the unchanged atlas polynomial with cached constants and matmul.

No parameter predictions are cached or approximated. The cache holds only
exact selections of the immutable correction table. Input checks, original
nodes, exact-node semantics and roundoff halo are preserved.
"""
from collections import OrderedDict
import numpy as np


class FastResidualAtlas:
    def __init__(self, atlas, *, table_cache_bytes=8*1024**2):
        if type(table_cache_bytes) is not int or table_cache_bytes < 1:
            raise ValueError('Positive exact-table cache budget')
        self.base = atlas
        self.axis = atlas.axis
        self.delta = atlas.delta
        self.labels = atlas.labels
        self.fingerprint = atlas.fingerprint
        self.provenance_fingerprint = atlas.provenance_fingerprint
        self.weights = (-1.)**np.arange(len(self.axis))
        self.weights[[0, -1]] *= .5
        self.weights.setflags(write=False)
        self.table_cache_bytes = table_cache_bytes
        self.cache = OrderedDict()
        self.allocated_bytes = 0

    def _basis(self, query):
        distance = query[:, None]-self.axis
        hits = distance == 0
        exact = hits.any(axis=1)
        basis = np.empty_like(distance)
        basis[exact] = hits[exact]
        if np.any(~exact):
            raw = self.weights/distance[~exact]
            basis[~exact] = raw/raw.sum(axis=1)[:, None]
        if not np.isfinite(basis).all():
            raise FloatingPointError('Unresolved interpolation basis')
        return basis

    def _table(self, selected):
        if selected in self.cache:
            value = self.cache.pop(selected)
            self.cache[selected] = value
            return value
        indices = [self.labels.index(v) for v in selected]
        value = np.ascontiguousarray(self.delta[:, :, indices, :]).reshape(len(self.axis)**2, -1)
        value.setflags(write=False)
        if value.nbytes <= self.table_cache_bytes:
            while self.allocated_bytes+value.nbytes > self.table_cache_bytes:
                _, removed = self.cache.popitem(last=False)
                self.allocated_bytes -= removed.nbytes
            self.cache[selected] = value
            self.allocated_bytes += value.nbytes
        return value

    def predict(self, unit, labels, *, chunk=4096):
        u = np.asarray(unit, float)
        selected = tuple(labels)
        if u.ndim != 2 or u.shape[1] != 2 or not np.isfinite(u).all() or not selected:
            raise ValueError('Finite unit[batch,2] and nonempty labels')
        if type(chunk) is not int or chunk < 1:
            raise ValueError('Bounded positive query chunk')
        halo = 64*np.finfo(float).eps
        if np.any(u < -halo) or np.any(u > 1+halo):
            raise ValueError('No extrapolation outside prior beyond roundoff halo')
        table = self._table(selected)
        output = np.empty((len(u), len(selected), self.delta.shape[3]))
        for start in range(0, len(u), chunk):
            query = u[start:start+chunk]
            a = self._basis(query[:, 0])
            b = self._basis(query[:, 1])
            product = (a[:, :, None]*b[:, None, :]).reshape(len(query), -1)
            output[start:start+len(query)] = (product @ table).reshape(len(query), len(selected), -1)
        if not np.isfinite(output).all():
            raise FloatingPointError('Nonfinite interpolated correction')
        return output

    def coarsen(self, n):
        return FastResidualAtlas(self.base.coarsen(n), table_cache_bytes=self.table_cache_bytes)
