"""Exact scalar adaptive-node memoization; no approximation or likelihood change."""
from functools import lru_cache
import numpy as np
from experiments.campaign_c38_bar_posterior import BarPosteriorBatch

class CachedBarPosteriorBatch(BarPosteriorBatch):
    def __init__(self,*args,**kwargs):
        self._scalar_parts=lru_cache(maxsize=4096)(lambda u:super(CachedBarPosteriorBatch,self).parts(u))
        super().__init__(*args,**kwargs)
    def parts(self,u):
        if np.ndim(u)==0:return self._scalar_parts(float(u))
        return super().parts(u)
