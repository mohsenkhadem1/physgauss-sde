"""Explicit temporal programs: physical time and actuator positions, not FM time."""
from dataclasses import dataclass
import math
import torch


@dataclass
class BoundaryProgram:
    """Piecewise-linear physical commands with explicit on/off event intervals.

    Values are [B,T,3] target displacements or tractions. Forecasting outside the
    supplied command horizon is rejected: a fitted history is not a future law.
    Values remain differentiable; event times are treated as discrete inputs.
    """
    times: torch.Tensor
    values: torch.Tensor
    active_start: float = float('-inf')
    active_end: float = float('inf')

    def __post_init__(self):
        if self.times.ndim!=1 or len(self.times)<2 or not bool(torch.isfinite(self.times).all()) or not bool((self.times[1:]>self.times[:-1]).all()):
            raise ValueError('Finite increasing command times required')
        if self.values.ndim!=3 or self.values.shape[1:]!=(len(self.times),3) or not bool(torch.isfinite(self.values).all()):
            raise ValueError('Finite command values [B,T,3] required')
        if math.isnan(self.active_start) or math.isnan(self.active_end) or self.active_start>=self.active_end:
            raise ValueError('Activation bounds must be ordered and not NaN')

    def at(self,time):
        t=torch.as_tensor(time,dtype=self.times.dtype,device=self.times.device)
        if t.ndim or not bool(torch.isfinite(t)) or bool((t<self.times[0])|(t>self.times[-1])):
            raise ValueError('Scalar query within supplied command horizon required')
        i=torch.searchsorted(self.times,t,right=True).sub(1).clamp(0,len(self.times)-2)
        dt=self.times[i+1]-self.times[i]
        velocity=(self.values[:,i+1]-self.values[:,i])/dt
        value=self.values[:,i]+(t-self.times[i])*velocity
        active=(t>=self.active_start)&(t<self.active_end)
        return value,velocity,active.to(self.values.dtype)
