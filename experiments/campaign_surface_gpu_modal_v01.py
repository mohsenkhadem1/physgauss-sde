"""Optional pinned-device float64 modal evaluation; no automatic GPU use.

CPU computes the identical physical parameter decoding. Only denominator and
matrix arithmetic run on the explicitly allowed GPU, retaining every mode.
Independent CPU parity and resource qualification belong to the caller.
"""
import os
import numpy as np
from experiments.campaign_surface_p2_inference_v01 import LabelledModal,physical_parameters

FORBIDDEN_GPU_UUID='GPU-9dbe9939-379d-9560-2028-462a2f8b3ac1'


class Float64GPUModal(LabelledModal):
    def __init__(self,modal,labels,*,physical_UUID,allowed_UUIDs,maximum_batch=4096,allocated_cap_bytes=512*1024**2):
        if physical_UUID==FORBIDDEN_GPU_UUID or physical_UUID not in tuple(allowed_UUIDs) or os.environ.get('CUDA_VISIBLE_DEVICES')!=physical_UUID:
            raise ValueError('Explicit allowed GPU UUID must be the only visible device; GPU3 forbidden')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Frozen deterministic cuBLAS workspace required')
        if not isinstance(maximum_batch,int) or maximum_batch<1 or allocated_cap_bytes<=0:raise ValueError('Positive bounded GPU resources')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one pinned CUDA device required')
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
        super().__init__(modal,labels);self.torch=torch;self.device=torch.device('cuda:0');self.physical_UUID=physical_UUID
        self.maximum_batch=maximum_batch;self.allocated_cap_bytes=allocated_cap_bytes
        free,_=torch.cuda.mem_get_info(self.device)
        if free<2*1024**3:raise MemoryError('At least2GiB free VRAM required before owned GPU initialization')
        self.GPU_eigenvalues=torch.as_tensor(np.asarray(modal.eigenvalues),dtype=torch.float64,device=self.device)
        actual_UUID=getattr(torch.cuda.get_device_properties(self.device),'uuid',None)
        if actual_UUID is not None and str(actual_UUID).removeprefix('GPU-')!=physical_UUID.removeprefix('GPU-'):
            raise RuntimeError('CUDA device UUID differs from the pinned physical device')
        self.GPU_coefficients={};self._guard()

    def _guard(self):
        if self.torch.cuda.max_memory_allocated(self.device)>self.allocated_cap_bytes:raise MemoryError('Owned GPU allocation cap exceeded')

    def select(self,labels):
        labels=tuple(labels)
        if not labels:raise ValueError('Nonempty labels required')
        if labels not in self.GPU_coefficients:
            # Parent establishes the identical label/output ordering first.
            super().select(labels)
            if self.torch.cuda.memory_allocated(self.device)+self._selected[labels].nbytes>self.allocated_cap_bytes:
                raise MemoryError('Coefficient allocation preflight exceeds declared cap')
            self.GPU_coefficients[labels]=self.torch.as_tensor(np.ascontiguousarray(self._selected[labels].T),dtype=self.torch.float64,device=self.device)
            self._guard()
        coefficients=self.GPU_coefficients[labels];torch=self.torch
        def predict(unit):
            unit=np.asarray(unit,float)
            if len(unit)>self.maximum_batch:raise ValueError('Split input into declared bounded GPU batches')
            theta=physical_parameters(unit)
            # Conservative floats/intermediate-mask/output plus workspace bound.
            projected=self.torch.cuda.memory_allocated(self.device)+8*len(theta)*(3*len(self.modal.eigenvalues)+len(labels)*self.channels)+16*1024**2
            if projected>self.allocated_cap_bytes:raise MemoryError('Batch allocation preflight exceeds declared cap')
            with torch.no_grad():
                values=torch.as_tensor(theta,dtype=torch.float64,device=self.device)
                denominator=values[:,0,None]+values[:,1,None]*self.GPU_eigenvalues
                if not bool(torch.isfinite(denominator).all()) or bool((denominator<=0).any()):raise FloatingPointError('Invalid modal denominator')
                result=((1/denominator)@coefficients).reshape(len(theta),len(labels),self.channels).cpu().numpy()
            self._guard();return result
        return predict
