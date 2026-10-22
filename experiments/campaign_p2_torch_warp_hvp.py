"""Prospective zero-copy Torch/Warp elastic-HVP mixin only.

All original Torch energy, gradient, J guards, node blocks, mass, preconditioner,
CG acceptance, and final physical residual remain inherited. Warp launches use
Torch's CURRENT CUDA stream. No host array transfer, stored Q*E gradients,
contact, constitutive change, or integration change is introduced here.
"""
import torch
import warp as wp
from experiments.campaign_p2_warp_bulk_hvp import bulk_hvp

class TorchWarpBulkHVPMixin:
    """Place before StreamingTwoLevelP2/StreamingSupportedP2 in the MRO.

    Persistent geometry is aliased read-only. Every result owns fresh Torch
    storage, preserving previous CG products. This is not an autograd operator.
    CPU interop is tested; CUDA use requires separately frozen GPU parity.
    """
    def initialize_warp_hvp(self):
        wp.init()
        tensors=dict(elements=self.elements,linear_gradients=self.linear_gradients,
                     bary=self.bary,weights=self.quadrature_weights,volumes=self.volumes)
        if self.rest.dtype!=torch.float64:raise ValueError('FP64 mechanics required')
        for name,t in tensors.items():
            dtype=torch.int64 if name=='elements' else torch.float64
            if t.dtype!=dtype or t.device!=self.rest.device or not t.is_contiguous() or t.requires_grad:raise ValueError('Immutable contiguous matching geometry required: '+name)
        self._warp_hvp_tensors=tensors
        self._warp_hvp_geometry={name:wp.from_torch(t,requires_grad=False) for name,t in tensors.items()}
        self._warp_hvp_device=wp.device_from_torch(self.rest.device)
        self._warp_hvp_mu=self.young/(2*(1+self.poisson))
        self._warp_hvp_kappa=self.young*self.poisson/((1+self.poisson)*(1-2*self.poisson))+self._warp_hvp_mu

    def tangent(self,cache,v):
        if v.shape!=self.rest.shape or v.dtype!=torch.float64 or v.device!=self.rest.device or v.requires_grad or not bool(torch.isfinite(v).all()):raise ValueError('Finite device-matched FP64 direction required')
        local=cache['local_u']
        if local.shape!=(len(self.elements),10,3) or local.dtype!=torch.float64 or local.device!=self.rest.device or local.requires_grad:raise ValueError('Accepted original local displacement cache required')
        if not hasattr(self,'_warp_hvp_geometry'):self.initialize_warp_hvp()
        local=local.contiguous();direction=v.contiguous();out=torch.zeros_like(self.rest)
        # Warp views retain Torch owners. All allocated output/input storage is
        # Torch-owned, so the same current stream orders zeroing/launch/use and
        # the Torch allocator's normal stream lifetime semantics apply.
        inputs=[wp.from_torch(local,requires_grad=False),wp.from_torch(direction,requires_grad=False)]
        a=self._warp_hvp_geometry
        inputs += [a['elements'],a['linear_gradients'],a['bary'],a['weights'],a['volumes'],self._warp_hvp_mu,self._warp_hvp_kappa,wp.from_torch(out,requires_grad=False)]
        if self.rest.device.type=='cuda':
            stream=wp.stream_from_torch(torch.cuda.current_stream(self.rest.device))
            wp.launch(bulk_hvp,dim=(len(self.bary),len(self.elements)),inputs=inputs,device=self._warp_hvp_device,stream=stream)
        else:
            wp.launch(bulk_hvp,dim=(len(self.bary),len(self.elements)),inputs=inputs,device=self._warp_hvp_device)
        return out
