"""Existing unchanged PhysTwin spring kernel with this study's shared contact.

NOT the full native PhysTwin reconstruction/training pipeline. GPU index is
logical zero under CUDA_VISIBLE_DEVICES=1; no other physical GPU is accessed.
"""
from scripts.coupled_contact_v13.native_adapter import NativeSpringSystem
from .bowl_physics import TriangleContact

class BowlNative(TriangleContact,NativeSpringSystem):
    def __init__(self,*args,device='cuda:0',**kwargs):
        super().__init__(*args,device=device,**kwargs)
