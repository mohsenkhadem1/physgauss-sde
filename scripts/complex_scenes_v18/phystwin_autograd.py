"""Torch reverse-mode bridge through the unchanged author Warp spring kernel.

The bridge differentiates position, velocity and per-edge log stiffness. Native
dashpot damping is fixed here; boundary forces have their own Torch derivatives.
"""
import torch
import warp as wp
from scripts.complex_scenes_v18.phystwin_scene import kernels


class NativeSpringForce(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,v,log_young,edges,rest,damping):
        native=kernels();device=str(x.device)
        if x.dtype!=torch.float32 or v.dtype!=x.dtype or log_young.dtype!=x.dtype:raise ValueError('Native kernel uses float32')
        # Private buffers prevent Warp's gradient allocation changing leaf .grad.
        xx=x.detach().clone().contiguous();vv=v.detach().clone().contiguous();yy=log_young.detach().clone().contiguous()
        wx=wp.from_torch(xx,dtype=wp.vec3,requires_grad=True);wv=wp.from_torch(vv,dtype=wp.vec3,requires_grad=True);wy=wp.from_torch(yy,requires_grad=True)
        we=wp.from_torch(edges.contiguous(),dtype=wp.vec2i);wr=wp.from_torch(rest.contiguous())
        force=torch.zeros_like(x);wf=wp.from_torch(force,dtype=wp.vec3,requires_grad=True);empty=wp.zeros(0,dtype=wp.vec3,device=device)
        tape=wp.Tape()
        with tape:
            wp.launch(native.eval_springs,dim=len(edges),inputs=[wx,wv,empty,empty,len(x),we,wr,wy,float(damping),0.,1e8,wf],device=device)
        ctx.tape=tape;ctx.arrays=(wx,wv,wy,wf,we,wr,empty);ctx.buffers=(xx,vv,yy,force,edges,rest)
        return force

    @staticmethod
    def backward(ctx,gradient):
        wx,wv,wy,wf,*_=ctx.arrays
        ctx.tape.backward(grads={wf:wp.from_torch(gradient.contiguous(),dtype=wp.vec3)})
        return wp.to_torch(wx.grad).clone(),wp.to_torch(wv.grad).clone(),wp.to_torch(wy.grad).clone(),None,None,None


def spring_force(x,v,log_young,edges,rest,damping=0.):
    return NativeSpringForce.apply(x,v,log_young,edges,rest,damping)
