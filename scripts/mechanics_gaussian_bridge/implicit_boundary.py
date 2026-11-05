"""Exact equilibrium adjoint for inferred attachment targets and material parameters.
Uses the paper's existing NeuralFEM residual, Hessian and converged solve.
"""
import torch

class BoundaryEquilibrium(torch.autograd.Function):
 @staticmethod
 def forward(ctx,theta,targets,model,frame,initial):
  old=model.targets
  try:
   model.targets=targets;x,H,diagnostic=model.solve(theta,frame,initial);model.last_diagnostic=diagnostic;model.last_solved_state=x.detach()
  finally:model.targets=old
  ctx.save_for_backward(theta,targets,x,H);ctx.model=model;ctx.frame=frame;return x
 @staticmethod
 def backward(ctx,gradient):
  theta,targets,x,H=ctx.saved_tensors;adjoint=torch.linalg.solve(H.T,gradient.reshape(-1)).reshape_as(x);model=ctx.model;old=model.targets
  try:
   with torch.enable_grad():
    th=theta.detach().requires_grad_();target=targets.detach().requires_grad_();model.targets=target
    _,residual,_=model.evaluate(x.detach(),th,ctx.frame)
    gt,gc=torch.autograd.grad(residual,(th,target),grad_outputs=-adjoint,allow_unused=False)
  finally:model.targets=old
  return gt,gc,None,None,None
