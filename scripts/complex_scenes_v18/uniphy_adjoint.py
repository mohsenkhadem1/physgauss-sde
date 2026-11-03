"""Optional complete Torch/Taichi adjoint bridge; author forward is unchanged.

The released demo overwrites accumulated F adjoints and omits C/U/V neural-input
adjoints. This adapter accumulates all input paths through the existing author
SVD and MPM backward kernels. Finite-difference qualification remains required.
"""
import torch
import taichi as ti


@ti.data_oriented
class CompleteAdjointBridge:
    def __init__(self,sim):self.sim=sim

    @ti.kernel
    def add_stress_inputs(self,F:ti.types.ndarray(),C:ti.types.ndarray(),step:ti.i32):
        for p in range(self.sim.n_particles[None]):
            for i,j in ti.static(ti.ndrange(3,3)):
                self.sim.F.grad[p,step+1][i,j]+=F[p,i,j]
                self.sim.C.grad[p,step][i,j]+=C[p,i,j]

    @ti.kernel
    def add_projection_inputs(self,F:ti.types.ndarray(),U:ti.types.ndarray(),V:ti.types.ndarray(),step:ti.i32):
        for p in range(self.sim.n_particles[None]):
            for i,j in ti.static(ti.ndrange(3,3)):
                self.sim.F_tmp.grad[p,step+1][i,j]+=F[p,i,j]
                self.sim.U.grad[p,step+1][i,j]+=U[p,i,j]
                self.sim.V.grad[p,step+1][i,j]+=V[p,i,j]

    @staticmethod
    def accumulate_latent(parameter,gradient):
        if parameter.grad is None:parameter.grad=gradient.detach().clone()
        else:parameter.grad.add_(gradient.detach())

    def stress_grad(self,t,traj_id):
        s=self.sim;n=s.n_particles[None];F=s.F_substep[:n,t+1].detach().clone().requires_grad_(True);C=s.C_substep[:n,t].detach().clone().requires_grad_(True)
        s.get_stress_grad(t+1,s.stress_grad);z=s.stress_model.trajectory_latent.weight
        output=s.stress_model(F,C,traj_id);dF,dC,dz=torch.autograd.grad(output,[F,C,z],grad_outputs=s.stress_grad[:n,t+1].detach())
        self.add_stress_inputs(dF.contiguous(),dC.contiguous(),t);self.accumulate_latent(z,dz)

    def projection_grad(self,t,traj_id):
        s=self.sim;n=s.n_particles[None];F=s.Ftmp_substep[:n,t+1].detach().clone().requires_grad_(True);U=s.U_substep[:n,t+1].detach().clone().requires_grad_(True);V=s.V_substep[:n,t+1].detach().clone().requires_grad_(True)
        s.get_F_grad(t+1,s.F_grad);z=s.fproj_model.trajectory_latent.weight;output=s.fproj_model(F,U,V,traj_id)
        dF,dU,dV,dz=torch.autograd.grad(output,[F,U,V,z],grad_outputs=s.F_grad[:n,t+1].detach())
        self.add_projection_inputs(dF.contiguous(),dU.contiguous(),dV.contiguous(),t);self.accumulate_latent(z,dz)

    def attach(self):
        self.sim.F_to_stress_grad=self.stress_grad;self.sim.Ftmp_to_Fproj_grad=self.projection_grad
        return self
