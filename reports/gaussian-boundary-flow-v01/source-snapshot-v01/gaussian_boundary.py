"""SI-unit Gaussian surface operators, separate from appearance Gaussians.

Reference-area integration gives a consistent surface force. Material-frame
Robin operators are conservative for static anchors. World/actuator gates are
available for contact localization, not silently treated as conservative springs.
"""
from dataclasses import dataclass
from collections import Counter
import torch


def surface_quadrature(mesh):
    """Three-point triangle rule; N maps nodal displacements to quadrature points."""
    faces = Counter(tuple(sorted(face)) for tet in mesh.cells.tolist()
                    for face in ((tet[0],tet[1],tet[2]),(tet[0],tet[1],tet[3]),
                                 (tet[0],tet[2],tet[3]),(tet[1],tet[2],tet[3])))
    ids = torch.tensor([f for f,n in faces.items() if n == 1], device=mesh.rest.device)
    xyz = mesh.rest[ids]
    areas = torch.linalg.vector_norm(torch.linalg.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),dim=-1)/2
    bary = mesh.rest.new_tensor([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    N = mesh.rest.new_zeros((len(ids),3,len(mesh.rest)))
    for j in range(3):
        N.scatter_(2,ids[:,None,j,None].expand(-1,3,1),bary[None,:,j,None].expand(len(ids),-1,-1))
    N = N.flatten(0,1)
    return N @ mesh.rest, areas[:,None].expand(-1,3).flatten()/3, N


def gaussian_gate(points, center, covariance, *, rotation=None, translation=None):
    """World points -> task frame; rotation columns are frame axes in world.

    No mixture normalization: influence tends to zero far from every kernel.
    Covariance is spatial extent, NOT a physical stiffness or posterior variance.
    """
    if rotation is not None:
        points = (points-translation) @ rotation
    L = torch.linalg.cholesky(covariance)
    delta = points-center[...,None,:]
    white = torch.linalg.solve_triangular(L,delta.transpose(-1,-2),upper=False)
    return torch.exp(-.5*white.square().sum(-2))


def psd_matrix(factor):
    return factor @ factor.transpose(-1,-2)


def robin_matrices(N, areas, gates, stiffness, damping):
    """K/D [B,3,3], gates [B,Q], output consistent [B,3N,3N] operators."""
    scalar = torch.einsum('qi,bq,q,qj->bij',N,gates,areas,N)
    def expand(A):
        return torch.einsum('bij,bkl->bikjl',scalar,A).flatten(-2).flatten(-3,-2)
    return expand(stiffness), expand(damping)


def traction_force(N, areas, gates, traction):
    """Traction in N/m^2, returned nodal force in N."""
    return torch.einsum('qi,q,bq,bj->bij',N,areas,gates,traction)


@dataclass
class LinearBoundarySystem:
    """Backward Euler, exact prescribed DOFs via elimination, no inverse shooting.

    Constant linear elastic/material-frame Robin matrices allow one factorization
    per parameter sample. Changing contact topology requires a different solver.
    """
    mass: torch.Tensor
    stiffness: torch.Tensor
    damping: torch.Tensor
    dt: float
    fixed: torch.Tensor | None = None

    def __post_init__(self):
        if self.dt <= 0 or bool((self.mass <= 0).any()):
            raise ValueError('Positive dt and masses required')
        n = self.stiffness.shape[-1]
        if self.fixed is None:
            self.fixed = torch.zeros(n,dtype=torch.bool,device=self.mass.device)
        self.free = (~self.fixed).nonzero().flatten()
        self.A = torch.diag_embed(self.mass)+self.dt*self.damping+self.dt**2*self.stiffness
        self.L = torch.linalg.cholesky(self.A[...,self.free[:,None],self.free])

    def step(self,u,v,force,target=None):
        rhs = self.mass*v+self.dt*(force-torch.einsum('bij,bj->bi',self.stiffness,u))
        next_v = torch.zeros_like(v)
        if bool(self.fixed.any()):
            if target is None: target = torch.zeros_like(u)
            next_v[:,self.fixed] = (target[:,self.fixed]-u[:,self.fixed])/self.dt
            rhs = rhs-torch.einsum('bij,bj->bi',self.A,next_v)
        free_rhs = rhs[:,self.free,None]
        next_v[:,self.free] = torch.cholesky_solve(free_rhs,self.L).squeeze(-1)
        next_u = u+self.dt*next_v
        # M dv/dt + D v_next + K u_next = applied + reaction.
        reaction = self.mass*(next_v-v)/self.dt+torch.einsum('bij,bj->bi',self.damping,next_v)+torch.einsum('bij,bj->bi',self.stiffness,next_u)-force
        return next_u,next_v,reaction


def plane_impact(positions, velocities, masses, normal, offset, restitution=0., friction=0.):
    """Explicit isolated nodal plane-impact primitive, not a coupled FEM solver.

    Returns normal/tangential impulses with no attraction and Coulomb bound.
    Penetration repair is reported separately; do not claim exact impact timing.
    """
    if not 0<=restitution<=1 or friction<0:
        raise ValueError('Invalid restitution/friction')
    normal = normal/torch.linalg.vector_norm(normal)
    gap = positions @ normal-offset
    vn = velocities @ normal
    hit = (gap<=0)&(vn<0)
    jn = torch.where(hit,-(1+restitution)*masses*vn,0.)
    tangent = velocities-vn[...,None]*normal
    speed = torch.linalg.vector_norm(tangent,dim=-1)
    jt = -torch.minimum(masses*speed,friction*jn)[...,None]*tangent/speed.clamp_min(1e-15)[...,None]
    impulse = jn[...,None]*normal+jt
    repaired = positions-gap.clamp_max(0)[...,None]*normal
    return repaired,velocities+impulse/masses[...,None],impulse
