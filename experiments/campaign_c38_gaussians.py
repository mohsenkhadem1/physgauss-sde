"""Compatible P2 material-Gaussian attachments, independent of inference.

One mass-carrying kernel per straight reference tetrahedron. Its center and
local F use the EXACT quadratic nodal field at the centroid. FΣ0Fᵀ is a local
Gaussian linearization of that field, not the exact pushforward distribution
under a quadratic map, nor an uncertainty covariance/J-positivity certificate.
"""
from dataclasses import dataclass
import torch
from experiments.campaign_material_gaussians import MaterialGaussianState

EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))

def p2_shapes(barycentric):
    b=barycentric
    if b.shape[-1]!=4:raise ValueError('Four barycentric coordinates required')
    return torch.cat([b*(2*b-1),torch.stack([4*b[...,i]*b[...,j] for i,j in EDGES],-1)],-1)

def p2_gradients(barycentric,linear_gradients):
    b,g=barycentric,linear_gradients
    return torch.cat([(4*b[...,None]-1)*g,torch.stack([4*(b[...,i,None]*g[...,j,:]+b[...,j,None]*g[...,i,:]) for i,j in EDGES],-2)],-2)

@dataclass(frozen=True)
class P2MaterialGaussians:
    rest_nodes:torch.Tensor
    elements:torch.Tensor
    reference_centers:torch.Tensor
    reference_covariances:torch.Tensor
    reference_volumes:torch.Tensor
    reference_density:torch.Tensor
    masses:torch.Tensor
    centroid_weights:torch.Tensor
    centroid_gradients:torch.Tensor

    @property
    def total_mass(self):return self.masses.sum()

    @property
    def mass_weights(self):return self.masses/self.total_mass

    @classmethod
    def from_elements(cls,rest_nodes,elements,density):
        x=rest_nodes
        if not isinstance(x,torch.Tensor) or x.dtype not in (torch.float32,torch.float64) or x.ndim!=2 or x.shape[1]!=3 or not bool(torch.isfinite(x).all()):raise ValueError('Finite float32/64 rest nodes[N,3] required')
        e=torch.as_tensor(elements,device=x.device)
        if e.ndim!=2 or e.shape[1]!=10 or e.dtype not in (torch.int32,torch.int64) or not len(e) or bool((e<0).any()) or bool((e>=len(x)).any()):raise ValueError('Valid P2 elements[E,10] required')
        e=e.long();corners=x[e[:,:4]];edges=(corners[:,1:]-corners[:,:1]).transpose(-1,-2);det=torch.linalg.det(edges)
        if bool((det<=0).any()):raise ValueError('Positive nondegenerate reference tetrahedra required')
        expected=torch.stack([(corners[:,i]+corners[:,j])/2 for i,j in EDGES],1)
        tolerance=32*torch.finfo(x.dtype).eps*max(float(torch.ptp(x,dim=0).max()) if hasattr(torch,'ptp') else float((x.max(0).values-x.min(0).values).max()),torch.finfo(x.dtype).tiny)
        if not bool(torch.allclose(x[e[:,4:]],expected,rtol=32*torch.finfo(x.dtype).eps,atol=tolerance)):raise ValueError('Straight reference P2 midpoint ordering required')
        inverse=torch.linalg.inv(edges);g=torch.cat([-inverse.sum(1)[:,None],inverse],1)
        b=x.new_full((len(e),4),.25);weights=p2_shapes(b);grad=p2_gradients(b,g)
        center=corners.mean(1);offset=corners-center[:,None];cov=offset.transpose(-1,-2)@offset/20
        if bool((torch.linalg.cholesky_ex(cov)[1]!=0).any()):raise ValueError('SPD physical reference kernel covariance required')
        rho=torch.as_tensor(density,dtype=x.dtype,device=x.device)
        if rho.ndim==0:rho=rho.expand(len(e))
        if rho.shape!=(len(e),) or not bool(torch.isfinite(rho).all()) or bool((rho<=0).any()):raise ValueError('Positive finite scalar/per-element density required')
        volume=det/6
        return cls(x,e,center,cov,volume,rho,rho*volume,weights,grad)

    def transport(self,nodes,*,validate=True):
        if nodes.shape[-2:]!=self.rest_nodes.shape or nodes.dtype!=self.rest_nodes.dtype or nodes.device!=self.rest_nodes.device:raise ValueError('Compatible [...,N,3] node dtype/device/shape required')
        if validate and not bool(torch.isfinite(nodes).all()):raise ValueError('Finite compatible nodal field required')
        u=(nodes-self.rest_nodes)[...,self.elements,:]
        centers=self.reference_centers+torch.einsum('ea,...eai->...ei',self.centroid_weights,u)
        F=torch.eye(3,dtype=nodes.dtype,device=nodes.device)+torch.einsum('...eai,eaJ->...eiJ',u,self.centroid_gradients)
        J=torch.linalg.det(F)
        if validate and (not bool(torch.isfinite(J).all()) or bool((J<=0).any())):raise ValueError('Positive finite local centroid deformation Jacobian required')
        covariance=F@self.reference_covariances@F.transpose(-1,-2)
        return MaterialGaussianState(centers,F,covariance,J,J*self.reference_volumes,self.reference_density/J,self.masses)
