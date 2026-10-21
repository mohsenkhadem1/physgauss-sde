"""Mass-carrying material Gaussians transported by a compatible nodal field.

One kernel per tetrahedron uses the exact reference centroid and uniform-tet
second moment. Covariances describe physical kernel SHAPE; posterior/material/
boundary uncertainty belongs to a separate model. No mechanics solver or
renderer is invoked. Coordinates use metres, volume m³ and mass kg.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


def _floating(value, name):
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise TypeError(name+' must be a float32/float64 tensor')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name+' must be finite')


def _indices(value, device, width, name):
    value=torch.as_tensor(value,device=device)
    if value.ndim!=2 or value.shape[1]!=width or not value.numel() or value.dtype not in (torch.int32,torch.int64):
        raise ValueError(name+' must be a nonempty integer index array')
    return value.long()


@dataclass(frozen=True)
class MaterialGaussianState:
    centers: torch.Tensor                   # [..., T, 3]
    deformation_gradients: torch.Tensor     # [..., T, 3, 3]
    covariances: torch.Tensor               # [..., T, 3, 3], physical shape
    jacobians: torch.Tensor                 # [..., T]
    volumes: torch.Tensor                   # [..., T], current volume
    bulk_density: torch.Tensor              # [..., T], rho0 / J
    masses: torch.Tensor                    # [T], conserved

    @property
    def total_mass(self):
        return self.masses.sum()

    @property
    def mass_weights(self):
        return self.masses/self.total_mass

    def log_density(self, queries, *, normalized=False):
        """Evaluate mass density; queries[..., Q, 3] broadcast over state batches.

        normalized=True gives unit-integral material sampling density (m^-3).
        Otherwise the Gaussian mixture integrates to total mass (kg).
        """
        _floating(queries,'queries')
        if queries.ndim<2 or queries.shape[-1]!=3:
            raise ValueError('queries must end in [Q,3]')
        if queries.dtype!=self.centers.dtype or queries.device!=self.centers.device:
            raise ValueError('queries and state must share dtype/device')
        factor=torch.linalg.cholesky(self.covariances)
        delta=queries.unsqueeze(-2)-self.centers.unsqueeze(-3)
        # Explicit quadratic form avoids batch-vector outer-product backward
        # kernels that require a JIT compiler in some Torch GPU installations.
        precision=torch.cholesky_inverse(factor)
        squared_distance=(delta.unsqueeze(-1)*precision.unsqueeze(-4)*delta.unsqueeze(-2)).sum((-2,-1))
        log_kernel=-.5*squared_distance-1.5*math.log(2*math.pi)
        log_kernel-=factor.diagonal(dim1=-2,dim2=-1).log().sum(-1).unsqueeze(-2)
        weights=self.mass_weights if normalized else self.masses
        return torch.logsumexp(log_kernel+weights.log(),dim=-1)

    def density(self, queries):
        """Smooth mass density (kg/m³), with exact mixture mass normalization."""
        return self.log_density(queries).exp()

    def normalized_occupancy(self, queries):
        """Unit-integral material-mass sampling density, suitable for attachments."""
        return self.log_density(queries,normalized=True).exp()


@dataclass(frozen=True)
class MaterialGaussians:
    rest_vertices: torch.Tensor
    tetrahedra: torch.Tensor
    reference_centers: torch.Tensor
    reference_covariances: torch.Tensor
    reference_volumes: torch.Tensor
    reference_density: torch.Tensor
    masses: torch.Tensor
    inverse_reference_edges: torch.Tensor

    @classmethod
    def from_tetrahedra(cls, rest_vertices, tetrahedra, density):
        """Build a shared reference mesh; density is scalar or one value per tet.

        The uniform-tetrahedron covariance is sum_i (v_i-c)(v_i-c)^T /20.
        Signed mesh winding cancels in F; inverted deformations are rejected.
        No covariance jitter or stiffness/geometry repair is added.
        """
        _floating(rest_vertices,'rest_vertices')
        if rest_vertices.ndim!=2 or rest_vertices.shape[1]!=3:
            raise ValueError('rest_vertices must have shape [N,3]')
        ids=_indices(tetrahedra,rest_vertices.device,4,'tetrahedra')
        if bool((ids<0).any()) or bool((ids>=len(rest_vertices)).any()):
            raise ValueError('tetrahedron index outside vertex array')
        corners=rest_vertices[ids]
        edges=(corners[:,1:]-corners[:,:1]).transpose(-1,-2)
        determinant=torch.linalg.det(edges)
        if not bool(torch.isfinite(determinant).all()) or bool((determinant==0).any()):
            raise ValueError('Nondegenerate finite reference tetrahedra required')
        volume=determinant.abs()/6
        center=corners.mean(-2);offset=corners-center.unsqueeze(-2)
        covariance=offset.transpose(-1,-2)@offset/20
        _,info=torch.linalg.cholesky_ex(covariance)
        if bool((info!=0).any()):
            raise ValueError('Reference physical shape covariance is not SPD')
        rho=torch.as_tensor(density,dtype=rest_vertices.dtype,device=rest_vertices.device)
        if rho.ndim==0:rho=rho.expand(len(ids))
        if rho.shape!=(len(ids),) or not bool(torch.isfinite(rho).all()) or bool((rho<=0).any()):
            raise ValueError('Positive finite scalar or per-tetrahedron density required')
        inverse=torch.linalg.inv(edges)
        if not bool(torch.isfinite(inverse).all()):
            raise ValueError('Reference edge inverse is not finite')
        return cls(rest_vertices,ids,center,covariance,volume,rho,rho*volume,inverse)

    @property
    def total_mass(self):
        return self.masses.sum()

    @property
    def mass_weights(self):
        return self.masses/self.total_mass

    @property
    def volume_weights(self):
        return self.reference_volumes/self.reference_volumes.sum()

    def to(self, device=None, dtype=None):
        dtype=self.rest_vertices.dtype if dtype is None else dtype
        if dtype not in (torch.float32,torch.float64):
            raise TypeError('Material Gaussians require float32 or float64')
        fields={name:(value.to(device=device) if name=='tetrahedra' else value.to(device=device,dtype=dtype))
                for name,value in self.__dict__.items()}
        return type(self)(**fields)

    def transport(self, vertices, *, validate=True):
        """Exact piecewise-affine compatible transport; vertices[..., N, 3].

        Arbitrary leading axes support batches and trajectories on CPU or GPU.
        Validation synchronizes device predicates but preserves autograd graphs.
        """
        if vertices.dtype!=self.rest_vertices.dtype or vertices.device!=self.rest_vertices.device:
            raise ValueError('vertices and reference must share dtype/device')
        if vertices.ndim<2 or vertices.shape[-2:]!=self.rest_vertices.shape:
            raise ValueError('vertices must end in reference [N,3] shape')
        if validate:_floating(vertices,'vertices')
        corners=vertices[...,self.tetrahedra,:]
        edges=(corners[...,1:,:]-corners[...,:1,:]).transpose(-1,-2)
        F=edges@self.inverse_reference_edges
        J=torch.linalg.det(F)
        if validate and (not bool(torch.isfinite(J).all()) or bool((J<=0).any())):
            raise ValueError('Positive finite deformation Jacobians required')
        covariance=F@self.reference_covariances@F.transpose(-1,-2)
        return MaterialGaussianState(corners.mean(-2),F,covariance,J,
            J*self.reference_volumes,self.reference_density/J,self.masses)


@dataclass(frozen=True)
class SurfaceAppearanceAttachments:
    """Fixed face-centroid appearance attached to the SAME moving nodal field."""
    faces: torch.Tensor                    # [A,3], vertex IDs
    attributes: torch.Tensor               # [A,C], e.g. linear RGB/features

    @classmethod
    def from_surface_faces(cls, faces, attributes):
        _floating(attributes,'attributes')
        ids=_indices(faces,attributes.device,3,'surface_faces')
        if attributes.ndim!=2 or attributes.shape[0]!=len(ids):
            raise ValueError('One attribute vector per surface face required')
        if bool((ids<0).any()):raise ValueError('Negative surface vertex index')
        return cls(ids,attributes)

    def transport(self, vertices):
        _floating(vertices,'vertices')
        if vertices.ndim<2 or vertices.shape[-1]!=3 or bool((self.faces>=vertices.shape[-2]).any()):
            raise ValueError('Valid [...,N,3] nodal positions required')
        if vertices.device!=self.attributes.device or vertices.dtype!=self.attributes.dtype:
            raise ValueError('appearance and nodal field must share dtype/device')
        return vertices[...,self.faces,:].mean(-2),self.attributes
