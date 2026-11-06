"""3D linear-elastic Gaussian material/interaction reference.

SI units; xyz coordinates with z vertical. Quasistatic, small strain, fixed
Poisson ratio. Boundary springs are bilateral compliant supports, not unilateral
contact. No rigid-mode pin, solver jitter, or hidden regularizing force is used.
The only anchoring is the declared positive Gaussian bottom support.
"""
from __future__ import annotations
from itertools import permutations
import numpy as np
import torch

PARAMETERS = ('log_E_Pa', 'material_contrast', 'support_x_m',
              'log_support_N_per_m', 'load_x_m')


def _mesh(shape, cells):
    nx, ny, nz = map(int, cells)
    if min(nx, ny, nz) < 1:
        raise ValueError('At least one cell per axis is required')
    if shape not in ('block', 'step'):
        raise ValueError("shape must be 'block' or 'step'")
    if shape == 'step' and (nx % 2 or nz % 2):
        raise ValueError('Step geometry requires even nx and nz')
    axes = [np.linspace(-.06, .06, nx+1), np.linspace(-.03, .03, ny+1),
            np.linspace(0., .06, nz+1)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    ids = np.arange(len(points)).reshape(nx+1, ny+1, nz+1)
    tet = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if shape == 'step' and i >= nx//2 and k >= nz//2:
                    continue
                origin = np.array([i,j,k])
                for order in permutations(range(3)):
                    v = [origin.copy()]
                    for axis in order:
                        v.append(v[-1].copy()); v[-1][axis] += 1
                    tet.append([ids[tuple(q)] for q in v])
    tet = np.asarray(tet, np.int64)
    used, inverse = np.unique(tet, return_inverse=True)
    points = points[used]; tet = inverse.reshape(-1,4)
    # Signed volume is immaterial to gradients but orient all tetrahedra alike.
    dm = np.stack([points[tet[:,a]]-points[tet[:,0]] for a in (1,2,3)], -1)
    negative = np.linalg.det(dm) < 0
    temp=tet[negative,1].copy();tet[negative,1]=tet[negative,2];tet[negative,2]=temp
    return points,tet


class LinearGaussianBody:
    """Batch differentiable Gaussian-field linear FEM on a regular tetra mesh.

    ``theta`` has columns PARAMETERS in physical/log physical units; no sigmoid
    or undisclosed clipping is applied. ``solve`` returns unit-load responses;
    multiplying them by forces in N gives displacement in metres. Gravity is
    deliberately absent from these reset incremental-load experiments.
    """
    def __init__(self, shape='block', cells=(4,2,2), device='cpu', dtype=torch.float64,
                 poisson=.30, density=500., footprint_width=.018):
        if not -1 < poisson < .5:
            raise ValueError('Poisson ratio must be in (-1,.5)')
        if footprint_width <= 0 or density <= 0:
            raise ValueError('Positive width and density required')
        self.shape=shape; self.cells=tuple(cells); self.dtype=dtype; self.device=torch.device(device)
        self.poisson=float(poisson);self.density=float(density);self.footprint_width=float(footprint_width)
        p,tet=_mesh(shape,cells)
        self.points=torch.as_tensor(p,dtype=dtype,device=device)
        self.rest=self.points
        self.elements=torch.as_tensor(tet,dtype=torch.long,device=device)
        self.n=len(p);self.ndof=3*self.n
        dm=torch.stack([self.points[self.elements[:,a]]-self.points[self.elements[:,0]] for a in (1,2,3)],-1)
        inv=torch.linalg.inv(dm)
        self.grad=torch.cat([-inv.sum(1,keepdim=True),inv],1)
        self.volumes=torch.linalg.det(dm)/6
        self.centers=self.points[self.elements].mean(1)
        self.mass=self.points.new_zeros(self.n).index_add(0,self.elements.flatten(),
            (self.volumes[:,None].expand(-1,4)*density/4).flatten())
        eye=torch.eye(3,dtype=dtype,device=device)
        lam=poisson/((1+poisson)*(1-2*poisson));mu=1/(2*(1+poisson))
        g=self.grad
        local=lam*torch.einsum('eai,ebj->eaibj',g,g)
        local+=mu*torch.einsum('eak,ebk,ij->eaibj',g,g,eye)
        local+=mu*torch.einsum('eaj,ebi->eaibj',g,g)
        self.local_unit_stiffness=(self.volumes[:,None,None,None,None]*local).reshape(-1,12,12)
        dofs=(self.elements[:,:,None]*3+torch.arange(3,device=device)).reshape(-1,12)
        self._assembly_index=(dofs[:,:,None]*self.ndof+dofs[:,None,:]).flatten()
        # Fixed two-kernel material partition, volume centered in log modulus.
        centers=self.points.new_tensor([[-.035,0.,.03],[.035,0.,.03]])
        feature=torch.exp(-.5*(((self.centers[:,None]-centers[None])/
                            self.points.new_tensor([.035,.06,.06]))**2).sum(-1))
        feature=feature/feature.sum(-1,keepdim=True)
        contrast=feature[:,1]-feature[:,0]
        self.material_features=feature
        self.material_log_center=(contrast*self.volumes).sum()/self.volumes.sum()
        self.material_contrast_feature=contrast-self.material_log_center
        self._surface(tet)
        self.probe_specs=[dict(x=x,mode=mode) for x in (-.04,0.,.04) for mode in ('top','side')]
        probe=[]
        for spec in self.probe_specs:
            weight=self.footprint(self.points.new_tensor([spec['x']]),spec['mode'])[0]
            direction=self.points.new_tensor([0.,0.,-1.] if spec['mode']=='top' else [0.,-1.,0.])
            probe.append((weight[:,None]*direction).flatten())
        self.probe_forces=torch.stack(probe,-1)

    def _surface(self,tet):
        faces={}
        for ei,elem in enumerate(tet):
            for face in ((elem[1],elem[2],elem[3]),(elem[0],elem[3],elem[2]),
                         (elem[0],elem[1],elem[3]),(elem[0],elem[2],elem[1])):
                key=tuple(sorted(face))
                if key in faces:faces[key]=None
                else:faces[key]=(face,ei)
        entries=[v for v in faces.values() if v is not None]
        tri=torch.as_tensor([v[0] for v in entries],dtype=torch.long,device=self.device)
        owners=torch.as_tensor([v[1] for v in entries],dtype=torch.long,device=self.device)
        xyz=self.points[tri];normal=torch.linalg.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        area=normal.norm(dim=-1)/2;normal=normal/(2*area[:,None])
        centers=xyz.mean(1)
        normal=torch.where(((centers-self.centers[owners])*normal).sum(-1)[:,None]<0,-normal,normal)
        self.surface_triangles=tri;self.surface_centers=centers;self.surface_areas=area;self.surface_normals=normal
        # Three degree-two triangle quadrature nodes, each with area/3.
        bary=self.points.new_tensor([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
        self.surface_points=torch.einsum('qa,fad->fqd',bary,xyz).reshape(-1,3)
        self.surface_quadrature_weights=(area[:,None].expand(-1,3)/3).flatten()
        N=self.points.new_zeros(len(tri)*3,self.n)
        for a in range(3):
            N.scatter_add_(1,tri[:,a,None].expand(-1,3).reshape(-1,1),bary[:,a].repeat(len(tri))[:,None])
        self.surface_interpolation=N
        self._surface_masks={
            'bottom':(normal[:,2]<-.99).repeat_interleave(3),
            'top':(normal[:,2]>.99).repeat_interleave(3),
            'side':(normal[:,1]>.99).repeat_interleave(3)}
        self.surface_lumped_area=self.surface_quadrature_weights@N

    def footprint(self,x,face):
        """Normalized nodal quadrature weights; sum=1, independent of mesh area.

        Footprints vary in x and are broad in the other surface direction. The
        top footprint includes both horizontal terraces of the stepped body.
        """
        x=torch.as_tensor(x,dtype=self.dtype,device=self.device).reshape(-1)
        q=self.surface_points
        gate=torch.exp(-.5*((q[None,:,0]-x[:,None])/self.footprint_width)**2)
        gate=gate*self._surface_masks[face][None]*self.surface_quadrature_weights[None]
        denominator=gate.sum(-1,keepdim=True)
        if bool((denominator<=0).any()):raise ValueError('Empty/underflowed surface footprint')
        return (gate/denominator)@self.surface_interpolation

    def material_modulus(self,theta):
        return torch.exp(theta[:,0,None]+theta[:,1,None]*self.material_contrast_feature[None])

    def operators(self,theta):
        theta=torch.as_tensor(theta,dtype=self.dtype,device=self.device)
        if theta.ndim==1:theta=theta[None]
        if theta.ndim!=2 or theta.shape[1]!=5 or not bool(torch.isfinite(theta).all()):
            raise ValueError('Expected finite theta [batch,5] in declared physical/log units')
        E=self.material_modulus(theta)
        values=(E[:,:,None,None]*self.local_unit_stiffness[None]).flatten(1)
        elastic=self.points.new_zeros(len(theta),self.ndof*self.ndof).index_add(1,self._assembly_index,values)
        elastic=elastic.reshape(-1,self.ndof,self.ndof)
        support=self.footprint(theta[:,2],'bottom')
        diagonal=(theta[:,3].exp()[:,None]*support).repeat_interleave(3,dim=1)
        stiffness=elastic+torch.diag_embed(diagonal)
        top=self.footprint(theta[:,4],'top');side=self.footprint(theta[:,4],'side')
        forces=torch.stack([(top[:,:,None]*self.points.new_tensor([0.,0.,-1.])).flatten(1),
                            (side[:,:,None]*self.points.new_tensor([0.,-1.,0.])).flatten(1)],-1)
        return dict(stiffness=stiffness,elastic_stiffness=elastic,unknown_forces=forces,
                    probe_forces=self.probe_forces,element_modulus=E,support_weights=support,
                    load_weights=torch.stack([top,side],1))

    def custom_probe_forces(self,specs):
        """Force columns for [{'x': metres, 'mode': 'top'|'side'}, ...]."""
        columns=[]
        for spec in specs:
            if spec['mode'] not in ('top','side'):
                raise ValueError('Probe mode must be top or side')
            weight=self.footprint(self.points.new_tensor([spec['x']]),spec['mode'])[0]
            direction=self.points.new_tensor([0.,0.,-1.] if spec['mode']=='top' else [0.,-1.,0.])
            columns.append((weight[:,None]*direction).flatten())
        return torch.stack(columns,-1)

    def modulus_at_points(self,theta,points):
        """Same continuous two-Gaussian log material field at reference points."""
        theta=torch.as_tensor(theta,dtype=self.dtype,device=self.device)
        if theta.ndim==1:theta=theta[None]
        points=torch.as_tensor(points,dtype=self.dtype,device=self.device).reshape(-1,3)
        centers=self.points.new_tensor([[-.035,0.,.03],[.035,0.,.03]])
        feature=torch.exp(-.5*(((points[:,None]-centers[None])/
                            self.points.new_tensor([.035,.06,.06]))**2).sum(-1))
        feature=feature/feature.sum(-1,keepdim=True)
        contrast=feature[:,1]-feature[:,0]-self.material_log_center
        return torch.exp(theta[:,0,None]+theta[:,1,None]*contrast[None])

    def solve(self,theta,include_operators=False,probe_specs=None):
        """Solve both unknown loads and default/custom calibrated probes together."""
        op=self.operators(theta)
        probe_forces=self.probe_forces if probe_specs is None else self.custom_probe_forces(probe_specs)
        force=torch.cat([op['unknown_forces'],probe_forces[None].expand(len(op['stiffness']),-1,-1)],-1)
        chol=torch.linalg.cholesky(op['stiffness'])
        response=torch.cholesky_solve(force,chol)
        result=dict(unit_response=response[:,:,:2].reshape(-1,self.n,3,2),
                    probe_response=response[:,:,2:].reshape(-1,self.n,3,probe_forces.shape[-1]))
        if include_operators:
            result.update(op);result['probe_forces']=probe_forces
        return result

    def interpolation(self,points,tolerance=1e-8):
        """P1 barycentric [point,node] map; rejects points outside the volume."""
        points=torch.as_tensor(points,dtype=self.dtype,device=self.device).reshape(-1,3)
        delta=points[:,None]-self.points[self.elements[:,0]][None]
        # barycentric coordinates of vertices1:3 are grad_i dot (p-x0).
        tail=torch.einsum('pei,eai->pea',delta,self.grad[:,1:])
        bary=torch.cat([1-tail.sum(-1,keepdim=True),tail],-1)
        inside=(bary>=-tolerance).all(-1)&(bary<=1+tolerance).all(-1)
        if not bool(inside.any(-1).all()):
            raise ValueError('Interpolation point lies outside the tetrahedral volume')
        owner=inside.to(torch.int64).argmax(-1)
        result=self.points.new_zeros(len(points),self.n)
        result.scatter_add_(1,self.elements[owner],bary[torch.arange(len(points),device=self.device),owner])
        return result

    def point_response(self,response,points):
        """Interpolate response [...,node,xyz,load] at fixed reference points."""
        return torch.einsum('pn,...ncl->...pcl',self.interpolation(points),response)

    def strain(self,displacement):
        """Infinitesimal strain [...,element,3,3] from [...,node,3]."""
        gradient=torch.einsum('...eai,eaj->...eij',displacement[...,self.elements,:],self.grad)
        return .5*(gradient+gradient.transpose(-1,-2))

    def metadata(self):
        return dict(shape=self.shape,cells=list(self.cells),nodes=self.n,elements=len(self.elements),
                    extent_m=[.12,.06,.06],volume_m3=float(self.volumes.sum()),poisson=self.poisson,
                    density_kg_m3=self.density,parameters=list(PARAMETERS),
                    material='Two fixed Gaussian material kernels; volume-centered log-modulus contrast.',
                    support='Bilateral full-xyz bottom Gaussian spring; normalized surface weights, total N/m.',
                    loads='Two unknown total-N amplitudes: top -z and side -y; shared inferred x footprint.',
                    footprint_width_m=self.footprint_width,probe_specs=self.probe_specs,
                    scope='Known geometry, quasistatic small-strain incremental response; no gravity/contact/inertia.',
                    hidden_regularization='None; declared bottom support removes rigid null modes.')
