"""NEW static rectangular block: exact supports, gravity and broad dead traction.

Only frozen compatible Neo-Hookean elasticity/static Newton is reused. No rigid
handles, artificial spring, mass-as-inertia time step, clipping or repair.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
try:
    from .boundary_block import block_mesh
    from .campaign_static_uniaxial_bar import PrismaticBar,face_area_weights,lame
except ImportError:
    from boundary_block import block_mesh
    from campaign_static_uniaxial_bar import PrismaticBar,face_area_weights,lame

SIZE=(.24,.08,.08)
KINDS=('left_face','right_face','both_ends','bottom_face','separated_bottom_patches')
CASES=(
    dict(id='left_broad_axial',boundary='left_face',young_Pa=5000.,gravity=0.,traction_N=1.,expected_deformation=True),
    dict(id='left_gravity_cantilever',boundary='left_face',young_Pa=300000.,gravity=9.81,traction_N=0.,expected_deformation=True),
    dict(id='right_gravity_cantilever',boundary='right_face',young_Pa=300000.,gravity=9.81,traction_N=0.,expected_deformation=True),
    dict(id='both_ends_gravity',boundary='both_ends',young_Pa=300000.,gravity=9.81,traction_N=0.,expected_deformation=False),
    dict(id='bottom_face_gravity',boundary='bottom_face',young_Pa=300000.,gravity=9.81,traction_N=0.,expected_deformation=False),
    dict(id='separated_patches_gravity',boundary='separated_bottom_patches',young_Pa=300000.,gravity=9.81,traction_N=0.,expected_deformation=False),
)


def support_nodes(points,kind):
    X=np.asarray(points);left=abs(X[:,0])<1e-13;right=abs(X[:,0]-SIZE[0])<1e-13;bottom=abs(X[:,2]+SIZE[2]/2)<1e-13
    if kind not in KINDS:raise ValueError('Unsupported static support; free fall is a dynamic problem')
    if kind=='left_face':return left
    if kind=='right_face':return right
    if kind=='both_ends':return left|right
    if kind=='bottom_face':return bottom
    # Physical patch edges align every declared x-grid (quarter-length patches).
    return bottom&((X[:,0]<=.06+1e-13)|(X[:,0]>=.18-1e-13))


class StaticSupportedBlock(PrismaticBar):
    def __init__(self,cells=(4,2,2),*,boundary='left_face',young=300000.,poisson=.35,density=625.,gravity=9.81,traction=0.,rotation=None,translation=None):
        lame(young,poisson)
        if density<=0 or not all(np.isfinite(v) for v in [density,gravity,traction]):raise ValueError('Finite positive density/known loads required')
        if len(cells)!=3 or cells[0]%4 or any(type(n) is not int or n<2 or n%2 for n in cells):raise ValueError('Declared structured support-conforming cells required')
        Q=np.eye(3) if rotation is None else np.asarray(rotation,dtype=float);shift=np.zeros(3) if translation is None else np.asarray(translation,dtype=float)
        if Q.shape!=(3,3) or shift.shape!=(3,) or not np.isfinite(Q).all() or not np.isfinite(shift).all() or np.max(abs(Q.T@Q-np.eye(3)))>1e-13 or abs(np.linalg.det(Q)-1)>1e-13:raise ValueError('Proper finite observer frame required')
        self.rotation,self.translation=Q,shift;self.reference_local,self.tets,self.faces=block_mesh(SIZE,cells,(SIZE[0]/2,0,0))
        self.rest=self.reference_local@Q.T+shift;self.young,self.poisson,self.density=float(young),float(poisson),float(density);self.minimum_J=.1
        corners=self.rest[self.tets];edges=np.swapaxes(corners[:,1:]-corners[:,:1],-1,-2);det=np.linalg.det(edges)
        if np.any(det<=0):raise ValueError('Positive authored tetrahedra required')
        self.volumes=det/6;inverse=np.linalg.inv(edges);self.gradients=np.concatenate([-inverse.sum(1)[:,None],inverse],axis=1)
        dofs=(self.tets[:,:,None]*3+np.arange(3)).reshape(-1,12);self.rows=np.repeat(dofs,12,axis=1).ravel();self.cols=np.tile(dofs,(1,12)).ravel()
        self.node_mass=np.zeros(len(self.rest));np.add.at(self.node_mass,self.tets,self.density*self.volumes[:,None]*np.ones((1,4))/4)
        self.load=np.zeros_like(self.rest);self.load[:,2]=-gravity*self.node_mass
        area,_=face_area_weights(self.reference_local,self.faces,0,SIZE[0]);self.traction_area=area
        self.load[:,0]+=traction*area/(SIZE[1]*SIZE[2]);self.force=float(np.linalg.norm(self.load.sum(0)))
        self.mask=np.repeat(support_nodes(self.reference_local,boundary)[:,None],3,axis=1);self.free=np.flatnonzero(~self.mask.ravel());self.boundary=boundary
        self.T=sparse.kron(sparse.eye(len(self.rest),format='csr'),Q,format='csr')

    def evaluate(self,u,tangent=False):
        u=np.asarray(u,dtype=float)
        if u.shape!=self.rest.shape or not np.isfinite(u).all():raise ValueError('Finite local displacement required')
        positions=self.rest+u@self.rotation.T;elastic,grad,H,J=self.elastic(positions,tangent)
        residual=grad@self.rotation-self.load;work=float(np.sum(self.load*u))
        return dict(potential_J=float(elastic-work),elastic_J=elastic,spring_J=0.,work_J=work,residual=residual,
                    tangent=None if H is None else self.T.T@H@self.T,J=J,positions=positions)

    def bulk_nonrigid_RMS(self,positions):
        """Exact material-volume P1 quadratic norm, not node-count-dependent RMS."""
        X=self.rest;Y=np.asarray(positions);m=self.density*self.volumes;total=m.sum()
        cx=np.einsum('n,ni->i',self.node_mass,X)/total;cy=np.einsum('n,ni->i',self.node_mass,Y)/total
        a=(X-cx)[self.tets];b=(Y-cy)[self.tets]
        C=(np.einsum('e,eai,eaj->ij',m,a,b)+np.einsum('e,ei,ej->ij',m,a.sum(1),b.sum(1)))/20
        U,_,V=np.linalg.svd(C);fix=np.eye(3);fix[2,2]=np.linalg.det(U@V);Q=U@fix@V
        e=Y-((X-cx)@Q+cy);ec=e[self.tets]
        squared=np.sum(m*(np.sum(ec**2,axis=(1,2))+np.sum(ec.sum(1)**2,axis=1)))/20
        return float(np.sqrt(squared/(3*total)))

    def diagnostics(self,answer):
        q=answer['positions'];u=answer['displacement_local_m'];_,_,_,J=self.elastic(q)
        F=np.eye(3)+np.einsum('eai,eaJ->eiJ',(q-self.rest)[self.tets],self.gradients);strain=np.linalg.svd(F,compute_uv=False)-1
        tet_peak=np.max(abs(strain),axis=1);order=np.argsort(tet_peak,kind='stable');cumulative=np.cumsum(self.volumes[order])/self.volumes.sum()
        percentile99=float(tet_peak[order[min(np.searchsorted(cumulative,.99,side='left'),len(order)-1)]])
        strain_rms=float(np.sqrt(np.sum(self.volumes[:,None]*strain**2)/(3*self.volumes.sum())))
        reaction=np.where(self.mask,answer['residual'],0.)@self.rotation.T;load_world=self.load@self.rotation.T
        balance=(reaction+load_world).sum(0);moment=np.cross(q,reaction+load_world).sum(0)
        return dict(minimum_J=float(J.min()),maximum_absolute_principal_stretch_strain=float(np.max(abs(strain))),volume_weighted99th_percentile_tet_max_abs_principal_strain=percentile99,volume_weighted_principal_strain_RMS=strain_rms,bulk_material_volume_nonrigid_coordinate_RMS_m=self.bulk_nonrigid_RMS(q),
            maximum_displacement_m=float(np.max(np.linalg.norm(u,axis=1))),maximum_constraint_error_m=float(np.max(abs(u[self.mask]))),
            original_free_residual_N=float(np.linalg.norm(answer['residual'].ravel()[self.free])),residual_tolerance_N=answer['tolerance_N'],reaction_resultant_world_N=reaction.sum(0).tolist(),
            applied_resultant_world_N=load_world.sum(0).tolist(),force_balance_defect_N=float(np.linalg.norm(balance)),actual_moment_balance_defect_Nm=float(np.linalg.norm(moment)),
            total_solid_mass_kg=float(self.node_mass.sum()),iterations=answer['iterations'],elastic_energy_J=answer['elastic_J'],external_work_J=answer['work_J'])
