"""NEW corrected reference E/g: exact bulkKeast24 + root64 compliant plane.

This explicitly makes TWO quadrature corrections relative to the old Q3/Duffy
contact operator. Frozen bulk/checkJ kernels and root64 surface are reused,
without old Duffy launches/subtraction. All141 guard points, physical geometry,
full mass and body force are retained. No integrator or production Hessian.
"""
import numpy as np
import warp as wp
from experiments.campaign_static_p2_block import shape_gradients
from experiments.campaign_p2_keast_quadrature import positive_keast24
from experiments.campaign_p2_exact_quadrature import exact_degree6_rule
from experiments.campaign_p2_warp_rhs import bulk,check_jacobian
from experiments.campaign_p2_warp_root_contact import WarpRootContact,root_surface
from experiments.campaign_p2_torch_contact import boundary_trace


def corrected_arrays(model):
 b,w=positive_keast24();old80,_=exact_degree6_rule();checks=np.concatenate([model.check_bary,model.bary,old80,b])
 if len(model.bary)!=27 or len(model.check_bary)!=10 or checks.shape!=(141,4):raise ValueError('ExplicitoldQ3+10/new80+24 guard lineage required')
 return dict(rest=np.asarray(model.rest).copy(),elements=np.asarray(model.elements).copy(),grad=np.stack([shape_gradients(x[None],model.linear_gradients) for x in b]),check_grad=np.stack([shape_gradients(x[None],model.linear_gradients) for x in checks]),weights=w[:,None]*model.volumes[None],bary=b,guard_bary=checks)

class WarpReferenceForce:
    def __init__(self,model,*,kappa_A=1e9,floor_z=0.,device='cpu'):
        if not np.isfinite([model.young,model.poisson,model.minimum_J]).all() or model.young<=0 or not -1<model.poisson<.5:raise ValueError('Finitepositive physical material required')
        self.device=wp.get_device(device);self.arrays=corrected_arrays(model);a=self.arrays;faces,areas=boundary_trace(model);self.surface=WarpRootContact(rest=a['rest'],surface_elements=faces,reference_areas=areas,kappa_A=kappa_A,floor_z=floor_z,outer_order=64,device=self.device)
        self.u=self.surface.u;self.gradient=self.surface.gradient;self.n=len(a['rest']);self.minimum_J=model.minimum_J;self.mu=model.young/(2*(1+model.poisson));self.kappa=model.young*model.poisson/((1+model.poisson)*(1-2*model.poisson))+self.mu
        self.data={k:wp.array(np.ascontiguousarray(a[k]),dtype=wp.int64 if k=='elements' else wp.float64,device=self.device) for k in ['elements','grad','check_grad','weights']};self.bulk_E=wp.empty(a['weights'].shape,dtype=wp.float64,device=self.device);self.J=wp.empty(a['weights'].shape,dtype=wp.float64,device=self.device);self.Jcheck=wp.empty(a['check_grad'].shape[:2],dtype=wp.float64,device=self.device)
    def evaluate(self,u):
        if not isinstance(u,np.ndarray) or u.dtype!=np.float64 or u.shape!=(self.n,3) or not np.isfinite(u).all():raise ValueError('FiniteFP64compatible displacement required')
        wp.copy(self.u,wp.array(np.ascontiguousarray(u),dtype=wp.float64,device=self.device));self.gradient.zero_();a=self.data
        wp.launch(bulk,dim=self.bulk_E.shape,inputs=[self.u,a['elements'],a['grad'],a['weights'],self.mu,self.kappa,self.bulk_E,self.J,self.gradient],device=self.device)
        wp.launch(check_jacobian,dim=self.Jcheck.shape,inputs=[self.u,a['elements'],a['check_grad'],self.Jcheck],device=self.device)
        s=self.surface;wp.launch(root_surface,dim=s.E.shape,inputs=[self.u,s.rest,s.faces,s.areas,s.t,s.w,s.kappa_A,s.floor_z,s.E,self.gradient],device=self.device)
        J=self.J.numpy();Jc=self.Jcheck.numpy();minimum=float(min(J.min(),Jc.min()))
        if not np.isfinite(J).all() or not np.isfinite(Jc).all() or minimum<=self.minimum_J:raise ValueError('All141 retained Jacobian guards failed; no repair')
        Ue=float(self.bulk_E.numpy().sum());Uc=float(s.E.numpy().sum());g=self.gradient.numpy().copy()
        if not np.isfinite([Ue,Uc]).all() or not np.isfinite(g).all():raise ValueError('Nonfinite same-law E/gradient')
        return dict(energy=Ue+Uc,elastic_energy=Ue,contact_energy=Uc,gradient=g,minimum_J=minimum,bulk_rule='Keast24degree6exact',surface_rule='root64forceE',guard_points=141,production_Hessian=False)
