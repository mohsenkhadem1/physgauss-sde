"""NEW controlled outer32 candidate; all frozen64 source stays unchanged.

Only root-split surface outer t/w/E dimensions change. The SAME root_surface
kernel, exactbulkKeast24, all141 J guards and physical parameters are reused.
No production Hessian. This operator is not admitted for trajectories here.
"""
import numpy as np
import warp as wp
from experiments.campaign_p2_warp_reference_force import WarpReferenceForce,corrected_arrays

class Root32Buffers:
    def __init__(self,rest,faces,areas,*,kappa_A,floor_z,device):
        if rest.ndim!=2 or rest.shape[1]!=3 or rest.dtype!=np.float64 or faces.ndim!=2 or faces.shape[1]!=6 or not np.issubdtype(faces.dtype,np.integer) or np.any(faces<0) or np.any(faces>=len(rest)) or areas.shape!=(len(faces),) or not np.isfinite(rest).all() or not np.isfinite(areas).all() or np.any(areas<=0):raise ValueError('Finite compatibleT6reference arrays required')
        if not np.isfinite([kappa_A,floor_z]).all() or kappa_A<=0:raise ValueError('Samepositivefinitecompliance required')
        x,w=np.polynomial.legendre.leggauss(32);array=lambda a,dtype=wp.float64:wp.array(np.ascontiguousarray(a),dtype=dtype,device=device)
        self.rest=array(rest);self.faces=array(faces,wp.int64);self.areas=array(areas);self.t=array((x+1)/2);self.w=array(w/2);self.kappa_A=kappa_A;self.floor_z=floor_z;self.u=wp.empty(rest.shape,dtype=wp.float64,device=device);self.gradient=wp.empty(rest.shape,dtype=wp.float64,device=device);self.E=wp.empty((32,len(faces)),dtype=wp.float64,device=device);self.order=32

class WarpReferenceForce32(WarpReferenceForce):
    def __init__(self,model,*,kappa_A=1e9,floor_z=0.,device='cpu'):
        from experiments.campaign_p2_torch_contact import boundary_trace
        if not np.isfinite([model.young,model.poisson,model.minimum_J]).all() or model.young<=0 or not -1<model.poisson<.5:raise ValueError('Finitepositivephysicalmaterialrequired')
        self.device=wp.get_device(device);self.arrays=corrected_arrays(model);a=self.arrays;faces,areas=boundary_trace(model);self.surface=Root32Buffers(a['rest'],faces,areas,kappa_A=kappa_A,floor_z=floor_z,device=self.device);self.u=self.surface.u;self.gradient=self.surface.gradient;self.n=len(a['rest']);self.minimum_J=model.minimum_J;self.mu=model.young/(2*(1+model.poisson));self.kappa=model.young*model.poisson/((1+model.poisson)*(1-2*model.poisson))+self.mu
        self.data={k:wp.array(np.ascontiguousarray(a[k]),dtype=wp.int64 if k=='elements' else wp.float64,device=self.device) for k in ['elements','grad','check_grad','weights']};self.bulk_E=wp.empty(a['weights'].shape,dtype=wp.float64,device=self.device);self.J=wp.empty(a['weights'].shape,dtype=wp.float64,device=self.device);self.Jcheck=wp.empty(a['check_grad'].shape[:2],dtype=wp.float64,device=self.device)
    def evaluate(self,u):
        a=super().evaluate(u);a['surface_rule']='root32forceE';return a
