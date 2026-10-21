"""Batched differentiable supplied-material queries on a fixed enriched basis.

Physical coordinate order is E[Pa], rho[kg/m3], nu[1], body drag[1/s].
Complete known support geometry/law/hardware and known commands stay fixed.
No observation, posterior, finite simulation or constructor per particle.
"""
from __future__ import annotations
import numpy as np
import torch


class BatchedModalMaterialForward:
    def __init__(self,enriched_modal):
        self.modal=enriched_modal;self._cache={}

    def _prepare(self,clock,device):
        try:from campaign_hypothesis_dynamics import validate_clock
        except ImportError:from experiments.campaign_hypothesis_dynamics import validate_clock
        key=(clock.public_manifest()['manifest_sha256'],str(device))
        if key in self._cache:return self._cache[key]
        modal=self.modal;refined=validate_clock(clock,120,len(modal.full.points),len(modal.full.handle_dofs));h=1/120
        rows=[];d=modal.lift(0.);dv=torch.zeros(3,dtype=torch.float64);t=0.
        if float(d.abs().max())>1e-12:raise ValueError('Initial rest-compatible support required')
        lifts=[d.clone()]
        for i in range(360):
            new_t=t+h;new_d=modal.lift(new_t);db=new_d-d
            load=modal.G@torch.tensor(refined.handle_forces_world_N[i].copy()).flatten()
            nodal=refined.nodal_forces_world_N[i]
            if len(nodal):load+=modal.AV.T@torch.tensor(nodal.copy()).flatten()
            rows.append((load,modal.M_lift@(dv/h-db/h**2),modal.M_lift@(-db/h),
                modal.Ks_lift@new_d,modal.Kv_lift@new_d))
            dv=db/h;d=new_d;t=new_t;lifts.append(d.clone())
        buffers={name:getattr(modal,name).detach().to(device=device,dtype=torch.float64) for name in ['Mb','Mh','Ks','Kv','AV','S']}
        for j,name in enumerate(['known_load','lift_inertia','lift_drag','lift_shear','lift_volume']):
            buffers[name]=torch.stack([row[j] for row in rows]).to(device)
        buffers['lifts']=torch.stack(lifts).to(device);buffers['rest']=modal.full.points.detach().to(device)
        handles=modal.full.handle_dofs
        translation=[i for dofs in handles for i in dofs[:3]];rotation=[i for dofs in handles for i in dofs[3:]]
        buffers['center_map']=modal.basis[translation].detach().to(device)
        buffers['rotation_map']=modal.basis[rotation].detach().to(device)
        buffers['origins']=modal.full.points[modal.full.model.origins].detach().to(device)
        buffers['output_indices']=set(refined.output_step_indices.tolist());buffers['handle_count']=len(handles)
        self._cache[key]=buffers;return buffers

    def predict(self,clock,physical_coordinates,device='cpu'):
        """Return float64 positions[B,61,N,3], retaining material autograd."""
        if not isinstance(physical_coordinates,torch.Tensor) or physical_coordinates.dtype!=torch.float64:
            raise TypeError('Explicit float64 torch physical coordinates[B,4] required')
        if physical_coordinates.ndim!=2 or physical_coordinates.shape[1]!=4 or not len(physical_coordinates):
            raise ValueError('Nonempty batch[B,4] in E/rho/nu/drag order required')
        theta=physical_coordinates.to(device);E,rho,nu,drag=theta.unbind(1)
        if not bool(torch.isfinite(theta).all() and (E>0).all() and (rho>0).all() and (nu>-1).all() and (nu<.5).all() and (drag>=0).all()):
            raise ValueError('Valid finite physical material hypotheses required')
        b=self._prepare(clock,device);mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));h=1/120
        M=rho[:,None,None]*b['Mb']+b['Mh'];C=(drag*rho)[:,None,None]*b['Mb']
        K=mu[:,None,None]*b['Ks']+lam[:,None,None]*b['Kv'];effective=M/h**2+C/h+K
        factor=torch.linalg.cholesky((effective+effective.transpose(-1,-2))/2)
        z=theta.new_zeros((len(theta),self.modal.mode_count));v=torch.zeros_like(z)
        positions=[];centers=[];rotations=[]
        def capture(index):
            positions.append((z@b['AV'].T+b['lifts'][index]@b['S'].T).reshape(len(theta),-1,3)+b['rest'])
            centers.append((z@b['center_map'].T).reshape(len(theta),b['handle_count'],3)+b['origins'])
            rotations.append((z@b['rotation_map'].T).reshape(len(theta),b['handle_count'],3))
        capture(0)
        for i in range(360):
            rhs=b['known_load'][i]+torch.bmm(M,(z/h**2+v/h).unsqueeze(-1)).squeeze(-1)+torch.bmm(C,(z/h).unsqueeze(-1)).squeeze(-1)
            rhs+=rho[:,None]*(b['lift_inertia'][i]+drag[:,None]*b['lift_drag'][i])-mu[:,None]*b['lift_shear'][i]-lam[:,None]*b['lift_volume'][i]
            new_z=torch.cholesky_solve(rhs.unsqueeze(-1),factor).squeeze(-1)
            v=(new_z-z)/h;z=new_z
            if i+1 in b['output_indices']:capture(i+1)
        return {'positions_m':torch.stack(positions,1),'handle_centers_m':torch.stack(centers,1),
            'handle_rotation_coordinates_rad':torch.stack(rotations,1),
            'times_s':torch.tensor(clock.output_times_s.copy(),device=device),
            'role':'fixed_known_support_supplied_material_linear_modal_not_posterior'}
