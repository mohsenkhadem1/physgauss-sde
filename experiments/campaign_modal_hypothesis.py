"""Fixed rest-modal small-strain mechanics for known hardware/support geometry.

The96 lowest nominal generalized eigenmodes are fixed without trajectory data.
Material dependence remains explicit in projected mass/drag/shear/volume terms.
This is an approximation to the full linear baseline, never a finite reference
or posterior. No nonlinear dynamics step or response-based basis choice occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch


def constitutive_parts(physical):
    """Independent rest element assembly K=mu*Ks+lambda*Kv, no repairs."""
    gradients=physical.gradients;volume=physical.volumes
    ids=torch.as_tensor(physical.tets,dtype=torch.long);dofs=(3*ids[:,:,None]+torch.arange(3)).reshape(len(ids),12)
    B=torch.zeros((len(ids),6,12),dtype=torch.float64)
    for j in range(4):
        dx,dy,dz=gradients[:,j].unbind(-1)
        B[:,0,3*j]=dx;B[:,1,3*j+1]=dy;B[:,2,3*j+2]=dz
        B[:,3,3*j+1]=dz;B[:,3,3*j+2]=dy
        B[:,4,3*j]=dz;B[:,4,3*j+2]=dx
        B[:,5,3*j]=dy;B[:,5,3*j+1]=dx
    diagonal=torch.tensor([2.,2.,2.,1.,1.,1.],dtype=torch.float64)
    local_s=B.transpose(1,2)@(diagonal[None,:,None]*B)*volume[:,None,None]
    divergence=gradients.reshape(len(ids),12)
    local_v=divergence[:,:,None]*divergence[:,None,:]*volume[:,None,None]
    n=physical.points.numel();index=dofs[:,:,None]*n+dofs[:,None,:];parts=[]
    for local in [local_s,local_v]:
        result=torch.zeros((n,n),dtype=torch.float64)
        result.view(-1).index_add_(0,index.flatten(),local.flatten());parts.append(result)
    return parts


class FixedModalLinearDynamics:
    def __init__(self,full,mode_count=96):
        if type(mode_count) is not int or not 1<=mode_count<=full.size:raise ValueError('Fixed valid mode count required')
        self.full=full;self.mode_count=mode_count;physical=full.model
        Ks,Kv=constitutive_parts(physical);A=full.A
        self.Ks_full=Ks;self.Kv_full=Kv
        Kr_s=A.T@Ks@A;Kr_v=A.T@Kv@A
        Mb_unit=full.Mbody/physical.density
        M0=625.*Mb_unit+full.Maux
        mu0=5000./(2*1.35);lam0=5000.*.35/(1.35*.3)
        K0=mu0*Kr_s+lam0*Kr_v
        L=torch.linalg.cholesky(M0)
        left=torch.linalg.solve_triangular(L,K0,upper=False)
        whitened=torch.linalg.solve_triangular(L,left.T,upper=False).T
        values,U=torch.linalg.eigh((whitened+whitened.T)/2)
        if float(values.min()) < -1e-10*max(1.,float(values.abs().max())):raise ValueError('Nominal tangent has substantive negative eigenvalue; no repair')
        V=torch.linalg.solve_triangular(L.T,U[:,:mode_count],upper=True)
        self.basis=V
        self.Mb=V.T@Mb_unit@V;self.Mh=V.T@full.Maux@V
        self.Ks=V.T@Kr_s@V;self.Kv=V.T@Kr_v@V
        self.AV=A@V;self.JAV=full.JA@V
        S=torch.zeros((physical.points.numel(),3),dtype=torch.float64)
        for node in physical.fixture_nodes.tolist():S[3*node:3*node+3]=torch.eye(3,dtype=torch.float64)
        self.S=S;unit_mass=physical.body_mass.repeat_interleave(3)/physical.density
        self.M_lift=V.T@A.T@(unit_mass[:,None]*S)
        self.Ks_lift=V.T@A.T@Ks@S;self.Kv_lift=V.T@A.T@Kv@S
        G=torch.zeros((full.size,3*len(full.handle_dofs)),dtype=torch.float64)
        for j,dofs in enumerate(full.handle_dofs):G[dofs[:3],3*j:3*j+3]=torch.eye(3,dtype=torch.float64)
        self.G=V.T@G
        orth=float((V.T@M0@V-torch.eye(mode_count,dtype=torch.float64)).abs().max())
        if orth>1e-10:raise ValueError('Nominal mass orthonormality failed')
        zero=self.rigid_zero_modes();zero_error=0.
        if zero.shape[1]:
            projection=V@(V.T@M0@zero)
            for q,p in zip(zero.T,projection.T):zero_error=max(zero_error,float(torch.linalg.vector_norm(q-p)/torch.linalg.vector_norm(q)))
            if zero_error>1e-6:raise ValueError('Fixed basis failed to preserve known rigid zero modes')
        self.basis_diagnostics=dict(modes=mode_count,nominal_E_Pa=5000.,nominal_density_kg_m3=625.,nominal_nu=.35,
            minimum_nominal_eigenvalue=float(values.min()),maximum_retained_eigenvalue=float(values[mode_count-1]),
            maximum_nominal_eigenvalue=float(values.max()),M0_orthonormal_max_error=orth,
            known_rigid_zero_mode_count=zero.shape[1],maximum_rigid_zero_mode_relative_projection_error=zero_error,
            solver='full float64 mass-whitened symmetric eigensolve; lowest predeclared modes; no eigenvalue clipping',
            source_response_inputs_used=False)

    def rigid_zero_modes(self):
        """Exact global infinitesimal rigid motions allowed by these fixtures."""
        p=self.full.model;kind=p.boundary_kind
        if kind not in ['free','point_pin']:return torch.empty((self.full.size,0),dtype=torch.float64)
        center=p.points[p.fixture_nodes[0]] if kind=='point_pin' else p.points.mean(0)
        result=[]
        for omega in torch.eye(3,dtype=torch.float64):
            q=torch.zeros(self.full.size,dtype=torch.float64)
            q[:3*len(p.free_nodes)]=torch.linalg.cross(omega.expand(len(p.free_nodes),3),p.points[p.free_nodes]-center).flatten()
            for origin,dofs in zip(p.origins,p.handle_dofs):q[dofs[:3]]=torch.linalg.cross(omega,p.points[origin]-center);q[dofs[3:]]=omega
            result.append(q)
        if kind=='free':
            for direction in torch.eye(3,dtype=torch.float64):
                q=torch.zeros(self.full.size,dtype=torch.float64);q[:3*len(p.free_nodes)]=direction.repeat(len(p.free_nodes))
                for dofs in p.handle_dofs:q[dofs[:3]]=direction
                result.append(q)
        return torch.stack(result,1)

    def lift(self,t):
        p=self.full.model
        return p.targets(t)[0]-p.points[p.fixture_nodes[0]] if len(p.fixture_nodes) else torch.zeros(3,dtype=torch.float64)

    def predict(self,clock,material,*,integration_hz=120):
        try:from campaign_hypothesis_dynamics import validate_clock
        except ImportError:from experiments.campaign_hypothesis_dynamics import validate_clock
        refined=validate_clock(clock,integration_hz,len(self.full.points),len(self.full.handle_dofs))
        E=material['youngs_modulus_Pa'];rho=material['density_kg_m3'];nu=material['poissons_ratio'];drag=material['body_linear_drag_s_inverse']
        if not (E>0 and rho>0 and -1<nu<.5 and drag>=0):raise ValueError('Valid material hypothesis required')
        mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));h=1/integration_hz
        M=rho*self.Mb+self.Mh;C=drag*rho*self.Mb;K=mu*self.Ks+lam*self.Kv
        effective=M/h**2+C/h+K;factor=torch.linalg.cholesky((effective+effective.T)/2)
        z=torch.zeros(self.mode_count,dtype=torch.float64);v=torch.zeros_like(z);t=0.;d=self.lift(t);dv=torch.zeros(3,dtype=torch.float64)
        if float(d.abs().max())>1e-12:raise ValueError('Rest-compatible lift required')
        outputs=set(refined.output_step_indices.tolist());frames=[]
        def capture():
            q=self.basis@z
            x=self.full.points+(self.AV@z+self.S@d).reshape_as(self.full.points)
            frames.append((x.clone(),(self.AV@v+self.S@dv).reshape_as(self.full.points),
                torch.stack([self.full.points[o]+q[ids[:3]] for o,ids in zip(self.full.model.origins,self.full.handle_dofs)]) if self.full.handle_dofs else torch.empty((0,3),dtype=torch.float64),
                torch.stack([q[ids[3:]] for ids in self.full.handle_dofs]) if self.full.handle_dofs else torch.empty((0,3),dtype=torch.float64),t))
        capture()
        for i in range(len(refined.parent_command_indices)):
            new_t=t+h;new_d=self.lift(new_t);db=new_d-d
            F=torch.tensor(refined.handle_forces_world_N[i].copy())
            rhs=self.G@F.flatten()+M@(z/h**2+v/h)+C@z/h
            nodal=refined.nodal_forces_world_N[i]
            if len(nodal):rhs+=self.AV.T@torch.tensor(nodal.copy()).flatten()
            rhs+=rho*self.M_lift@(dv/h-db/h**2-drag*db/h)-(mu*self.Ks_lift+lam*self.Kv_lift)@new_d
            new_z=torch.cholesky_solve(rhs[:,None],factor)[:,0]
            v=(new_z-z)/h;z=new_z;dv=db/h;d=new_d;t=new_t
            if i+1 in outputs:capture()
        if len(frames)!=61:raise ValueError('Complete61 output frames required')
        return dict(positions_m=torch.stack([f[0] for f in frames]),velocities_point_chord_m_s=torch.stack([f[1] for f in frames]),
            handle_centers_m=torch.stack([f[2] for f in frames]),handle_rotation_coordinates_rad=torch.stack([f[3] for f in frames]),
            actual_state_times_s=np.array([f[4] for f in frames]))


def main(manifest_path,linear_root,output):
    try:
        from campaign_hypothesis_inputs import known_hardware_from_public
        from campaign_linear_hypothesis import LinearHypothesisDynamics
        from campaign_exciting_commands import make_exciting_commands
        from campaign_hypothesis_dynamics import runtime_sources
    except ImportError:
        from experiments.campaign_hypothesis_inputs import known_hardware_from_public
        from experiments.campaign_linear_hypothesis import LinearHypothesisDynamics
        from experiments.campaign_exciting_commands import make_exciting_commands
        from experiments.campaign_hypothesis_dynamics import runtime_sources
    torch.set_num_threads(2);started=time.monotonic();destination=Path(output);destination.mkdir(parents=True,exist_ok=False)
    design=json.loads(Path(manifest_path).read_text());hyp={h['id']:h for h in design['hypotheses']};public={h['id']:h for h in design['public_hardware']}
    fit=[c for c in design['cells'] if c['evaluation_role']=='fit'];assert len(fit)==32
    bases={};rows=[]
    for cell in fit:
        record=hyp[cell['hypothesis_id']];spec=record['spec'];key=record['support_draw_id']
        if key not in bases:
            known=known_hardware_from_public(public[cell['public_hardware_id']]);full=LinearHypothesisDynamics.from_hypothesis(known,spec)
            bases[key]=FixedModalLinearDynamics(full,mode_count=96)
        modal=bases[key];result=modal.predict(make_exciting_commands(cell['program_id']),spec['material'])
        # Only a predeclared fit-cell archive is opened, after basis fixation.
        path=Path(linear_root)/'cells'/cell['id']/'predictions.npz'
        with np.load(path,allow_pickle=False) as truth:
            q=truth['positions_m'];v=truth['velocities_point_chord_m_s']
            err=float(np.sqrt(np.mean((result['positions_m'].numpy()-q)**2)));signal=float(np.sqrt(np.mean((q-modal.full.points.numpy())**2)))
            velocity_error=float(np.sqrt(np.mean((result['velocities_point_chord_m_s'].numpy()-v)**2)))
        dest=destination/(cell['id']+'.npz');np.savez_compressed(dest,**{k:a.numpy() if isinstance(a,torch.Tensor) else a for k,a in result.items()})
        rows.append(dict(id=cell['id'],material=record['material_tuple_id'],boundary_kind=spec['boundary_kind'],
            coordinate_RMSE_vs_full_linear_m=err,full_linear_motion_RMS_m=signal,relative_coordinate_RMSE=err/max(signal,1e-30),
            velocity_coordinate_RMSE_m_s=velocity_error,full_linear_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(json.dumps(rows[-1]),flush=True)
    report=dict(status='FIXED96_MODE_FIT32_DIAGNOSTIC_ONLY',basis_count=len(bases),basis_modes=96,fit_cases=len(rows),
        basis_diagnostics={key:model.basis_diagnostics for key,model in bases.items()},rows=rows,elapsed_s=time.monotonic()-started,
        maximum_fit_coordinate_RMSE_m=max(r['coordinate_RMSE_vs_full_linear_m'] for r in rows),
        maximum_fit_relative_coordinate_RMSE=max(r['relative_coordinate_RMSE'] for r in rows),
        evaluation64_accessed=False,mode_count_or_nominal_parameters_selected_from_responses=False,
        nonlinear_reference=False,posterior_launched=False,production_qualified=False,
        limitation='Known hardware/support geometry; fixed96-dimensional approximation to linear mechanics only. Material coefficients explicit, complete auxiliary rigid inertia retained. No finite/camera/calibration or64evaluation data used.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_physical_dependencies_sha256=runtime_sources(),
        full_linear_source_sha256=hashlib.sha256(Path(__file__).with_name('campaign_linear_hypothesis.py').read_bytes()).hexdigest(),
        input_manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest())
    (destination/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True);parser.add_argument('--linear-root',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();main(args.manifest,args.linear_root,args.output)
