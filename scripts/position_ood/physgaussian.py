"""Position-coupled official PhysGaussian MPM core (explicit benchmark adaptation).

Uses native fixed-corotated jelly kernels, with particle Robin coupling for the
shared compliant tool interface. The optional field is for oracle diagnostics;
benchmark inverse fitting never receives its simulator labels.
"""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[2];UP=ROOT/'runs/third_party/PhysGaussian-v09/mpm_solver_warp'
sys.path.insert(0,str(UP));sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
from mpm_solver_warp import MPM_Simulator_WARP
from mpm_utils import zero_grid,compute_stress_from_F_trial,p2g_apic_with_stress,grid_normalization_and_gravity,add_damping_via_grid,g2p
from warp_utils import MPMStateStruct
from scripts.phystwin.sequential_fixture import points

@wp.kernel
def robin(state:MPMStateStruct, target:wp.array(dtype=wp.vec3), stiffness:wp.array(dtype=float),dt:float):
    i=wp.tid();force=stiffness[i]*(target[i]-state.particle_x[i]);state.particle_v[i]=state.particle_v[i]+dt*force/state.particle_mass[i]

@wp.kernel
def potential(state:MPMStateStruct,volume:wp.array(dtype=float),A:float,center:wp.vec3,width:float,dt:float):
    i=wp.tid();d=state.particle_x[i]-center;force=A*volume[i]*wp.exp(-wp.dot(d,d)/(2.*width*width))*d/(width*width);state.particle_v[i]=state.particle_v[i]+dt*force/state.particle_mass[i]

@wp.kernel
def reset(state:MPMStateStruct,rest:wp.array(dtype=wp.vec3)):
    i=wp.tid();state.particle_x[i]=rest[i];state.particle_v[i]=wp.vec3(0.);state.particle_C[i]=wp.mat33(0.);state.particle_F_trial[i]=wp.identity(n=3,dtype=float);state.particle_F[i]=wp.identity(n=3,dtype=float)

class PositionMPM:
    def __init__(self,resolution=2,grid=64,dt=.0002,damping=15.,device='cuda:0'):
        self.device=device;self.dt=dt;self.shift=np.array([.15,.24,.24]);shape=(24*resolution+1,8*resolution+1,8*resolution+1)
        self.rest=np.stack(np.meshgrid(np.linspace(0,.24,shape[0]),np.linspace(-.04,.04,shape[1]),np.linspace(-.04,.04,shape[2]),indexing='ij'),-1).reshape(-1,3)
        weights=np.ones(shape)
        for axis in range(3):
            sl=[slice(None)]*3;sl[axis]=[0,-1];weights[tuple(sl)]*=.5
        self.weights=weights.ravel()/weights.sum();self.sample_ids=np.argmin(np.linalg.norm(self.rest[:,None]-points()[None],axis=-1),axis=0)
        self.sim=MPM_Simulator_WARP(len(self.rest),n_grid=grid,grid_lim=.6,device=device)
        self.sim.load_initial_data_from_torch(torch.tensor(self.rest+self.shift,dtype=torch.float32,device=device),torch.tensor(.001536*self.weights,dtype=torch.float32,device=device),n_grid=grid,grid_lim=.6,device=device)
        self.sim.set_parameters_dict(dict(E=20000.,nu=.35,density=625.,material='jelly',g=[0,0,-9.81],grid_v_damping_scale=float(np.exp(-damping*dt))),device=device)
        self.sim.finalize_mu_lam(device=device);self.rest_wp=wp.from_numpy((self.rest+self.shift).astype('float32'),dtype=wp.vec3,device=device);self.target=wp.clone(self.rest_wp);self.stiffness=wp.zeros(len(self.rest),dtype=float,device=device);self.volumes=wp.from_numpy(self.weights.astype('float32'),dtype=float,device=device);self.A=0.;self.center=self.shift+np.array([.12,0,-.09]);self.width=.08;self.grid=grid
        self.graph=None
    def parameters(self,E,Kleft,Kright,layout,material_gradient=0.,gravity=(0,0,-9.81),field_amplitude_J=0.,field_center=(.12,0,-.09),field_width=.08):
        feature=np.exp(-((self.rest[:,0]-.06)/.07)**2)-np.exp(-((self.rest[:,0]-.18)/.07)**2)
        wp.copy(self.sim.mpm_model.E,wp.from_numpy((E*np.exp(material_gradient*feature)).astype('float32'),dtype=float,device=self.device));self.sim.finalize_mu_lam(device=self.device)
        self.sim.mpm_model.gravitational_accelaration=wp.vec3(*gravity);self.A=float(field_amplitude_J);self.center=self.shift+field_center;self.width=float(field_width)
        kinds={'parallel_lower':('lower','lower'),'crossed':('lower','upper'),'opposite_y':('yminus','yplus')}[layout];k=np.zeros(len(self.rest))
        for side,(K,kind) in enumerate(zip([Kleft,Kright],kinds)):
            face=np.isclose(self.rest[:,0],.24*side);axis=2 if kind in ('lower','upper') else 1;negative=kind in ('lower','yminus');coord=self.rest[:,axis];half=(coord<=1e-9) if negative else (coord>=-1e-9);w=self.weights*face*half;w[np.isclose(coord,0)]*=.5;k+=K*w/w.sum()
        wp.copy(self.stiffness,wp.from_numpy(k.astype('float32'),dtype=float,device=self.device));self.graph=None
    def set_pose(self,pose):
        target=self.rest+self.shift
        for side in range(2):
            mask=np.isclose(self.rest[:,0],.24*side);c=np.array([.24*side,0,0]);R=Rotation.from_rotvec(pose[side,3:]).as_matrix();target[mask]=(self.rest[mask]-c)@R.T+c+pose[side,:3]+self.shift
        wp.copy(self.target,wp.from_numpy(target.astype('float32'),dtype=wp.vec3,device=self.device))
    def step(self):
        s=self.sim.mpm_state;m=self.sim.mpm_model;n=len(self.rest);d=self.device;grid=(self.grid,)*3
        wp.launch(zero_grid,dim=grid,inputs=[s,m],device=d)
        wp.launch(robin,dim=n,inputs=[s,self.target,self.stiffness,self.dt],device=d)
        if self.A:wp.launch(potential,dim=n,inputs=[s,self.volumes,self.A,wp.vec3(*self.center),self.width,self.dt],device=d)
        wp.launch(compute_stress_from_F_trial,dim=n,inputs=[s,m,self.dt],device=d)
        wp.launch(p2g_apic_with_stress,dim=n,inputs=[s,m,self.dt],device=d)
        wp.launch(grid_normalization_and_gravity,dim=grid,inputs=[s,m,self.dt],device=d)
        wp.launch(add_damping_via_grid,dim=grid,inputs=[s,m.grid_v_damping_scale],device=d)
        wp.launch(g2p,dim=n,inputs=[s,m,self.dt],device=d)
    def solve(self,pose,seconds=2.,block=50):
        self.set_pose(pose)
        if self.graph is None:
            self.step() # compile before capture
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(block):self.step()
            self.graph=capture.graph
        wp.launch(reset,dim=len(self.rest),inputs=[self.sim.mpm_state,self.rest_wp],device=self.device)
        frames=max(1,int(np.ceil(seconds/self.dt/block)));previous=None;start=time.time()
        for j in range(frames):
            wp.capture_launch(self.graph);wp.synchronize_device(self.device)
            if j==frames-11:previous=wp.to_torch(self.sim.mpm_state.particle_x).detach().cpu().numpy()[self.sample_ids].copy()
        x=wp.to_torch(self.sim.mpm_state.particle_x).detach().cpu().numpy();v=wp.to_torch(self.sim.mpm_state.particle_v).detach().cpu().numpy();assert np.isfinite(x).all() and np.max(abs(x))<2.,'Divergent MPM'
        return dict(field=x[self.sample_ids]-self.shift-self.rest[self.sample_ids],drift_m=float(np.sqrt(np.mean((x[self.sample_ids]-previous)**2))) if previous is not None else None,max_speed=float(np.linalg.norm(v,axis=1).max()),elapsed_seconds=time.time()-start)

if __name__=='__main__':
    import argparse,json
    from scripts.position_ood.fixture import controls
    p=argparse.ArgumentParser();p.add_argument('--resolution',type=int,default=1);p.add_argument('--grid',type=int,default=64);p.add_argument('--seconds',type=float,default=2.);p.add_argument('--dt',type=float,default=.0002);a=p.parse_args();m=PositionMPM(a.resolution,a.grid,a.dt);m.parameters(40000,15000,15000,'opposite_y',field_amplitude_J=-.02)
    r=m.solve(controls('rest'),seconds=a.seconds);field=r.pop('field');q=json.loads(Path('reports/position-ood-v09/pilot-2-qualification.json').read_text());ref=np.load(ROOT/f'runs/position-ood-v09/pilot-2/n{q["mesh"]}-q3.npz')['field'][0];r['rmse_mm']=float(1000*np.sqrt(np.mean((field-ref)**2)));print(json.dumps(r),flush=True)
    out=ROOT/f'runs/position-ood-v09/mpm-pilot/r{a.resolution}-g{a.grid}-dt{a.dt}-s{a.seconds}';out.parent.mkdir(exist_ok=True);np.savez_compressed(str(out)+'.npz',field=field);Path(str(out)+'.json').write_text(json.dumps(r,indent=2)+'\n')
