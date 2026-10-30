"""Independent small coupled-contact qualification; not the benchmark predictor.

Spring-lattice deformable body + translating/rotating rigid sphere. Prescribed
pusher position is the only actuation; gravity/floor are explicit external terms.
Discrete nodes and penalty contact are approximate and allow finite penetration.
"""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/contact-benchmark-v12'


class CoupledSystem:
    def __init__(self, friction=.4, gravity=False, floor=False, pusher=False):
        grid=np.linspace(-.025,.025,3)
        self.x=np.array([[x-.075,y,z+.045] for x in grid for y in grid for z in grid])
        self.v=np.zeros_like(self.x);self.mass=.001;self.center=np.array([0.,0.,.04]);self.velocity=np.zeros(3)
        self.omega=np.zeros(3);self.radius=.025;self.rigid_mass=.04;self.inertia=.4*self.rigid_mass*self.radius**2
        pairs=np.array([(i,j) for i in range(len(self.x)) for j in range(i+1,len(self.x)) if np.linalg.norm(self.x[j]-self.x[i])<.044])
        self.pairs=pairs;self.rest=np.linalg.norm(self.x[pairs[:,1]]-self.x[pairs[:,0]],axis=1)
        self.k=18.;self.spring_damping=.006;self.contact_k=4000.;self.contact_damping=2*np.sqrt(self.contact_k*self.mass)
        self.friction=friction;self.gravity=gravity;self.floor=floor;self.pusher=pusher
        self.max_penetration=0.;self.friction_work=0.;self.contacts=0

    def momentum(self):
        return self.mass*self.v.sum(0)+self.rigid_mass*self.velocity

    def angular_momentum(self):
        return np.cross(self.x,self.mass*self.v).sum(0)+np.cross(self.center,self.rigid_mass*self.velocity)+self.inertia*self.omega

    def contact_forces(self):
        r=self.x-self.center;distance=np.linalg.norm(r,axis=1);normal=r/np.maximum(distance[:,None],1e-12)
        depth=np.maximum(self.radius-distance,0);active=depth>0
        relative=self.v-self.velocity-np.cross(self.omega,r)
        vn=np.sum(relative*normal,axis=1);normal_force=np.maximum(self.contact_k*depth-self.contact_damping*vn,0)*active
        tangent=relative-vn[:,None]*normal;speed=np.linalg.norm(tangent,axis=1)
        magnitude=np.minimum(.1*speed,self.friction*normal_force)
        tangential=-magnitude[:,None]*tangent/np.maximum(speed[:,None],1e-12)
        force=normal_force[:,None]*normal+tangential
        rigid=-force.sum(0);torque=np.cross(r,-force).sum(0)
        friction_power=float(np.sum(tangential*relative))
        return force,rigid,torque,float(depth.max()),friction_power,int(active.sum())

    def forces(self,t):
        i,j=self.pairs.T;d=self.x[j]-self.x[i];length=np.linalg.norm(d,axis=1);n=d/length[:,None]
        relative=np.sum((self.v[j]-self.v[i])*n,axis=1)
        link=(self.k*(length-self.rest)+self.spring_damping*relative)[:,None]*n
        force=np.zeros_like(self.x);np.add.at(force,i,link);np.add.at(force,j,-link)
        cf,rigid,torque,penetration,power,count=self.contact_forces();force+=cf
        if self.gravity:
            force[:,2]-=self.mass*9.81;rigid[2]-=self.rigid_mass*9.81
        if self.floor:
            depth=np.maximum(-self.x[:,2],0);normal=np.maximum(self.contact_k*depth-self.contact_damping*self.v[:,2],0)*(depth>0)
            force[:,2]+=normal
            tang=self.v[:,:2];speed=np.linalg.norm(tang,axis=1);mag=np.minimum(.1*speed,self.friction*normal)
            force[:,:2]-=mag[:,None]*tang/np.maximum(speed[:,None],1e-12)
            rd=max(self.radius-self.center[2],0.)
            rigid[2]+=max(self.contact_k*rd-(2*np.sqrt(self.contact_k*self.rigid_mass))*self.velocity[2],0.)*(rd>0)
            # Sphere-floor force/torque share the contact point.
            arm=np.array([0.,0.,-self.radius]);sv=self.velocity+np.cross(self.omega,arm);sv[2]=0
            sf=-sv/(np.linalg.norm(sv)+1e-12)*min(.1*np.linalg.norm(sv),self.friction*max(self.contact_k*rd-(2*np.sqrt(self.contact_k*self.rigid_mass))*self.velocity[2],0))*(rd>0)
            rigid+=sf;torque+=np.cross(arm,sf)
            penetration=max(penetration,float(depth.max()),rd)
        if self.pusher:
            plane=-.108+.08*min(t,.9);speed=.08 if t<.9 else 0.
            depth=np.maximum(plane-self.x[:,0],0)
            force[:,0]+=np.maximum(self.contact_k*depth-self.contact_damping*(self.v[:,0]-speed),0)*(depth>0)
            penetration=max(penetration,float(depth.max()))
        return force,rigid,torque,penetration,power,count

    def step(self,t,dt):
        force,rigid,torque,depth,power,count=self.forces(t)
        self.v+=dt*force/self.mass;self.velocity+=dt*rigid/self.rigid_mass;self.omega+=dt*torque/self.inertia
        self.x+=dt*self.v;self.center+=dt*self.velocity
        self.max_penetration=max(self.max_penetration,depth);self.friction_work+=power*dt;self.contacts+=count


def run(dt=1e-4,duration=1.2,mode='demo',friction=.4):
    system=CoupledSystem(friction=friction,gravity=mode=='demo',floor=mode=='demo',pusher=mode=='demo')
    if mode=='isolated':
        system.v[:]=[.12,.04,0];system.center[:]=[-.015,0,.045]
    initial_p=system.momentum();initial_l=system.angular_momentum();initial_shape=system.x.copy();initial_center=system.center.copy();p_error=0.;l_error=0.;frames=[];centers=[];times=[]
    stride=max(1,round(1/(50*dt)))
    for n in range(round(duration/dt)):
        system.step(n*dt,dt)
        if n%stride==0:
            frames.append(system.x.copy());centers.append(system.center.copy());times.append((n+1)*dt)
            p_error=max(p_error,float(np.linalg.norm(system.momentum()-initial_p)))
            l_error=max(l_error,float(np.linalg.norm(system.angular_momentum()-initial_l)))
    frames.append(system.x.copy());centers.append(system.center.copy());times.append(duration)
    from scripts.external_benchmark.data import rigid_map
    nonrigid=[]
    for frame in frames:
        R,b=rigid_map(initial_shape,frame);nonrigid.append(float(np.sqrt(np.mean((initial_shape@R+b-frame)**2))))
    return dict(dt=dt,mode=mode,friction=friction,max_nonrigid_coordinate_rms_m=max(nonrigid),sphere_displacement_m=float(np.linalg.norm(system.center-initial_center)),max_penetration_m=system.max_penetration,contact_samples=system.contacts,
                friction_work_j=system.friction_work,max_linear_momentum_error=p_error,max_angular_momentum_error=l_error,
                final_state=np.r_[system.x.ravel(),system.center],final_sphere_velocity=system.velocity,
                frames=np.asarray(frames),centers=np.asarray(centers),times=np.asarray(times))


def render(result,path):
    import matplotlib;matplotlib.use('Agg')
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    fig=plt.figure(figsize=(9,5));ax=fig.add_subplot(111,projection='3d')
    u,v=np.meshgrid(np.linspace(0,2*np.pi,24),np.linspace(0,np.pi,12));ball=.025*np.array([np.cos(u)*np.sin(v),np.sin(u)*np.sin(v),np.cos(v)])
    writer=FFMpegWriter(fps=10,bitrate=1600,extra_args=['-threads','2'])
    with writer.saving(fig,str(path),100):
        for k in range(0,len(result['frames']),2):
            ax.clear();x=result['frames'][k];c=result['centers'][k];t=result['times'][k]
            ax.scatter(*x.T,c=x[:,2],cmap='viridis',s=28,vmin=0,vmax=.08)
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            edges=CoupledSystem().pairs
            ax.add_collection3d(Line3DCollection(x[edges],colors='teal',linewidths=.5,alpha=.5))
            ax.plot_surface(ball[0]+c[0],ball[1]+c[1],ball[2]+c[2],color='tomato',alpha=.7)
            plane=-.108+.08*min(t,.9);yy,zz=np.meshgrid([-.04,.04],[0,.09]);ax.plot_surface(np.full_like(yy,plane),yy,zz,color='gray',alpha=.25)
            ax.set(xlim=(-.12,.12),ylim=(-.07,.07),zlim=(0,.11),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=f'Coupled qualification simulation: soft lattice + movable sphere | {t:.2f}s')
            ax.set_box_aspect((.24,.14,.11));ax.view_init(20, -60+20*t/1.2);writer.grab_frame()
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video',action='store_true');args=parser.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    isolated=run(mode='isolated',duration=.6);no_friction=run(mode='isolated',duration=.6,friction=0)
    runs=[run(dt=d) for d in [2e-4,1e-4,5e-5]]
    difference=[float(np.sqrt(np.mean((r['final_state']-runs[-1]['final_state'])**2))) for r in runs[:-1]]
    def summary(r):
        row={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items() if k not in ['frames','centers','times','final_state']}
        if r['mode']=='demo':
            row['max_linear_momentum_change_external_forces']=row.pop('max_linear_momentum_error')
            row['max_angular_momentum_change_external_forces']=row.pop('max_angular_momentum_error')
        return row
    report=dict(scope='Independent synthetic spring-lattice/rigid-sphere qualification; not benchmark predictor or ground-truth comparison',
        mechanics='Equal/opposite unilateral penalty contact; Coulomb-capped viscous friction; rigid translation/spin; position-driven pusher',
        limitations=['Discrete node collisions','Finite penalty penetration','Explicit symplectic Euler','Spring lattice not calibrated FEM','Sphere rotational state is angular velocity; orientation immaterial for sphere geometry'],
        isolated=summary(isolated),isolated_no_friction=summary(no_friction),refinement=[summary(r) for r in runs],final_state_rms_difference_to_finest_m=difference,
        gates=dict(linear_momentum=isolated['max_linear_momentum_error']<1e-10,angular_momentum=isolated['max_angular_momentum_error']<1e-10,
                   friction_dissipative=isolated['friction_work_j']<=0,contact_exercised=isolated['contact_samples']>0,
                   refinement_improves=difference[1]<difference[0],finest_penetration_below_2mm=runs[-1]['max_penetration_m']<.002))
    (OUT/'coupled-qualification.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item())+'\n')
    np.savez_compressed(ROOT/'runs/contact-benchmark-v12/coupled-qualification.npz',frames=runs[-1]['frames'],centers=runs[-1]['centers'],times=runs[-1]['times'])
    if args.video:render(runs[-1],OUT/'coupled-qualification.mp4')
    print(json.dumps(report,indent=2,default=lambda x:x.item()))

if __name__=='__main__':main()
