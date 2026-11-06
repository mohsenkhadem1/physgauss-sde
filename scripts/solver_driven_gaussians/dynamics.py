"""SI-unit dynamic FEM with the existing frozen law and Gaussian interactions.

World-space integration preserves independent rigid motion without a learned
motion network. All renderer positions must be interpolated from these nodes.
Contact is a finite penalty, not exact nonpenetration or continuous collision.
"""
from pathlib import Path
import itertools
import numpy as np
import torch
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.materials import teacher_energy

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'runs/compositional-neural-mechanics-v16/scale-separated-prototype/shared-law.pt'


def block_mesh(cells=(8, 4, 4), dimensions=(.12, .06, .06)):
    axes = [np.linspace(0, d, n+1) for n, d in zip(cells, dimensions)]
    x = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    x[:, 1:] -= np.array(dimensions[1:])/2
    shape = tuple(n+1 for n in cells)
    tetra = []
    for base in itertools.product(*(range(n) for n in cells)):
        for perm in itertools.permutations(range(3)):
            q = np.array(base); corners = [q.copy()]
            for axis in perm:
                q = q.copy(); q[axis] += 1; corners.append(q)
            tetra.append([np.ravel_multi_index(tuple(q), shape) for q in corners])
    tetra = np.asarray(tetra)
    dm = (x[tetra[:, 1:]] - x[tetra[:, :1]]).transpose(0, 2, 1)
    flip = np.linalg.det(dm) < 0
    tetra[flip, 1:3] = tetra[flip, 2:0:-1]
    return x, tetra


class DynamicFEM:
    def __init__(self, cells=(8,4,4), dimensions=(.12,.06,.06), density=625., device='cpu', dtype=torch.float64):
        self.device=device; self.dtype=dtype; self.cells=cells; self.dimensions=dimensions
        x, e = block_mesh(cells, dimensions)
        self.rest=self.t(x); self.elements=torch.tensor(e,device=device,dtype=torch.long)
        dm=(self.rest[self.elements[:,1:]]-self.rest[self.elements[:,:1]]).transpose(-1,-2)
        self.volume=torch.linalg.det(dm)/6
        inv=torch.linalg.inv(dm)
        self.grad=torch.cat([-inv.sum(1,keepdim=True),inv],1)
        self.mass=torch.zeros(len(x),device=device,dtype=dtype).index_add(0,self.elements.flatten(),(density*self.volume[:,None]/4).expand(-1,4).flatten())
        self.law=ScaleSeparatedMaterialLaw().to(device=device,dtype=dtype)
        self.law.load_state_dict(torch.load(CHECKPOINT,map_location=device,weights_only=False)['state_dict']);self.law.requires_grad_(False)
        self.coefficients=self.law.base.coefficients(self.t([0.,-1.])).detach()
        self.slopes=(torch.nn.functional.softplus(self.law.base.raw_slopes)+1e-4).detach()
        # Candidate attachment locations span both end faces; fitting chooses weights.
        self.centers=self.t([[xx,yy,zz] for xx in [0.,dimensions[0]] for yy in [-.015,.015] for zz in [-.015,.015]])
        raw=torch.exp(-.5*(((self.rest[:,None]-self.centers[None])/self.t([.012,.022,.022]))**2).sum(-1))
        raw=raw*self.mass[:,None]
        self.attachment_weights=raw/raw.sum(0,keepdim=True)
        # Surface quadrature makes contact stiffness independent of volume mesh size.
        area=np.zeros(len(x));grid=np.arange(len(x)).reshape(tuple(n+1 for n in cells))
        for axis in range(3):
            other=[j for j in range(3) if j!=axis]
            weights=[np.r_[.5,np.ones(cells[j]-1),.5]*dimensions[j]/cells[j] for j in other]
            face_area=np.outer(*weights)
            for side in [0,-1]:
                sl=[slice(None)]*3;sl[axis]=side;area[grid[tuple(sl)].ravel()]+=face_area.ravel()
        self.surface_area=self.t(area)
        self.floor=-.09; self.floor_penalty=4e7; self.floor_damping=.15

    def t(self,x): return torch.as_tensor(x,device=self.device,dtype=self.dtype)

    def deformation(self,x): return torch.einsum('...eai,eaj->...eij',x[...,self.elements,:],self.grad)

    def bulk_autograd(self,x,scale=1.,kind='learned',create_graph=False):
        with torch.enable_grad():
            F=self.deformation(x)
            if not F.requires_grad:F=F.detach().requires_grad_(True)
            J=torch.linalg.det(F)
            if not bool(torch.isfinite(J).all()) or float(J.detach().min())<=.05:raise ValueError('Inverted or nearly singular dynamic element')
            if kind=='learned':
                z=F.new_tensor([0.,0.,-1.]).expand(*F.shape[:-2],3)
                W=self.law.energy(F,z)*scale
            elif kind=='reference':
                z=F.new_zeros((*F.shape[:-2],2));W=teacher_energy(F,z,'mooney_rivlin')*scale
            elif kind=='analytic':
                from scripts.compositional_neural_mechanics.materials import invariants
                q1,_,volume=invariants(F);W=(2000*q1+20000*volume)*scale
            else:raise ValueError(kind)
            P=torch.autograd.grad(W.sum(),F,create_graph=create_graph)[0]
            local=-torch.einsum('e,...eij,eaj->...eai',self.volume,P,self.grad)
            force=torch.zeros_like(x).index_add(-2,self.elements.flatten(),local.reshape(*x.shape[:-2],-1,3))
            energy=(W*self.volume).sum(-1)
        if not create_graph:force=force.detach();energy=energy.detach();J=J.detach()
        return force,energy,J

    def bulk(self,x,scale=1.,kind='learned',create_graph=False):
        """Exact derivative of the frozen seven-basis law; avoids nested AD in rollout."""
        F=self.deformation(x);J=torch.linalg.det(F)
        if not bool(torch.isfinite(J).all()) or float(J.detach().min())<=.05:raise ValueError('Inverted or nearly singular dynamic element')
        C=F.transpose(-1,-2)@F;I1=C.diagonal(dim1=-2,dim2=-1).sum(-1);I2=.5*(I1.square()-C.square().sum((-1,-2)))
        q1=J.pow(-2/3)*I1-3;q2=J.pow(-4/3)*I2-3;volume=J-torch.log(J)-1
        invT=torch.linalg.inv(F).transpose(-1,-2)
        D1=2*J.pow(-2/3)[...,None,None]*F-(2/3)*(q1+3)[...,None,None]*invT
        D2=2*J.pow(-4/3)[...,None,None]*(I1[...,None,None]*F-F@C)-(4/3)*(q2+3)[...,None,None]*invT
        DV=(J-1)[...,None,None]*invT
        if kind=='learned':
            c=self.coefficients;s=self.slopes;q=q1[...,None]*s
            residual=torch.nn.functional.softplus(q)-np.log(2)-.5*q
            W=12000*(.5*c[0]*q1+.5*c[1]*q2+c[2]*volume+c[3]*q1.square()+c[4]*q1.pow(3)+(c[5:]*residual).sum(-1))
            a=.5*c[0]+2*c[3]*q1+3*c[4]*q1.square()+(c[5:]*(torch.sigmoid(q)-.5)*s).sum(-1)
            P=12000*(a[...,None,None]*D1+.5*c[1]*D2+c[2]*DV)
        elif kind=='reference':W=1100*q1+900*q2+20000*volume;P=1100*D1+900*D2+20000*DV
        elif kind=='analytic':W=2000*q1+20000*volume;P=2000*D1+20000*DV
        else:raise ValueError(kind)
        P=P*scale;W=W*scale
        local=-torch.einsum('e,...eij,eaj->...eai',self.volume,P,self.grad)
        force=torch.zeros_like(x).index_add(-2,self.elements.flatten(),local.reshape(*x.shape[:-2],-1,3));energy=(W*self.volume).sum(-1)
        if not create_graph:force=force.detach();energy=energy.detach();J=J.detach()
        return force,energy,J

    def control(self,t,case='calibration'):
        # Position input, in metres; smooth onset avoids impulsive constraints.
        if case not in ['calibration','side_poke','early_release','force_pulse','force_reverse','higher_drop']:raise ValueError(case)
        release=.18 if case!='early_release' else .12
        phase=min(max(t/.08,0.),1.);s=phase**3*(10-15*phase+6*phase**2)
        direction=self.t([0.,0.,-.025]) if case!='side_poke' else self.t([0.,.025,-.016])
        gate=1.-min(max((t-release)/.015,0.),1.)
        poke_gate=1.-min(max((t-.115)/.015,0.),1.)
        reference=self.rest+(self.t([0.,0.,.035]) if case=='higher_drop' else 0.)
        return dict(reference=reference,target=reference+s*direction,gate=gate,poke_gate=poke_gate,case=case,time=t)

    def components(self,x,v,t,case):
        ctrl=self.control(t,case);delta=ctrl['reference']-x
        support=self.attachment_weights[:,:,None]*delta[:,None,:]*ctrl['gate']
        point=self.t([.085,0.,.03]);w=torch.exp(-.5*(((self.rest-point)/self.t([.023,.022,.009]))**2).sum(-1));w=w*self.mass;w=w/w.sum()
        tool=w[:,None]*(ctrl['target']-x)*ctrl['poke_gate']
        damping=-self.mass[:,None]*v
        depth=torch.relu(self.floor-x[:,2]);active=(depth>0).to(x.dtype)
        stiffness=self.floor_penalty*self.surface_area
        fz=stiffness*depth-2*self.floor_damping*torch.sqrt(stiffness*self.mass)*torch.minimum(v[:,2],torch.zeros_like(depth))*active
        contact=torch.stack([torch.zeros_like(fz),torch.zeros_like(fz),fz],-1)
        gravity=self.mass[:,None]*self.t([0.,0.,-9.81])
        external=torch.zeros_like(x)
        if case in ['force_pulse','force_reverse'] and .04<=t<=.09:
            # A prescribed known test load, never a fitting target.
            external=w[:,None]*self.t([.6,0.,-.2] if case=='force_pulse' else [-1.2,.6,-.4])
        return support,tool,damping,contact+gravity+external

    def force(self,x,v,t,parameters,case,kind='learned',create_graph=False):
        fb,E,J=self.bulk(x,parameters[0],kind,create_graph)
        support,tool,damp,known=self.components(x,v,t,case)
        f=fb+torch.einsum('nki,k->ni',support,parameters[1:9])+parameters[9]*tool+parameters[10]*damp+known
        return f,E,J

    def rollout(self,parameters,case='calibration',kind='learned',duration=.42,dt=.0002,sample_dt=.002,initial=None,create_graph=False):
        parameters=self.t(parameters)
        x=self.rest.clone() if initial is None else initial[0]
        v=torch.zeros_like(x) if initial is None else initial[1]
        if case=='higher_drop' and initial is None:x=x+self.t([0.,0.,.035])
        stride=round(sample_dt/dt);steps=round(duration/dt)
        if abs(stride*dt-sample_dt)>1e-10:raise ValueError('Sample clock mismatch')
        states=[];velocities=[];times=[];minJ=1.;maxpenetration=0.
        for step in range(steps+1):
            t=step*dt
            f,E,J=self.force(x,v,t,parameters,case,kind,create_graph)
            if step%stride==0:
                states.append(x if create_graph else x.detach());velocities.append(v if create_graph else v.detach());times.append(t)
                minJ=min(minJ,float(J.detach().min()));maxpenetration=max(maxpenetration,float(torch.relu(self.floor-x[:,2]).max().detach()))
            if step==steps:break
            # Symplectic Euler; refinement is required for quantitative use.
            v=v+dt*f/self.mass[:,None];x=x+dt*v
            if not create_graph:x=x.detach();v=v.detach()
        return dict(x=torch.stack(states),v=torch.stack(velocities),times=np.asarray(times),minimum_J=minJ,max_penetration_m=maxpenetration)

