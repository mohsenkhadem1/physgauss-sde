"""Independent quasi-static Neo-Hookean reference for controlled attachment motion."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy import sparse
from experiments.campaign_static_uniaxial_bar import PrismaticBar,face_area_weights
from experiments.campaign_static_supported_block import StaticSupportedBlock

SIZE=(.24,.08,.08)
MATERIALS={'soft_firm':(2000.,1000.),'stiff_loose':(8000.,64.)}
ACTIONS={'axial':np.array([.009,0,0]),'sideways':np.array([0,.009,0]),'oblique':np.array([.004,-.007,.005])}

class ControlledBlock(StaticSupportedBlock):
    def __init__(self,n,young,support):
        super().__init__(cells=(4*n,n,n),boundary='left_face',young=young,poisson=.35,
                         density=625.,gravity=0.,traction=0.)
        self.mask[:]=False;self.free=np.arange(self.rest.size)
        self.left,_=face_area_weights(self.reference_local,self.faces,0,0.)
        self.right,_=face_area_weights(self.reference_local,self.faces,0,SIZE[0])
        self.left/=self.left.sum();self.right/=self.right.sum()
        self.support=support;self.actuator=1000.;self.target=np.zeros(3)
        self.diag=np.repeat(support*self.left+self.actuator*self.right,3)
        self.spring_matrix=sparse.diags(self.diag,format='csr')
        self.force=10.
    def evaluate(self,u,tangent=False):
        x=self.rest+u
        elastic,gradient,H,J=self.elastic(x,tangent)
        a=u-self.target
        spring=.5*self.support*np.sum(self.left[:,None]*u*u)+.5*self.actuator*np.sum(self.right[:,None]*a*a)
        residual=gradient+self.support*self.left[:,None]*u+self.actuator*self.right[:,None]*a
        return dict(potential_J=float(elastic+spring),elastic_J=float(elastic),spring_J=float(spring),
                    work_J=0.,residual=residual,tangent=H+self.spring_matrix if tangent else None,J=J,positions=x)


def generate(output):
    import json
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fields={};checks=[]
    for name,(E,K) in MATERIALS.items():
        for action,delta in ACTIONS.items():
            for n in [2,4,8]:
                model=ControlledBlock(n,E,K);u=np.zeros_like(model.rest);states=[];js=[];res=[]
                for alpha in np.linspace(0,1,9):
                    model.target=delta*alpha
                    answer=model.solve(u);u=answer['displacement_local_m']
                    states.append(answer['positions']);js.append(float(answer['J'].min()))
                    res.append(float(np.linalg.norm(answer['residual'])))
                trajectory=np.stack(states)
                # Nested structured grids: restrict exactly, without interpolation.
                restricted=trajectory.reshape(9,4*n+1,n+1,n+1,3)[:,::n//2,::n//2,::n//2].reshape(9,-1,3)
                fields[name,action,n]=restricted
                np.savez_compressed(output/f'{name}-{action}-n{n}.npz',positions=trajectory,
                    rest=model.rest,tets=model.tets,faces=model.faces,mass=model.node_mass,
                    sampled_positions=restricted,minimum_J=np.array(js),residual_N=np.array(res))
                print(name,action,n,'minJ',min(js),'res',max(res),flush=True)
            coarse=np.sqrt(np.mean((fields[name,action,2]-fields[name,action,4])**2))
            fine=np.sqrt(np.mean((fields[name,action,4]-fields[name,action,8])**2))
            checks.append(dict(object=name,action=action,coarse_rms_m=float(coarse),fine_rms_m=float(fine),
                max_refinement_m=float(np.max(abs(fields[name,action,4]-fields[name,action,8]))),
                passed=bool(fine<.0003 and fine<coarse)))
    result=dict(checks=checks,passed=all(c['passed'] for c in checks),
        scope='Static compressible Neo-Hookean P1, distributed end-face Robin attachments; no gravity/contact. Nested mesh checks at common material points. Not a dynamic continuum validation.')
    (output/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();generate(a.output)
