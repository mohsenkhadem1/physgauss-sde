"""Nonlinear supported block with unknown fixture side/extent and competing loads."""
from pathlib import Path
import argparse
import concurrent.futures
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy import sparse
from experiments.campaign_static_supported_block import StaticSupportedBlock
from experiments.campaign_static_uniaxial_bar import face_area_weights

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/phystwin-active-support-v03'
LAYOUTS=['left_full','left_lower','right_full','right_lower']
ACTIONS=['initial','push_left','push_right','held_oblique']
BOUNDS=np.log([[1500.,10000.],[400.,16000.]])


class ActiveBlock(StaticSupportedBlock):
    def __init__(self,n,E,K,layout):
        super().__init__(cells=(4*n,n,n),boundary='left_face',young=E,poisson=.35,density=625.,gravity=0.,traction=0.)
        self.mask[:]=False;self.free=np.arange(self.rest.size)
        side=0. if layout.startswith('left') else .24
        _,faces=face_area_weights(self.rest,self.faces,0,side)
        if layout.endswith('lower'):
            faces=faces[np.all(self.rest[faces,2]<=1e-13,axis=1)]
        p=self.rest[faces];areas=.5*np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        weights=np.zeros(len(self.rest))
        for j in range(3):np.add.at(weights,faces[:,j],areas/3)
        self.attachment_area=float(weights.sum());self.support_weights=weights/weights.sum();self.K=K
        self.spring_matrix=sparse.diags(np.repeat(K*self.support_weights,3),format='csr')
        self.left,_=face_area_weights(self.rest,self.faces,0,0.)
        self.right,_=face_area_weights(self.rest,self.faces,0,.24)
        self.left/=self.left.sum();self.right/=self.right.sum()
        self.force=10.;self.load=np.zeros_like(self.rest)
    def set_action(self,action,alpha=1.):
        if action=='initial':
            load=(self.left+self.right)[:,None]*np.array([.03,.0015,.001])
        elif action=='push_left':load=self.left[:,None]*np.array([0.,.03,0.])
        elif action=='push_right':load=self.right[:,None]*np.array([0.,.03,0.])
        elif action=='held_oblique':
            load=self.left[:,None]*np.array([.01,.007,.006])+self.right[:,None]*np.array([-.005,.009,-.004])
        else:raise ValueError(action)
        self.load=alpha*load
    def evaluate(self,u,tangent=False):
        x=self.rest+u;elastic,gradient,H,J=self.elastic(x,tangent)
        spring=.5*self.K*np.sum(self.support_weights[:,None]*u*u)
        work=np.sum(self.load*u)
        residual=gradient+self.K*self.support_weights[:,None]*u-self.load
        return dict(potential_J=float(elastic+spring-work),elastic_J=float(elastic),spring_J=float(spring),
            work_J=float(work),residual=residual,tangent=H+self.spring_matrix if tangent else None,J=J,positions=x)


def cohort(count,seed):
    rng=np.random.default_rng(seed)
    pars=np.exp(BOUNDS[:,0]+rng.random((count,2))*np.diff(BOUNDS,axis=1).ravel())
    return [dict(layout=LAYOUTS[i%4],E=float(p[0]),K=float(p[1])) for i,p in enumerate(pars)]


def protocol():
    spec=dict(version=3,layouts=LAYOUTS,layout_prior='uniform; side and extent latent, not given to policies',
        parameter_prior='independent log-uniform E[1500,10000] Pa, K[400,16000] N/m',
        development=cohort(12,401),calibration=cohort(20,402),test=cohort(24,403),
        bank_grid=9,bank_mesh_n=4,truth_mesh_n=8,noise_sigma_m=.0003,noise_repeats=4,
        initial='27 coordinates on center cross section under shared small axial/transverse dead load',
        probes='left or right end 0.03 N y force, observe three coordinates at that loaded end center; reset',
        target='all 243 coordinates under a separate asymmetric oblique load',
        physics='compressible Neo-Hookean, distributed isotropic Robin attachment, full or lower half end face, no gravity/contact',
        quadrature_sizes=[33,65,129],quadrature_gate_mean_sd_m=.00001,
        interpolation_gate_rms_m=.00003,probe_nonrigid_fraction_gate=.02,probe_nonrigid_rms_gate_m=.00005,
        discrepancy='development residual mean and 10%-diagonal-shrunk covariance plus 1 micrometre SD floor',
        utility='expected posterior whole-field variance, joint numerical discrepancy conditioned on prefix and probe',
        utility_samples=[256,1024],utility_refinement_samples=4096,
        baselines=['best fixed on calibration','exact uniform random','MAP fixture-side opposite-end heuristic'],
        calibration_rule='object-wise maximum across all budgets, coordinates and four repeats; 19th of 20 scores, 90% simultaneous bands',
        success='paired object-bootstrap 95% interval for RMSE improvement over best fixed excludes zero; calibrated simultaneous test object coverage >=90%; finite refinement checks pass',
        scope='new parameter pairs within four newly introduced fixture families, not unseen topology outside the prior; numerical reference, not learned PhysTwin')
    BASE.mkdir(parents=True,exist_ok=True);p=BASE/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==spec
    else:p.write_text(json.dumps(spec,indent=2)+'\n')
    return spec


def solve_job(job):
    name,E,K,layout,n=job;p=BASE/'fields'/f'{name}.npz'
    if p.exists():return name,'cached'
    fields=[];checks=[]
    for action in ACTIONS:
        m=ActiveBlock(n,E,K,layout);u=np.zeros_like(m.rest)
        for alpha in np.linspace(0,1,9)[1:]:
            m.set_action(action,alpha);answer=m.solve(u);u=answer['displacement_local_m']
        nonrigid=m.bulk_nonrigid_RMS(answer['positions']);motion=float(np.sqrt(np.mean(u*u)))
        fields.append(u.reshape(4*n+1,n+1,n+1,3)[::n//2,::n//2,::n//2].reshape(81,3))
        checks.append([float(np.linalg.norm(answer['residual'])),float(answer['J'].min()),nonrigid,motion,
                       float(np.max(np.linalg.norm(u,axis=1)))])
    tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,fields=fields,checks=checks,parameters=[E,K],layout=layout,n=n)
    tmp.replace(p);return name,'solved'


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8)
    p.add_argument('--split',choices=['all','bank','development','calibration','test'],default='all')
    p.add_argument('--bank-grid',type=int,choices=[9,17],default=9);a=p.parse_args()
    spec=protocol();(BASE/'fields').mkdir(exist_ok=True);jobs=[]
    if a.split in ['all','bank']:
        for l,layout in enumerate(LAYOUTS):
            for i,e in enumerate(np.linspace(*BOUNDS[0],a.bank_grid)):
                for j,k in enumerate(np.linspace(*BOUNDS[1],a.bank_grid)):
                    jobs.append((f'bank{a.bank_grid}-{l}-{i}-{j}',np.exp(e),np.exp(k),layout,4))
    for split in ['development','calibration','test']:
        if a.split not in ['all',split]:continue
        for i,r in enumerate(spec[split]):
            jobs.append((f'{split}-{i}-n8',r['E'],r['K'],r['layout'],8))
            if split=='development':jobs.append((f'{split}-{i}-n4',r['E'],r['K'],r['layout'],4))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,result in enumerate(pool.map(solve_job,jobs)):print(i+1,len(jobs),*result,flush=True)

if __name__=='__main__':main()
