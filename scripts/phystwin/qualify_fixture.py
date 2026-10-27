"""Frozen convergence qualification for v03's most mesh-sensitive fixture."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.stats import qmc
from scripts.phystwin.consistent_fixture import P1Fixture,P2Fixture,query
from scripts.phystwin.active_support_reference import ROOT,ACTIONS
BASE=ROOT/'runs/phystwin-fixture-convergence-v04'


def protocol():
    old=json.loads((ROOT/'reports/phystwin-active-support-v03/protocol.json').read_text())
    spec=dict(case=old['development'][11],source='v03 development object 11, chosen before new solves',actions=ACTIONS,
        model='same log-J Neo-Hookean and Robin surface stiffness; consistent surface integration',
        p1_levels=[4,8,12],p2_levels=[2,4,6,8],quadrature_orders=[3,4],
        displacement_rms_tolerance_m=.00002,displacement_max_tolerance_m=.0001,
        strain_relative_L2_tolerance=.05,energy_relative_tolerance=.01,traction_relative_L2_tolerance=.02,
        force_balance_tolerance_N=1e-8,moment_balance_tolerance_Nm=1e-9,nonlinear_residual_tolerance_N=1e-9,
        quadrature_displacement_rms_tolerance_m=.000001,quadrature_energy_relative_tolerance=.0001,
        sampling='425 structured displacement points; 512 scrambled Sobol interior strain points; 64 support-interior traction points',
        policy='Require all four actions to pass successive P2 spatial refinements and quadrature checks; do not rebuild inference on failed reference',
        scope='Single difficult development fixture, no inference/test-set retuning; finite convergence evidence, not analytic error bound')
    BASE.mkdir(parents=True,exist_ok=True);p=BASE/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==spec
    else:p.write_text(json.dumps(spec,indent=2)+'\n')
    return spec


def points(layout='right_lower'):
    disp=np.stack(np.meshgrid(np.linspace(0,.24,17),np.linspace(-.04,.04,5),np.linspace(-.04,.04,5),indexing='ij'),-1).reshape(-1,3)
    strain=np.array([0.,-.04,-.04])+qmc.Sobol(3,scramble=True,seed=911).random_base2(9)*[.24,.08,.08]
    y=-.04+(np.arange(8)+.5)*.08/8;z=-.04+(np.arange(8)+.5)*(.04 if layout.endswith('lower') else .08)/8
    support=np.stack(np.meshgrid([0. if layout.startswith('left') else .24],y,z,indexing='ij'),-1).reshape(-1,3)
    return disp,strain,support


def metrics(m,u,state):
    disp,interior,support=points(m.layout);G=query(m,u,interior,True);F=G+np.eye(3)
    strain=.5*(np.einsum('pji,pjk->pik',F,F)-np.eye(3))
    traction=-m.K/m.attachment_area*query(m,u,support)
    reaction=-(m.fixture_scalar@u);forces=reaction+m.load
    return dict(sampled_u=query(m,u,disp),strain=strain,traction=traction,
        force_resultant=reaction.sum(0),moment_resultant=np.cross(m.rest+u,reaction).sum(0),
        force_balance_N=float(np.linalg.norm(forces.sum(0))),moment_balance_Nm=float(np.linalg.norm(np.cross(m.rest+u,forces).sum(0))),
        residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),
        elastic_J=float(state['elastic_J']),spring_J=float(state['spring_J']),
        strain_rms=float(np.sqrt(np.mean(strain**2))),maximum_nodal_displacement_m=float(np.linalg.norm(u,axis=1).max()))


def build(degree,n,r,order,device,grading=False):
    return P1Fixture(n,r['E'],r['K'],r['layout']) if degree==1 else P2Fixture(n,r['E'],r['K'],r['layout'],order=order,device=device,grading=grading)


def run(action,degree,levels,order,device,grading=False,linear_solver='superlu'):
    spec=protocol();r=spec['case'];directory=BASE/'fields';directory.mkdir(exist_ok=True)
    tag=f'p{degree}'+('g' if grading else '')
    previous=None;previous_u=None
    for n in levels:
        path=directory/f'{tag}-n{n}-q{order}-{action}.npz'
        if previous is not None and getattr(previous,'engine',None) is not None:
            previous.engine=None
            import gc,torch
            gc.collect();torch.cuda.empty_cache()
        m=build(degree,n,r,order,device,grading);m.linear_solver=linear_solver;m.progress=True;m.set_action(action)
        if path.exists():
            with np.load(path) as z:u=z['u']
            previous,previous_u=m,u;print('CACHED',path.name,flush=True);continue
        start=time.time();warm=directory/f'{tag}-n{n}-q3-{action}.npz'
        if order==4 and warm.exists():u=np.load(warm)['u'];state=m.solve(u,force_tolerance_N=1e-10)
        elif previous is not None:
            u=query(previous,previous_u,m.rest);state=m.solve(u,force_tolerance_N=1e-10)
        else:
            u=np.zeros_like(m.rest)
            for alpha in [.25,.5,.75,1.]:
                m.set_action(action,alpha);state=m.solve(u,force_tolerance_N=1e-10);u=state['displacement_local_m']
        u=state['displacement_local_m'];result=metrics(m,u,state)
        temporary=path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,u=u,rest=m.rest,elapsed_seconds=time.time()-start,**result)
        temporary.replace(path)
        print('COMPLETE',path.name,'seconds',time.time()-start,'residual',result['residual_N'],flush=True)
        previous,previous_u=m,u


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=ACTIONS,required=True);p.add_argument('--degree',type=int,choices=[1,2],required=True)
    p.add_argument('--levels',type=int,nargs='+',required=True);p.add_argument('--order',type=int,choices=[3,4],default=3)
    p.add_argument('--device',default=None);p.add_argument('--grading',action='store_true');p.add_argument('--linear-solver',choices=['superlu','pardiso'],default='superlu');a=p.parse_args();run(a.action,a.degree,a.levels,a.order,a.device,a.grading,a.linear_solver)

if __name__=='__main__':main()
