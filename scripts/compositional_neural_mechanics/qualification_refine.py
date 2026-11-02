"""Bounded next-mesh and quadrature checks after n2/n3 development failure."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.compositional_neural_mechanics.mechanics import Q2Mechanics,GeometrySpec
from scripts.compositional_neural_mechanics.materials import teacher_energy
from scripts.compositional_neural_mechanics.qualification import ROOT,sha,write,nonrigid
BASE=ROOT/'runs/compositional-neural-mechanics-v16/mechanics-oracle-qualified'
OUT=ROOT/'runs/compositional-neural-mechanics-v16/mechanics-reference-refinement'


def main():
    if OUT.exists():raise ValueError('Preserve previous reference check')
    prior=json.loads((BASE/'report.json').read_text());plan=json.loads((BASE/'plan.json').read_text());hashes=dict(prior['source_sha256'])
    for p in [Path(__file__),BASE/'report.json',BASE/'plan.json']:hashes[str(p.relative_to(ROOT))]=sha(p)
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('Original diagnostic sources changed')
    write(OUT/'plan.json',dict(scope=__doc__,source_sha256=hashes,mesh_pair=[3,4],quadrature_pair=[3,4],quadrature_action='right_bend',gates=plan['gates'],device='cuda:1'))
    q=np.stack(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij'),-1).reshape(-1,3);rows=[];checks=[];new={};started=time.monotonic()
    for geometry_name,spec in plan['geometries'].items():
        geometry=GeometrySpec(**spec)
        for family in plan['families']:
            def field(q):return q.new_tensor([0.,.2]).expand(*q.shape[:-1],2)
            for order in (3,4):
                model=Q2Mechanics(geometry,n=4,order=order,bulk_energy=lambda F,z:teacher_energy(F,z,family=family),material_latent_field=field,device='cuda:1')
                for action,pose in plan['poses'].items():
                    if order==4 and action!='right_bend':continue
                    row=dict(geometry=geometry_name,family=family,action=action,n=4,order=order)
                    try:
                        state=None
                        for fraction in (.5,1.):
                            model.set_controls(np.array(pose)*fraction);state=model.solve(initial=None if state is None else state['displacement_local_m'],force_tolerance_N=1e-8)
                        field_value=model.sample_material(state['displacement_local_m'],q);w=model.reaction_wrenches(state)
                        data=dict(field=field_value,shape=nonrigid(geometry.map(q),field_value),forces_N=w['forces_N'],torques_Nm=w['torques_Nm'],rest=model.rest,displacement=state['displacement_local_m'])
                        path=OUT/geometry_name/family/f'n4-q{order}'/f'{action}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data);new[(geometry_name,family,action,order)]=data
                        row.update(state='complete',residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),force_balance_N=w['force_balance_N'],artifact=str(path.relative_to(ROOT)),sha256=sha(path))
                        reference=BASE/geometry_name/family/'teacher-n3'/f'{action}.npz' if order==3 else OUT/geometry_name/family/'n4-q3'/f'{action}.npz'
                        if order==3:
                            old=next(r for r in prior['rows'] if r['geometry']==geometry_name and r['family']==family and r['action']==action and r['model']=='teacher' and r['n']==3)
                            if sha(reference)!=old['sha256']:raise ValueError('Earlier reference artifact changed')
                        with np.load(reference) as olddata:
                            delta=float(np.sqrt(np.mean((olddata['field']-data['field'])**2)));force=np.linalg.norm(olddata['forces_N']-data['forces_N'],axis=1);torque=np.linalg.norm(olddata['torques_Nm']-data['torques_Nm'],axis=1)
                        passed=bool(delta<=2e-5 and (force<=1e-4+.005*np.linalg.norm(data['forces_N'],axis=1)).all() and (torque<=1e-5+.005*np.linalg.norm(data['torques_Nm'],axis=1)).all())
                        checks.append(dict(**row,comparison='mesh3to4' if order==3 else 'quadrature3to4',passed=passed,field_rms_mm=1000*delta,force_difference_N=force.tolist(),torque_difference_Nm=torque.tolist(),reference_sha256=sha(reference)))
                    except Exception as error:row.update(state='failed',error=repr(error));checks.append(dict(**row,passed=False))
                    rows.append(row);write(OUT/'progress.json',dict(rows=rows,checks=checks));print(row,flush=True)
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('Source changed during refinement')
    write(OUT/'report.json',dict(source_sha256=hashes,plan_sha256=sha(OUT/'plan.json'),rows=rows,checks=checks,all_gates_pass=all(c['passed'] for c in checks),elapsed_seconds=time.monotonic()-started,
        limitation='Only four worstbend cases checked for quadrature; broader poses/material/support range require protocol-specific qualification'))
if __name__=='__main__':main()
