"""Last bounded teacher reference mesh check n4→6; no tolerance relaxation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.compositional_neural_mechanics.mechanics import Q2Mechanics,GeometrySpec
from scripts.compositional_neural_mechanics.materials import teacher_energy
from scripts.compositional_neural_mechanics.qualification import ROOT,sha,write,nonrigid
BASE=ROOT/'runs/compositional-neural-mechanics-v16/mechanics-reference-refinement'
ORIGINAL=ROOT/'runs/compositional-neural-mechanics-v16/mechanics-oracle-qualified'


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',choices=['rectangular','tapered_arch'],required=True);p.add_argument('--device',choices=['cuda:1','cuda:2'],required=True);a=p.parse_args()
    out=ROOT/'runs/compositional-neural-mechanics-v16/mechanics-reference-fine'/a.geometry
    if out.exists():raise ValueError('Preserve previous fine check')
    prior=json.loads((BASE/'report.json').read_text());plan=json.loads((ORIGINAL/'plan.json').read_text());hashes=dict(prior['source_sha256'])
    mechanics='scripts/compositional_neural_mechanics/mechanics.py';oldhash=hashes[mechanics]
    archive=ROOT/'reports/compositional-neural-mechanics-v16/source-snapshots'/oldhash
    if sha(archive)!=oldhash:raise ValueError('Previous mechanics source snapshot missing')
    hashes[mechanics]=sha(ROOT/mechanics)
    for path in (Path(__file__),BASE/'report.json',ORIGINAL/'plan.json',ROOT/'tests/test_compositional_mechanics.py',archive):hashes[str(path.relative_to(ROOT))]=sha(path)
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('Prior diagnostic source changed beyond documented optional footprint extension')
    write(out/'plan.json',dict(scope=__doc__,source_sha256=hashes,previous_mechanics_sha256=oldhash,
        source_amendment='Optional differentiable interface geometry, disabled for all teacher checks; verify existing n4 equilibrium residuals before each fine solve',
        mesh_pair=[4,6],gates=plan['gates'],device=a.device,geometry=a.geometry))
    geometry=GeometrySpec(**plan['geometries'][a.geometry]);q=np.stack(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij'),-1).reshape(-1,3);rows=[];started=time.monotonic()
    for family in plan['families']:
        def latent(q):return q.new_tensor([0.,.2]).expand(*q.shape[:-1],2)
        energy=lambda F,z:teacher_energy(F,z,family=family)
        coarse=Q2Mechanics(geometry,n=4,bulk_energy=energy,material_latent_field=latent,device=a.device)
        fine=Q2Mechanics(geometry,n=6,bulk_energy=energy,material_latent_field=latent,device=a.device)
        for action,pose in plan['poses'].items():
            row=dict(geometry=a.geometry,family=family,action=action,n=6)
            try:
                reference=BASE/a.geometry/family/'n4-q3'/f'{action}.npz';oldrow=next(r for r in prior['rows'] if r['geometry']==a.geometry and r['family']==family and r['action']==action and r['order']==3)
                if sha(reference)!=oldrow['sha256']:raise ValueError('Prior n4artifactchanged')
                with np.load(reference) as data:old={k:data[k].copy() for k in data.files}
                coarse.set_controls(pose);check=coarse.evaluate(old['displacement']);oldres=float(np.linalg.norm(check['residual']))
                if oldres>1e-8:raise ValueError('Optional-source extension changes old n4equilibrium')
                state=None
                for fraction in (.5,1.):
                    fine.set_controls(np.asarray(pose)*fraction);state=fine.solve(initial=None if state is None else state['displacement_local_m'],force_tolerance_N=1e-8)
                field=fine.sample_material(state['displacement_local_m'],q);w=fine.reaction_wrenches(state)
                path=out/family/f'{action}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,field=field,shape=nonrigid(geometry.map(q),field),forces_N=w['forces_N'],torques_Nm=w['torques_Nm'],rest=fine.rest,displacement=state['displacement_local_m'])
                delta=float(np.sqrt(np.mean((field-old['field'])**2)));force=np.linalg.norm(w['forces_N']-old['forces_N'],axis=1);torque=np.linalg.norm(w['torques_Nm']-old['torques_Nm'],axis=1)
                passed=bool(delta<=2e-5 and (force<=1e-4+.005*np.linalg.norm(w['forces_N'],axis=1)).all() and (torque<=1e-5+.005*np.linalg.norm(w['torques_Nm'],axis=1)).all())
                row.update(state='complete',passed=passed,field_rms_mm=1000*delta,force_difference_N=force.tolist(),torque_difference_Nm=torque.tolist(),residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),force_balance_N=w['force_balance_N'],old_n4_residual_N=oldres,reference_sha256=sha(reference),artifact=str(path.relative_to(ROOT)),sha256=sha(path))
            except Exception as error:row.update(state='failed',passed=False,error=repr(error))
            rows.append(row);write(out/'progress.json',dict(rows=rows));print(row,flush=True)
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('Sources changed during fine check')
    write(out/'report.json',dict(source_sha256=hashes,plan_sha256=sha(out/'plan.json'),rows=rows,all_gates_pass=all(r['passed'] for r in rows),elapsed_seconds=time.monotonic()-started,
        limitation='This is the last bounded analytic teacher reference pair; learned/nonlinearinterface andscene-range qualification remain separate'))
if __name__=='__main__':main()
