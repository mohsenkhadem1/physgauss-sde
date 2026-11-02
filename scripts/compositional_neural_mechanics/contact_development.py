"""Bounded CPU contact development; no Stage A core edits or reserved cases."""
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from scripts.compositional_neural_mechanics.contact_mechanics import ContactQ2Mechanics,RigidPlane,RigidSphere,dense_surface_gap_check
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/compositional-neural-mechanics-v16/contact-development'
REPORT=ROOT/'reports/compositional-neural-mechanics-v16/contact-development.json'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    source=['scripts/compositional_neural_mechanics/'+p for p in ('mechanics.py','contact_mechanics.py','contact_development.py')]
    plan=dict(scope='Development only: fixed known geometry, normal penalty, quasistatic',source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source},
              cases=['gravity_plane','sphere_indentation','gravity_plane_sphere_indentation'],quadrature=[4,8,16],base_n=2,mesh_companion_n=3,
              base_penalty_N_m3=1e7,penalty_companion_N_m3=2e7,budget_seconds=1200,
              gates=dict(field_coordinate_rms_m=2e-5,reaction_absolute_N=.005,reaction_relative=.01,dense_depth_change_m=2e-5,
                         residual_N=1e-8,force_balance_N=1e-7,torque_balance_Nm=1e-7,minimum_J=.1),
              note='Penalty doubling is sensitivity to a changed model, not an exact-contact convergence certificate. Dense grids are not certified global minima.')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    start=time.monotonic();rows=[]
    q=np.stack(np.meshgrid(*[np.linspace(0,1,9)]*3,indexing='ij'),-1).reshape(-1,3)
    for case in plan['cases']:
        for n,order,k in [(2,o,1e7) for o in (4,8,16)]+[(2,16,2e7),(3,16,1e7)]:
            if time.monotonic()-start>1160:raise TimeoutError('Development stage budget reached')
            key=f'{case}-n{n}-q{order}-k{k:g}';row=dict(key=key,case=case,n=n,surface_order=order,penalty_N_m3=k)
            t=time.monotonic()
            try:
                colliders=[];gravity=(0,0,-9.81) if 'gravity' in case else (0,0,0)
                if 'plane' in case:colliders.append(RigidPlane(point=(0,0,-.04001),penalty_N_m3=k))
                # Prescribed sphere translates from separated z=.09 to z=.080 (5mm geometric intrusion).
                has_sphere='sphere' in case
                if has_sphere:colliders.append(RigidSphere(center=(.12,0,.09),radius=.045,penalty_N_m3=k))
                model=ContactQ2Mechanics(n=n,surface_order=order,colliders=colliders,gravity=gravity)
                state=None
                for z in ([.09,.0875,.085,.0825,.080] if has_sphere else [None]):
                    if has_sphere:model.colliders=tuple([c for c in model.colliders if isinstance(c,RigidPlane)]+[RigidSphere(center=(.12,0,z),radius=.045,penalty_N_m3=k)])
                    state=model.solve(initial=None if state is None else state['displacement_local_m'],force_tolerance_N=1e-8)
                u=state['displacement_local_m'];w=model.reaction_wrenches(state);dense=dense_surface_gap_check(model,u)
                field=model.sample_material(u,q);reaction=np.concatenate([w['forces_N'].ravel(),*[c['force_on_rigid_N'] for c in w['contact_reactions']]])
                np.savez_compressed(OUT/(key+'.npz'),u=u,field=field,reaction=reaction)
                row.update(completed=True,residual_N=float(np.linalg.norm(state['residual'])),minimum_J=float(state['J'].min()),force_balance_N=w['force_balance_N'],torque_balance_Nm=w['world_torque_balance_Nm'],
                           dense=dense,max_dense_penetration_m=max(d['max_dense_penetration_m'] for d in dense),
                           max_quadrature_penetration_m=max(c['max_sampled_penetration_m'] for c in w['contact_reactions']))
                row['equilibrium_gate']=row['residual_N']<1e-8 and row['minimum_J']>.1 and row['force_balance_N']<1e-7 and row['torque_balance_Nm']<1e-7 and not any(d['missed_contact'] for d in dense)
            except Exception as exc:row.update(completed=False,error=str(exc),traceback=traceback.format_exc())
            row['seconds']=time.monotonic()-t;rows.append(row);(OUT/(key+'.json')).write_text(json.dumps(row,indent=2)+'\n')
            print(key,row['completed'],flush=True)
    comparisons=[]
    for case in plan['cases']:
        def find(n,o,k):return next(r for r in rows if r['case']==case and (r['n'],r['surface_order'],r['penalty_N_m3'])==(n,o,k))
        for kind,a,b in [('quadrature4_8',find(2,4,1e7),find(2,8,1e7)),('quadrature8_16',find(2,8,1e7),find(2,16,1e7)),('mesh2_3',find(2,16,1e7),find(3,16,1e7)),('penalty_sensitivity',find(2,16,1e7),find(2,16,2e7))]:
            r=dict(case=case,kind=kind,a=a['key'],b=b['key'],passed=False)
            if a['completed'] and b['completed']:
                x=np.load(OUT/(a['key']+'.npz'));y=np.load(OUT/(b['key']+'.npz'))
                r.update(field_coordinate_rms_m=float(np.sqrt(np.mean((x['field']-y['field'])**2))),reaction_difference_N=float(np.linalg.norm(x['reaction']-y['reaction'])),
                         reaction_tolerance_N=.005+.01*float(np.linalg.norm(y['reaction'])),dense_penetration_change_m=abs(a['max_dense_penetration_m']-b['max_dense_penetration_m']))
                r['passed']=bool(a['equilibrium_gate'] and b['equilibrium_gate'] and r['field_coordinate_rms_m']<=2e-5 and r['reaction_difference_N']<=r['reaction_tolerance_N'] and r['dense_penetration_change_m']<=2e-5)
                if kind=='penalty_sensitivity':r['passed']=None;r['note']='Changed penalty model; no pass/fail declared.'
            comparisons.append(r)
    tiny=[]
    for order in (4,8,16):
        m=ContactQ2Mechanics(n=1,surface_order=order,colliders=(RigidSphere(center=(.12,0,-.07),radius=.031),))
        u=np.zeros_like(m.rest);_,_,_,c=m.contact_response(m.engine.tensor(u))
        tiny.append(dict(order=order,dense=dense_surface_gap_check(m,u)[0],reaction_N=c[0]['force_on_rigid_N'].tolist(),sampled_depth_m=c[0]['max_sampled_penetration_m']))
    report=dict(plan=plan,states=rows,comparisons=comparisons,tiny_patch=tiny,all_numerical_gates_pass=all(c['passed'] for c in comparisons if c['kind']!='penalty_sensitivity'),
                total_seconds=time.monotonic()-start,scope_limit='Known fixed plane/sphere geometry; no friction, dynamic impact, unseen geometry, or exact contact guarantee.')
    REPORT.write_text(json.dumps(report,indent=2)+'\n');print('REPORT',REPORT,flush=True)

if __name__=='__main__':main()
