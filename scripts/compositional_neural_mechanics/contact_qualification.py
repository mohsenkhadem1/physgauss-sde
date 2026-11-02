"""Small CPU operator evidence; explicitly NOT contact-scene qualification."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.contact_mechanics import ContactQ2Mechanics,RigidPlane,RigidSphere
ROOT=Path(__file__).resolve().parents[2]


def main():
    records=[]
    for name,collider,gravity in [
        ('gravity_plane',RigidPlane(point=(0,0,-.04001)),(0,0,-9.81)),
        ('broad_sphere',RigidSphere(center=(.12,0,-.07),radius=.045),(0,0,0))]:
        model=ContactQ2Mechanics(n=1,colliders=(collider,),gravity=gravity)
        state=model.solve(force_tolerance_N=1e-8);w=model.reaction_wrenches(state);c=w['contact_reactions'][0]
        records.append(dict(name=name,n=1,volume_order=3,surface_order=4,penalty_N_m3=collider.penalty_N_m3,
                            residual_norm_N=float(np.linalg.norm(state['residual'])),min_J=float(state['J'].min()),
                            force_balance_N=w['force_balance_N'],world_torque_balance_Nm=w['world_torque_balance_Nm'],
                            rigid_force_N=c['force_on_rigid_N'].tolist(),rigid_torque_Nm=c['torque_on_rigid_Nm'].tolist(),
                            max_sampled_penetration_m=c['max_sampled_penetration_m'],active_samples=c['active_quadrature_samples']))
    missed=[]
    for order in (4,6,8):
        m=ContactQ2Mechanics(n=1,surface_order=order,colliders=(RigidSphere(center=(.12,0,-.07),radius=.031),))
        _,_,_,r=m.contact_response(torch.zeros_like(m.engine.rest));c=r[0]
        missed.append(dict(surface_order=order,active_samples=c['active_quadrature_samples'],sampled_penetration_m=c['max_sampled_penetration_m'],
                           known_surface_center_penetration_m=.001,rigid_force_N=c['force_on_rigid_N'].tolist()))
    files=['scripts/compositional_neural_mechanics/contact_mechanics.py','scripts/compositional_neural_mechanics/contact_qualification.py',
           'scripts/compositional_neural_mechanics/mechanics.py','tests/test_compositional_contact.py']
    out=dict(scope='CPU operator/equilibrium smoke tests, not scene qualification',source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
             equilibrium_examples=records,small_contact_patch_counterexample=missed,
             limitations=['All six mapped reference boundary faces are sampled at deformed material positions.',
                          'Finite normal penalty allows penetration; no friction, collision detection or dynamic rigid response.',
                          'Small sphere intersection can be missed entirely by surface Gauss points. A zero sampled penetration is not evidence of nonintersection.',
                          'Sphere tangential penalty Hessian can be negative; no universal convexity claim.',
                          'Actual scene use requires mesh/surface quadrature/penalty refinement and reaction as well as shape checks.'],
             scene_qualified=False)
    path=ROOT/'reports/compositional-neural-mechanics-v16/contact-operator-development.json';path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
