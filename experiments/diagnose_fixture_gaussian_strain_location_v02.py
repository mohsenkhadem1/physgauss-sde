"""Saved-field ONLY location of the unchanged sampled symmetric-strain maximum."""
import json
from types import SimpleNamespace
import numpy as np
from experiments import run_active_fixture_gaussian_nh80_cost_v02 as stage


def locate(n,u):
    t=stage.bank.core._topology(n);v=t['corner_X'][t['corner_tets']];dm=(v[:,1:]-v[:,:1]).transpose(0,2,1);g=np.linalg.inv(dm);g=np.concatenate([-g.sum(1,keepdims=True),g],axis=1)
    b24,_=stage.bank.core.volume_rule(24);b80,_=stage.bank.core.volume_rule(80)
    checks=np.concatenate([np.eye(4),np.array([(np.eye(4)[i]+np.eye(4)[j])/2 for i,j in stage.bank.core.EDGES]),b24,b80]);best=None
    for q,b in enumerate(checks):
        grad=stage.bank.core.shape_gradients(b[None],g);G=np.einsum('eai,eaJ->eiJ',u[t['elements']],grad);eigen=np.linalg.eigvalsh(.5*(G+G.transpose(0,2,1)));flat=int(np.argmax(abs(eigen)));element,principal=np.unravel_index(flat,eigen.shape);value=float(abs(eigen[element,principal]))
        if best is None or value>best['maximum_absolute_symmetric_strain']:
            point=b@v[element];best=dict(maximum_absolute_symmetric_strain=value,signed_symmetric_strain_eigenvalue=float(eigen[element,principal]),material_point_m=point.tolist(),element=int(element),guard_point_index=q,distance_to_nearest_endclamp_m=float(min(point[0],.36-point[0])),sampled_only=True)
    return best


def run():
    base=stage.LOCAL;stage.validate(stage.sha(base/'protocol.json'));e=json.loads((base/'execution.json').read_text());r=json.loads((base/'result-LOCAL.json').read_text())
    if not e['terminal'] or not e['completed'] or e['worker_exit_code']!=0 or e['cleanup_exit_code']!=0 or e['result']!=stage.record(base/'result-LOCAL.json') or len(r['rows'])!=6:raise ValueError('Complete six-state seal required')
    stage.prior.shared.old.absent(e['controller_pid']);stage.prior.shared.old.absent(e['worker_pid']);answer=[]
    for row in r['rows']:
        if row['capture']!=stage.record(stage.ROOT/row['capture']['path']):raise ValueError('Capture hash')
        with np.load(stage.ROOT/row['capture']['path'],allow_pickle=False) as z:d=locate(row['n'],z['displacement_m'])
        if not np.isclose(d['maximum_absolute_symmetric_strain'],row['deformation']['maximum_absolute_symmetric_strain_eigenvalue'],rtol=0,atol=1e-15):raise ValueError('Original maximum reproduced')
        p=np.array(d['material_point_m']);load=next(s for s in stage.bank.load_specs() if s['id']==row['label']);_,lc=stage.bank.gaussian_integrals(load['center_m'][:2]);_,sc=stage.bank.original.gaussian_integrals(stage.bank.layout_spec(row['layout']))
        d.update(n=row['n'],label=row['label'],distance_to_load_truncated_centroid_m=float(np.linalg.norm(p-lc)),distance_to_support_truncated_centroid_m=float(np.linalg.norm(p-sc)),source_capture=row['capture']);answer.append(d)
    path=base/'strain-location-diagnostic.json';stage.write(path,dict(schema='Gaussian-load-strain-location-saved-only-v02',source_result=stage.record(base/'result-LOCAL.json'),source=stage.record(stage.ROOT/'experiments/diagnose_fixture_gaussian_strain_location_v02.py'),rows=answer,new_PDE_or_equilibrium_calls=0,scope='Sampled peak locations only, no gate changes or causal proof. Clamp-edge coincidence can motivate interpretation of refinement growth; it never rescues the original5%maxstrain gate.'))
    print(json.dumps(dict(report=stage.record(path),rows=answer)))

if __name__=='__main__':run()
