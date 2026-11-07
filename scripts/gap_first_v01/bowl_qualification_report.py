"""Summarize retained development checks without promoting them to test results."""
import json,hashlib
from pathlib import Path
import numpy as np
from .bowl_campaign import common_nodes,target
from .bowl_audit import audit_file

def point_rms(a,b):
    return float(np.sqrt(np.mean(np.sum((a-b).reshape(-1,3)**2,axis=1)))*1000)

def main():
    directory=Path('runs/gap-first-v01/bowl');root=Path('reports/gap-first-v01/bowl')
    result=dict(scope='Development numerical qualification, not held-out benchmark outcomes',
        decision='Proceed as a bounded nonlinear contact approximation and explicit model-discretization stress test; no continuum-truth, exact-contact, or real-physics calibration claim',
        geometry='Identical piecewise-linear tapered curved solid across refinements, mass0.12809501019059244kg; known open hemispherical bowl with no rim thickness',
        checks={},files={},limitations=['No continuous collision detection or self-contact','Penalty permits small penetration','Regularized friction has no exact stiction','Known geometry, density, viscosity and prescribed rigid motion','Reference mesh is not a continuum convergence certificate'])
    for young in (2500,10000):
        p=directory/f'refinement-development-n6-E{young}.npz';q=directory/f'refinement-development-n12-E{young}.npz'
        a,b=dict(np.load(p)),dict(np.load(q))
        result['checks'][f'E{young}']=dict(mesh_target_point_rms_difference_mm=point_rms(target(a,common_nodes(6)),target(b,common_nodes(12))),
            inference_min_j=float(a['min_j']),reference_min_j=float(b['min_j']),inference=audit_file(p),reference=audit_file(q))
        for file in (p,q):result['files'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    p=directory/'refinement-development-n6-E2500.npz';q=directory/'refinement-development-n6-halfdt.npz'
    a,b=dict(np.load(p)),dict(np.load(q))
    result['checks']['half_dt']=dict(target_point_rms_difference_mm=point_rms(target(a,common_nodes(6)),target(b,common_nodes(6))),dt_s=float(a['dt']),refined_dt_s=float(b['dt']))
    result['gpu_equation_parity']=json.loads((root/'gpu-parity.json').read_text())
    result['gpu_boxcar_parity']=json.loads((root/'gpu-boxcar-parity.json').read_text())
    result['adapter_qualification']='PhysTwin spring-force core adapter failed positive-volume validity across CPU/GPU and timestep refinement; no native-method accuracy ranking'
    (root/'numerical-qualification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['checks'],indent=2))

if __name__=='__main__':main()
