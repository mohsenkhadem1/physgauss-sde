"""Additional documented fixed-corotated proxy; never native energy readback.

Preserves the original common-corotated audit utility and outputs. PhysX5.8
public documentation supplies the determinant-based volumetric formula; that
source does not prove the exact constitutive implementation of this runtime.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
try:
    import campaign_native_energy_audit as common
except ImportError:
    from experiments import campaign_native_energy_audit as common

FORMULA_SOURCE='https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/DeformableVolume.html#constitutive-model'


def fixed_corotated_energy(rest,tets,positions,young,poisson):
    rest=np.asarray(rest,dtype=float);tets=np.asarray(tets,dtype=int);q=np.asarray(positions,dtype=float)
    if not (young>0 and -1<poisson<.5):raise ValueError('Invalid material')
    if q.ndim!=3 or q.shape[1:]!=rest.shape or tets.ndim!=2 or tets.shape[1]!=4:raise ValueError('Aligned frame/node/xyz and tetrahedron arrays required')
    if not np.isfinite(rest).all() or not np.isfinite(q).all():raise ValueError('Finite states required')
    Dm=(rest[tets[:,1:]]-rest[tets[:,:1]]).transpose(0,2,1);det0=np.linalg.det(Dm)
    if np.any(det0<=0):raise ValueError('Positive reference tetrahedra required')
    F=(q[:,tets[:,1:]]-q[:,tets[:,:1]]).transpose(0,1,3,2)@np.linalg.inv(Dm)
    J=np.linalg.det(F)
    if np.any(J<=0):raise ValueError('Inverted state; no energy interpretation')
    singular=np.linalg.svd(F,compute_uv=False)
    mu=young/(2*(1+poisson));lam=young*poisson/((1+poisson)*(1-2*poisson))
    arap=mu*np.sum((singular-1)**2,axis=-1)
    fixed=arap+.5*lam*(J-1)**2
    return np.sum(fixed*(det0/6),axis=-1)


def audit(private):
    private=Path(private);original=common.audit(private)
    with np.load(private/'reference.npz') as a:rest,tets=a['rest_points_m'],a['tetrahedra']
    with np.load(private/'evaluation/trajectory.npz') as a:q=a['positions_m']
    material=json.loads((private/'truth/recipe.json').read_text())['material']
    U=fixed_corotated_energy(rest,tets,q,material['youngs_modulus_Pa'],material['poissons_ratio'])
    original['declared_fixed_corotated_elastic_energy_J']=U.tolist()
    original['additional_fixed_corotated_formula']='mu*||F-R||_F^2 + lambda/2*(det(F)-1)^2, integrated with authored reference tetrahedron volumes'
    original['installed_backend_formula_verified']=False
    original['fixed_corotated_qualification']='Additional proxy from documentedPhysX5.8 formula, not native energy readback or verified implementation identity for installedIsaac6.2. Original common-corotated and fixed-corotated curves both retained. Actual engine nodal mass and constraint energy remain unverified.'
    return original


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing derived audit')
    result={'source_sha256':common.sha(__file__),'original_common_proxy_source_sha256':common.sha(common.__file__),
        'formula_source':FORMULA_SOURCE,'cases':[audit(path) for path in args.private]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cases':len(result['cases']),'output':str(args.output)}))
