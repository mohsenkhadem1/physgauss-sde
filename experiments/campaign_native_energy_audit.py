"""Declared continuum energy proxies and exact held-force translation work.

These are not native PhysX energy buffers: actual engine nodal masses, rigid
angular kinetic energy, and internal constraint energies are unavailable here.
Use only fixed-fixture TRAIN diagnostics, never a conservation certificate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def held_force_work(forces, before_positions, final_positions, frame_steps):
    """For each held world force, work is F dot (x_after-x_before)."""
    f=np.asarray(forces,dtype=float);p=np.asarray(before_positions,dtype=float)
    final=np.asarray(final_positions,dtype=float);steps=np.asarray(frame_steps)
    if f.shape!=p.shape or f.ndim!=3 or f.shape[-1]!=3 or final.shape!=p.shape[1:]:
        raise ValueError('Aligned step/handle/xyz arrays required')
    if not np.isfinite(f).all() or not np.isfinite(p).all() or not np.isfinite(final).all():
        raise ValueError('Finite force/positions required')
    if not np.issubdtype(steps.dtype,np.integer) or np.any(steps<0) or np.any(steps>len(f)):
        raise ValueError('Captured frames require integer step indices')
    after=np.concatenate([p[1:],final[None]],axis=0)
    increments=np.sum(f*(after-p),axis=(1,2))
    cumulative=np.concatenate([[0.],np.cumsum(increments)])
    # An indicative coordinate-roundoff bound, assuming each logged position
    # represents rounding to float32. This excludes solver and force errors.
    ulp_before=np.abs(np.spacing(p.astype(np.float32))).astype(float)
    ulp_after=np.abs(np.spacing(after.astype(np.float32))).astype(float)
    bound=np.sum(np.abs(f)*(.5*ulp_before+.5*ulp_after))
    return cumulative[steps],float(bound)


def continuum_proxies(rest,tets,positions,velocities,young,poisson,density):
    """Common corotated energy density and declared lumped body KE only.

    See PhysX5.8 DeformableVolume constitutive-model documentation. Its
    commonly-used formula is not asserted to equal this installed backend.
    """
    rest=np.asarray(rest,dtype=float);ids=np.asarray(tets,dtype=int)
    q=np.asarray(positions,dtype=float);v=np.asarray(velocities,dtype=float)
    if not (young>0 and -1<poisson<.5 and density>0):raise ValueError('Invalid material')
    if q.shape!=v.shape or q.ndim!=3 or q.shape[1:]!=rest.shape:
        raise ValueError('Aligned frame/node/xyz arrays required')
    if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Finite states required')
    Dm=(rest[ids[:,1:]]-rest[ids[:,:1]]).transpose(0,2,1)
    det=np.linalg.det(Dm)
    if np.any(det<=0):raise ValueError('Positive reference tetrahedra required')
    volume=det/6
    Ds=(q[:,ids[:,1:]]-q[:,ids[:,:1]]).transpose(0,1,3,2)
    F=Ds@np.linalg.inv(Dm)
    if np.any(np.linalg.det(F)<=0):raise ValueError('Inverted state; no energy interpretation')
    singular=np.linalg.svd(F,compute_uv=False)
    mu=young/(2*(1+poisson));lam=young*poisson/((1+poisson)*(1-2*poisson))
    energy_density=mu*np.sum((singular-1)**2,axis=-1)+.5*lam*np.sum(singular-1,axis=-1)**2
    elastic=np.sum(energy_density*volume,axis=-1)
    mass=np.zeros(len(rest))
    np.add.at(mass,ids.ravel(),np.repeat(density*volume/4,4))
    kinetic=.5*np.sum(mass[None,:,None]*v*v,axis=(1,2))
    return elastic,kinetic,float(mass.sum())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(private):
    private=Path(private);case_path=private/'truth/recipe.json';spec_path=private/'truth/boundaries.json'
    case=json.loads(case_path.read_text());spec=json.loads(spec_path.read_text())
    if (case['split'],case['episode_id'],case['boundary_type'])!=('train','t0001','fixed_patch') or spec['motion_amplitude_m']!=0:
        raise ValueError('Only fixed, stationary-fixture TRAIN t0001 diagnostic allowed')
    audit=json.loads((private/'validation.json').read_text())
    if not audit.get('passed'):raise ValueError('Full QC required before proxy analysis')
    reference=private/'reference.npz';trajectory=private/'evaluation/trajectory.npz'
    with np.load(reference) as a:rest,tets=a['rest_points_m'],a['tetrahedra']
    with np.load(trajectory) as a:data={k:a[k].copy() for k in a.files}
    hz=case['physics_hz'];times=data['times_s'];steps=np.rint(times*hz).astype(int)
    if not np.allclose(times,steps/hz,atol=1e-12,rtol=0):raise ValueError('Capture steps misaligned')
    if len(data['force_step_commands_N'])!=round(case['seconds']*hz):raise ValueError('Incomplete force log')
    work,bound=held_force_work(data['force_step_commands_N'],data['force_step_application_positions_m'],data['handle_positions_m'][-1],steps)
    m=case['material']
    U,T,mass=continuum_proxies(rest,tets,data['positions_m'],data['velocities_m_s'],m['youngs_modulus_Pa'],m['poissons_ratio'],m['density_kg_m3'])
    return {'private_directory':str(private),'physics_hz':hz,'times_s':times.tolist(),
        'held_force_translation_work_J':work.tolist(),'indicative_float32_position_work_roundoff_bound_J':bound,
        'declared_corotated_elastic_energy_J':U.tolist(),'declared_lumped_body_kinetic_energy_J':T.tolist(),
        'declared_body_mass_kg':mass,
        'inputs_sha256':{str(p):sha(p) for p in [case_path,spec_path,reference,trajectory,private/'validation.json']},
        'qualification':'Independent declared continuum proxies, not native total energy. Actual engine nodal mass and rigid kinetic/constraint energy unverified; no energy-conservation verdict. Fixed fixture does no prescribed work; zero external torque on handle origins. Roundoff bound is indicative and excludes engine state/force errors.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--private',type=Path,action='append',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing energy audit')
    result={'source_sha256':sha(__file__),'formula_source':'https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/DeformableVolume.html#constitutive-model','cases':[audit(p) for p in args.private]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cases':len(result['cases']),'output':str(args.output)}))
