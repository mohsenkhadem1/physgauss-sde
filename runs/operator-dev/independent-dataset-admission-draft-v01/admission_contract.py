"""Offline proposed public action contract. No physics, response or recipe API.

This is an executable draft, NOT an admitted production compiler. Its purpose
is to test factorization and clocks before a separately reviewed implementation.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments'))
try:
    from isaac_dataset import envelope
finally:
    sys.path.pop(0)

H=2;S=2;K=1
COMMAND_HZ=120;OUTPUT_HZ=20;SECONDS=3
PROPOSED_COMMON_AMPLITUDES_N=(.01,.02)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def menu(*,public_handle_origins_m,public_menu_seed=971,common_peak_N=.01):
    """Only public hardware and externally randomized menu inputs are accepted.

    Public seed assignment itself must be independent of hidden state and
    checked across the full generated manifest; this function cannot prove its
    caller obtained the seed/amplitude without accessing truth.
    """
    origins=np.asarray(public_handle_origins_m,dtype=float)
    if origins.shape!=(2,3) or not np.isfinite(origins).all():raise ValueError('Both finite public origins required')
    separation=origins[1]-origins[0]
    if np.linalg.norm(separation)<1e-9:raise ValueError('Distinct origins required')
    if isinstance(public_menu_seed,bool) or not isinstance(public_menu_seed,int):raise ValueError('Explicit integer public randomization seed required')
    if common_peak_N not in PROPOSED_COMMON_AMPLITUDES_N:raise ValueError('Only prospectively declared common tiers allowed; neither is physically qualified here')
    rng=np.random.default_rng(public_menu_seed)
    directions=np.eye(3)[rng.permutation(3)]*rng.choice([-1.,1.],(3,1))
    axis=separation/np.linalg.norm(separation)
    first=np.cross(axis,np.eye(3)[np.argmin(np.abs(axis))]);first/=np.linalg.norm(first)
    couple=np.stack([first,np.cross(axis,first)])
    actions=[{'id':'zero','family':'push_pulse','peak_world_N':np.zeros((2,3)).tolist(),'cost':1.}]
    for family in ['push_pulse','pull_hold','shear_oscillation','twist_couple']:
        vectors=couple if family=='twist_couple' else directions
        for i,direction in enumerate(vectors):
            force=np.zeros((2,3));force[0]=common_peak_N*direction
            if family=='twist_couple':force[1]=-force[0]
            actions.append({'id':family+'__direction'+str(i),'family':family,'peak_world_N':force.tolist(),'cost':1.})
    return {'role':'known_actuator_menu_DRAFT_NOT_PHYSICALLY_QUALIFIED','handle_capacity':H,'support_capacity':S,'known_program_capacity':K,
        'seconds':SECONDS,'command_hz':COMMAND_HZ,'output_hz':OUTPUT_HZ,'common_peak_N':common_peak_N,'actions':actions}


def compile_known_action(public_menu,action_id):
    """Fixed120Hz command sampling; refinement must hold THIS history fixed."""
    action=next(row for row in public_menu['actions'] if row['id']==action_id)
    endpoints=np.arange(SECONDS*COMMAND_HZ+1,dtype=float)/COMMAND_HZ
    midpoint=(np.arange(SECONDS*COMMAND_HZ,dtype=float)+.5)/COMMAND_HZ
    values=np.asarray([np.asarray(envelope(action['family'],t,SECONDS))*np.asarray(action['peak_world_N']) for t in midpoint],dtype=np.float32).astype(np.float64)
    return {'command_boundary_times_s':endpoints,'requested_output_times_s':np.arange(61,dtype=float)/OUTPUT_HZ,
        'known_handle_forces_N':values[None],'known_handle_pure_torques_Nm':np.zeros((K,len(midpoint),H,3))}


def held_force_at_private_midpoints(compiled,solver_hz):
    """Force adapter only: no latent support interpolation is claimed here."""
    if isinstance(solver_hz,bool) or not isinstance(solver_hz,int) or solver_hz%COMMAND_HZ:raise ValueError('Positive integer multiple of fixed command clock required')
    if solver_hz<COMMAND_HZ:raise ValueError('Solver may not skip known command discontinuities')
    ids=np.arange(SECONDS*solver_hz)//(solver_hz//COMMAND_HZ)
    return compiled['known_handle_forces_N'][:,ids]


def check_counterfactual_manifest(rows,*,expected_boundary_ids=(0,1,2,3),expected_material_ids=(1500,5000,20000)):
    """Each public H/menu replicate must have equal query bytes across latents.

    Different H may have different public bases. Hidden labels are used ONLY
    by this offline admission checker, never public input/model compilation.
    """
    if not rows:raise ValueError('Nonempty prospective cohort required')
    grouped={};coverage={}
    for row in rows:
        key=(row['public_hardware_id'],row['menu_replicate'])
        signature=(row['menu_sha256'],row['command_clock_sha256'],row['camera_program_sha256'],row['feature_schema_sha256'])
        old=grouped.setdefault(key,signature)
        if old!=signature:raise ValueError('Counterfactual public menu/clock/camera/schema changed with hidden condition')
        if row['known_capacities']!={'H':2,'S':2,'K':1}:raise ValueError('Fixed public capacities required')
        if row.get('public_recipe_fields'):raise ValueError('Recipe/hidden labels/IDs are not neural public features')
        pair=(row['hidden_BC'],row['hidden_material'])
        if pair in coverage.setdefault(key,set()):raise ValueError('Duplicate hidden counterfactual condition')
        coverage[key].add(pair)
    expected={(b,e) for b in expected_boundary_ids for e in expected_material_ids}
    if any(observed!=expected for observed in coverage.values()):raise ValueError('Missing/extra required latent crossproduct; no category dropping')
    return True
