"""Extract16verified TRAIN engineering captures for a fixed hybrid forecast pilot.

No physics, reserved observations, native responses, fitting or target-based
selection. Targets stay LOCAL; material/support inputs are explicit hypotheses.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FAMILIES=('push_pulse','pull_hold','shear_oscillation','twist_couple')
CASES=('t0000','t0001','t0002','t0003')
HELD={('t0000','pull_hold'),('t0001','shear_oscillation'),('t0002','twist_couple'),('t0003','push_pulse')}
ORIGINAL_AUDIT='runs/operator-dev/authored-stage1-monitor-v01/initial-stage-terminal-audit-v01.json'
D_AUDIT='runs/operator-dev/authored-solver-trajectory-audit-v01/completed-audit.json'
FIXTURE='configs/gaussian_independent_diagnostic_inputs_v01.json'


def resolve(relative):
    value=str(relative);p=Path(value)
    if p.is_absolute() or '..' in p.parts or value!=p.as_posix():raise ValueError('Canonical repository-relative path required')
    result=(ROOT/p).resolve()
    if not result.is_relative_to(ROOT):raise ValueError('Path escapes repository')
    return result


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def bin_known_commands(intervals,forces,times):
    """Causal50ms meanF and centeredfirstmoment/dt², both inN; frame0zero.

    Interval forces are already known actuator commands. No measured motion is
    accepted. Actual interval overlap accounts for compiled/state-clock roundoff.
    """
    intervals=np.asarray(intervals,dtype=np.float64);forces=np.asarray(forces,dtype=np.float64);times=np.asarray(times,dtype=np.float64)
    if (intervals.ndim!=2 or intervals.shape[1]!=2 or forces.shape!=(len(intervals),2,3)
            or times.shape!=(61,) or not np.array_equal(times,np.arange(61)/20)
            or not np.isfinite(intervals).all() or not np.isfinite(forces).all()
            or np.any(intervals[:,1]<=intervals[:,0]) or intervals[0,0]!=0
            or not np.array_equal(intervals[1:,0],intervals[:-1,1]) or abs(intervals[-1,1]-3.)>1e-12):
        raise ValueError('Complete finite contiguous knowncommandhistory and61common20Hz times required')
    result=np.zeros((61,12),dtype=np.float64);coverage=[]
    for index,(start,end) in enumerate(zip(times[:-1],times[1:]),1):
        lo=np.maximum(intervals[:,0],start);hi=np.minimum(intervals[:,1],end)
        widths=np.maximum(hi-lo,0.);dt=end-start;center=(start+end)/2
        if abs(widths.sum()-dt)>1e-12:raise ValueError('Commandhistory does not cover declared output interval')
        result[index,:6]=np.einsum('t,thc->hc',widths,forces).reshape(6)/dt
        result[index,6:]=np.einsum('t,thc->hc',widths*((lo+hi)/2-center),forces).reshape(6)/(dt*dt)
        coverage.append(abs(float(widths.sum()-dt)))
    return result,max(coverage)


def select_sources(original,d_audit):
    """Highestrate among verifiedcompleted original captures only, never errors."""
    selected={}
    for record in original['cells']:
        if record.get('passed') is not True or record.get('state')!='verified':continue
        cell=record['cell'];key=(cell['case'],cell['response'])
        if key[0] not in CASES or key[1] not in FAMILIES:continue
        if key not in selected or cell['hz']>selected[key]['cell']['hz']:selected[key]=record
    if d_audit.get('passed') is not True or d_audit['cell_audit'].get('passed') is not True:raise ValueError('IndependentD trajectory audit must pass')
    d=d_audit['cell_audit']
    if (d['cell']['case'],d['cell']['response'])!=('t0000','twist_couple'):raise ValueError('D replacement must be exactt0twist')
    selected[('t0000','twist_couple')]=d
    if set(selected)!={(case,family) for case in CASES for family in FAMILIES}:raise ValueError('Exactly16verifiedscenariofamilies required')
    return selected


def prepare(output,*,original_audit=ORIGINAL_AUDIT,d_audit=D_AUDIT):
    destination=resolve(output)
    if destination.exists():raise FileExistsError('FreshLOCALdatasetdirectory required')
    original=read(resolve(original_audit));replacement=read(resolve(d_audit));chosen=select_sources(original,replacement)
    canonical=read(resolve(FIXTURE));cases={c['episode_id']:c for c in canonical['cases']}
    wanted={};records=[];shared_tets=shared_faces=None;times=np.arange(61)/20
    def append(key,value):wanted.setdefault(key,[]).append(value)
    for case_id in CASES:
        for family in FAMILIES:
            record=chosen[(case_id,family)];cell=record['cell'];directory=resolve(record['output'])
            for proof in record['source_evidence_records']:
                if sha(resolve(proof['path']))!=proof['sha256']:raise ValueError('Verifiedevidencechanged:'+proof['path'])
            summary=read(directory/'summary.json');execution=read(directory/'execution.json')
            if not execution['terminal'] or not execution['completed'] or not summary['completed']:raise ValueError('Selectedcaptureincomplete')
            if (directory/'failure.json').exists():raise ValueError('Failedcapturecannotenterpilot')
            private=read(directory/'private-provenance/physical-inputs.json');recipe,spec,hardware=private['case'],private['spec'],private['hardware']
            reference=cases[case_id]
            if recipe!=reference['recipe'] or spec!=reference['boundaries'] or recipe['split']!='train':raise ValueError('OnlyexactpublicTRAINstaticinputsallowed')
            with np.load(directory/'private-provenance/static-reference.npz',allow_pickle=False) as raw:
                rest=raw['rest_points_m'].copy();tets=raw['tetrahedra'].copy();faces=raw['surface_faces'].copy()
            for descriptor in reference['reference_static_arrays']:
                array={'rest_points_m':rest,'tetrahedra':tets,'surface_faces':faces}[descriptor['key']]
                if (str(array.dtype)!=descriptor['dtype'] or list(array.shape)!=descriptor['shape']
                        or hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()!=descriptor['sha256']):raise ValueError('Canonicalstaticgeometrychanged')
            if rest.shape!=(455,3):raise ValueError('Full455nodegeometryrequired')
            if shared_tets is None:shared_tets,shared_faces=tets,faces
            elif not np.array_equal(tets,shared_tets) or not np.array_equal(faces,shared_faces):raise ValueError('Commonconnectivityassumptionfailed')
            if sha(directory/'trajectory.npz')!=summary['trajectory_sha256']:raise ValueError('Targettrajectorydigestchanged')
            # Only q is measured motion. Velocities, futureobservations and inferred labels are not features.
            with np.load(directory/'trajectory.npz',allow_pickle=False) as raw:
                q=raw['positions_m'].copy();actual=raw['actual_state_times_s'].copy()
                if not np.array_equal(raw['times_s'],times) or q.shape!=(61,455,3) or not np.isfinite(q).all() or np.max(abs(actual-times))>1e-12:raise ValueError('Fullactual61frameqrequired')
                controls,coverage=bin_known_commands(raw['force_interval_actual_times_s'],raw['force_step_commands_N'],times)
            material=recipe['material'];mask=np.zeros(455,dtype=bool);mask[spec['fixture_nodes']]=True
            centers=np.zeros((2,3));active=np.zeros(2,dtype=bool);count=len(spec['support_centers_m'])
            if not 1<=count<=2:raise ValueError('DeclaredS2supportcapacityrequired')
            centers[:count]=spec['support_centers_m'];active[:count]=True
            offset=float(spec['motion_amplitude_m'])*np.sin(np.pi*np.clip(times/3,0,1))[:,None]**2*np.asarray(spec['motion_direction_world'])
            patches=np.array([h['ordered_patch_nodes'] for h in hardware],dtype=np.int64)
            if patches.shape!=(2,9) or any(len(set(row))!=9 for row in patches):raise ValueError('Exact2complete9nodehandlesrequired')
            values={'rest':rest,'q':q,'controls':controls,'material':np.array([material['youngs_modulus_Pa'],material['density_kg_m3'],material['poissons_ratio'],material['linear_damping_s_inverse']]),
                'support_centers':centers,'support_active':active,'fixture_mask':mask,'support_offsets':offset,
                'handle_patches':patches,'handle_centers':np.array([h['reference_origin_m'] for h in hardware]),
                'handle_masses':np.array([h['mass_kg'] for h in hardware]),'handle_inertias':np.array([h['full_COM_inertia_kg_m2'] for h in hardware]),
                'handle_com_offsets':np.array([h['COM_offset_m'] for h in hardware]),'handle_initial_rotations':np.array([h['initial_rotation_matrix'] for h in hardware]),
                'train_mask':(case_id,family) not in HELD,'ids':case_id+'__'+family}
            for key,value in values.items():append(key,value)
            records.append({'id':case_id+'__'+family,'source_output':record['output'],'source_cell_id':cell['id'],'source_physics_hz_AUDIT_ONLY':cell['hz'],
                'source_trajectory_sha256':summary['trajectory_sha256'],'source_role':'newmetric2diagnostic' if (case_id,family)==('t0000','twist_couple') else 'verifiedcompletedoriginalengineeringcapture',
                'split':'fit' if values['train_mask'] else 'held_cross_action','maximum_knowncommand_interval_coverage_error_s':coverage})
    arrays={key:np.asarray(value) for key,value in wanted.items()};arrays.update(tets=shared_tets,faces=shared_faces,times=times)
    if arrays['train_mask'].dtype!=bool or int(arrays['train_mask'].sum())!=12:raise ValueError('Frozen12fit/4heldsplitfailed')
    destination.mkdir(parents=True,exist_ok=False);np.savez_compressed(destination/'dataset.npz',**arrays)
    manifest={'schema':'known-hypothesis-hybrid-forecast-pilot-v0.1','role':'TRAINengineeringcapturedatasetonly;nottemporal/productionqualification',
        'source_builder_sha256':sha(Path(__file__)),'dataset_sha256':sha(destination/'dataset.npz'),'raw_dataset_LOCAL':True,
        'source_audits':[{'path':str(p),'sha256':sha(resolve(p))} for p in [original_audit,d_audit]],'canonical_fixture_sha256':sha(resolve(FIXTURE)),
        'records':records,'arrays':{key:{'shape':list(value.shape),'dtype':str(value.dtype)} for key,value in arrays.items()},
        'information_roles':{'rest_tets_faces_handles':'publicknownrest/hardware','controls_times':'knownplannedactuatorcommands,causalintervalfeatures',
            'material_support_fixture_inputs':'known_parameter_training_or_diagnostic_hypotheses','q':'measuredsimulatedtargetONLY','ids':'AUDIT_ONLYnotneuralfeature','train_mask':'frozenexperimentalroleONLY'},
        'feature_allowlist':[k for k in arrays if k not in ['q','ids','train_mask']],
        'control_formula':'Atframej>0,over[t[j-1],t[j]]: first6=integralFdt/dt; last6=integral((t-midpoint)*F)dt/dt^2. BothunitsN. Initial12channelszero. Piecewiseheldactualcommandintervaloverlaps; nofuturemeasuredfeature.',
        'material_order':['E_Pa','rho_kg_m3','nu','body_drag_s_inverse'],'fit_cases':12,'held_cases':4,
        'split_design':'FixedHgroupssharedacrossactions;held(t0,pull),(t1,shear),(t2,twist),(t3,push). Noheldscore/data-dependentselection. No unseenlayoutclaim.',
        'source_selection':'HighestavailableVERIFIEDcompletedoriginalcaptureperpair;newauditedDonlyt0twist. Neverchoosefromtargeterrors.',
        'known_correlations':'EngineeringactionamplitudesandchosenratescorrelatewithH/source. Actualknowncommandsretainexecuteddiscretization. NoexplicitHz/familyonehot/IDneuralfeatures; thisisnotproductionlatent-safeadmission.',
        'normalization':'Modelmustusedeclaredconstants; builderdoesnotnormalizetotruetrajectoryamplitude.',
        'actual_force_envelope_qualified':False,'production_dataset_admitted':False,'unseen_layout_test':False,'prediction_qualified':False}
    write(destination/'manifest.json',manifest);return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/gaussian-campaign-v01/hybrid-pilot-v01');a=p.parse_args()
    result=prepare(a.output);print(json.dumps({'fit_cases':result['fit_cases'],'held_cases':result['held_cases'],'dataset_sha256':result['dataset_sha256']}))
