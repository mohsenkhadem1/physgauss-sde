"""Corrected prior-library posterior with one persistent Gaussian discrepancy.

This utility loads a COMPLETE observation-independent corrected library only.
It never reads cohort responses, labels, old archive rays/likelihoods/weights,
or a GPU/forward model. A caller supplies one observation and a frozen joint
Gaussian model. All55 categories and three independent seeds remain present.
Finite inverted corrected geometries remain diagnostics, never filtered.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import numpy as np

try:
    from campaign_point_material_reference import mask_probabilities,prefix_indices,summarize
    from campaign_point_prior_posterior import systematic_resample,_stability
    from campaign_joint_gaussian_discrepancy import JointGaussianDiscrepancy
    from censored_range_observation import CensoredRanges
except ImportError:
    from experiments.campaign_point_material_reference import mask_probabilities,prefix_indices,summarize
    from experiments.campaign_point_prior_posterior import systematic_resample,_stability
    from experiments.campaign_joint_gaussian_discrepancy import JointGaussianDiscrepancy
    from experiments.censored_range_observation import CensoredRanges

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(20261112,20261113,20261114)
COUNTS=(256,1024)
PARTICLES=2048
RESAMPLING_SEED=2026120700


def _readonly(a):
    value=np.array(a,copy=True);value.flags.writeable=False;return value


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def _checked(root,record):
    p=Path(record['path'])
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=record['path']:
        raise ValueError('Canonical repository-relative path required')
    p=(root/p).resolve()
    if not p.is_relative_to(root) or _sha(p)!=record['sha256']:
        raise ValueError('Frozen library artifact changed: '+record['path'])
    return p


def _read(path):return json.loads(Path(path).read_text())


def _same_public_hardware(root,first,second):
    """Artifact identity is checked bytes, not the location of equal copies."""
    a,b=_checked(root,first),_checked(root,second)
    return first['sha256']==second['sha256'] and a.read_bytes()==b.read_bytes()


@dataclass(frozen=True)
class CorrectedReference:
    seed:int
    geometry_ids:np.ndarray
    physical_coordinates:np.ndarray
    predicted_visible:np.ndarray
    predicted_residual_m:np.ndarray
    minimum_J:np.ndarray
    nonpositive_tet_frame_count:np.ndarray


@dataclass(frozen=True)
class CorrectedLibraries:
    references:tuple[CorrectedReference,...]
    probabilities:Mapping[int,float]
    counts_per_mask:tuple[int,...]
    summary_record:Mapping[str,str]
    plan_record:Mapping[str,str]


def validate_corrected_arrays(data,probabilities,*,count=1024,prefix=256):
    """Strict member allowlist and integration checks, synthetic-testable.

    No ``weights`` or ``log_likelihood`` access, even if those members exist.
    The saved prior masses are checked against fresh p(mask)/N; conditioning
    recomputes that measure rather than treating saved arrays as posteriors.
    """
    ids=data['geometry_ids'];physical=data['physical_coordinates']
    visible=data['predicted_visible'];residual=data['predicted_residual_m']
    prefix_rows=data['prefix_indices_256'];full_mass=data['prior_mass_1024'];small_mass=data['prior_mass_256']
    minimum=data['minimum_J'];inversions=data['nonpositive_tet_frame_count']
    expected=np.repeat(sorted(probabilities),count);n=len(expected)
    if (ids.dtype!=np.dtype('int64') or not np.array_equal(ids,expected)
            or physical.dtype!=np.dtype('float64') or physical.shape!=(n,4)
            or not np.isfinite(physical).all()
            or (physical<np.array([1500.,400.,.28,.35])).any()
            or (physical>np.array([20000.,850.,.42,.85])).any()
            or visible.dtype!=np.dtype('bool') or visible.shape!=(n,72)
            or residual.dtype!=np.dtype('float64') or residual.shape!=(n,72)
            or not np.isfinite(residual[visible]).all()):
        raise ValueError('Complete ordered corrected prior/material/72ray arrays required')
    rows=prefix_indices(ids,prefix,probabilities)
    expected_mass=np.array([probabilities[int(i)]/count for i in ids])
    expected_small=np.array([probabilities[int(i)]/prefix for i in ids[rows]])
    if (prefix_rows.dtype!=np.dtype('int64') or not np.array_equal(prefix_rows,rows)
            or full_mass.dtype!=np.dtype('float64') or not np.array_equal(full_mass,expected_mass)
            or small_mass.dtype!=np.dtype('float64') or not np.array_equal(small_mass,expected_small)
            or minimum.dtype!=np.dtype('float64') or minimum.shape!=(n,) or not np.isfinite(minimum).all()
            or inversions.dtype!=np.dtype('int64') or inversions.shape!=(n,) or (inversions<0).any()
            or not np.array_equal(minimum<=0,inversions>0)):
        raise ValueError('Exact nested prior mass and complete finite geometry diagnostics required; no filtering/repair')
    return tuple(_readonly(a) for a in (ids,physical,visible,np.where(visible,residual,0.),minimum,inversions))


def load_rotation_libraries(summary_path,summary_sha256,*,root=ROOT):
    """Verify completed summary→plan/source/cache/prior/ALL3 archives.

    No old input archive or static tensor payload is opened. Library summary
    hash comes from the caller's frozen execution gate. All archive hashes are
    checked BEFORE reading the first NPZ member. Source/cache/prior checks use
    repository-relative records and support clean relocation.
    """
    root=Path(root).resolve();summary_record=dict(path=str(summary_path),sha256=summary_sha256)
    summary_file=_checked(root,summary_record);summary=_read(summary_file)
    if (summary.get('status')!='COMPLETED_ALL3_CORRECTED_OBSERVATION_INDEPENDENT_PRIOR_LIBRARIES'
            or summary.get('old_archive_members_read')!=['geometry_ids','physical_coordinates']
            or summary.get('old_ray_or_likelihood_or_posterior_members_read') is not False
            or summary.get('cohort_or_truth_used') is not False or summary.get('posterior_computed') is not False):
        raise ValueError('Completed observation-independent corrected library required')
    plan=_read(_checked(root,summary['plan']))
    if (plan.get('schema')!='point-rotation-observation-independent-prior-library-v0.2'
            or tuple(plan['seeds'])!=SEEDS or tuple(plan['counts_per_mask'])!=COUNTS
            or plan['mask_count']!=55 or plan['particles_per_seed']!=55*1024
            or plan['total_particles']!=3*55*1024
            or plan['allowed_old_archive_members']!=['geometry_ids','physical_coordinates']
            or plan['observations_or_cohort_truth_used'] is not False
            or plan['likelihood_or_posterior_computed'] is not False
            or summary['source_sha256']!=plan['source_sha256']
            or summary_file!=(root/plan['output']/'summary.json').resolve()):
        raise ValueError('Frozen exact55/3seed/256-1024 corrected-library plan required')
    execution=_read(summary_file.parent/'execution.json')
    if execution.get('terminal') is not True or execution.get('completed') is not True:
        raise ValueError('Library execution not terminal complete')
    for name,digest in plan['source_sha256'].items():_checked(root,dict(path=name,sha256=digest))
    cache=_read(_checked(root,plan['cache']))
    if (cache.get('completed') is not True or len(cache['categories'])!=55
            or any(plan['source_sha256'].get(k)!=v for k,v in cache['source_sha256'].items())):
        raise ValueError('Complete frozen-source all55 basis cache required')
    for field in ('public_manifest','public_hardware','public_queries'):_checked(root,plan[field])
    distribution=_read(_checked(root,plan['support_prior']));probabilities=mask_probabilities(distribution)
    if (len(probabilities)!=55 or [r['node_id'] for r in cache['categories']]!=sorted(probabilities)
            or not _same_public_hardware(root,cache['public_hardware'],plan['public_hardware'])
            or _read(_checked(root,cache['distribution']))!=distribution):
        raise ValueError('Library/cache/support prior category or public hardware mismatch')
    records=summary['records']
    if len(records)!=3 or [r['seed'] for r in records]!=list(SEEDS):
        raise ValueError('ALL3 ordered reference seeds required')
    paths=[]
    for seed,record in zip(SEEDS,records):
        path=_checked(root,record['library'])
        if (path!=summary_file.parent/f'corrected-prior-{seed}.npz'
                or record['particles']!=55*1024 or record.get('categories_dropped')!=[]):
            raise ValueError('Complete unchanged corrected prior rows required')
        paths.append(path)
    references=[]
    for seed,record,path in zip(SEEDS,records,paths):
        with np.load(path,allow_pickle=False) as data:arrays=validate_corrected_arrays(data,probabilities)
        ids,physical,visible,residual,minimum,inversions=arrays
        if (hashlib.sha256(ids.tobytes()).hexdigest()!=record['geometry_array_sha256']
                or hashlib.sha256(physical.tobytes()).hexdigest()!=record['physical_array_sha256']
                or float(minimum.min())!=record['minimum_J']
                or int((inversions>0).sum())!=record['particles_with_nonpositive_J']
                or int(inversions.sum())!=record['nonpositive_tet_frame_count']):
            raise ValueError('Archive identity or retained inversion diagnostics disagree with summary')
        references.append(CorrectedReference(seed,*arrays))
    return CorrectedLibraries(tuple(references),MappingProxyType(probabilities),COUNTS,
        MappingProxyType(summary_record),MappingProxyType(dict(summary['plan'])))


def condition_rotation_reference(libraries,observation,joint_model):
    """Correct physical-mixture weights AND shared latent-z conditional state."""
    if not isinstance(libraries,CorrectedLibraries) or not isinstance(joint_model,JointGaussianDiscrepancy):
        raise TypeError('Verified corrected libraries and frozen JointGaussianDiscrepancy required')
    if (tuple(r.seed for r in libraries.references)!=SEEDS or len(libraries.counts_per_mask)!=2
            or tuple(sorted(set(libraries.counts_per_mask)))!=libraries.counts_per_mask):
        raise ValueError('All3 fixed ordered seeds and both nested counts required')
    if not isinstance(observation,CensoredRanges) or observation.visible.shape!=(72,):
        raise ValueError('One supplied acquired72ray observation required')
    seeds,reports=[],[];primary=None
    for library in libraries.references:
        prediction=CensoredRanges(library.predicted_visible,library.predicted_residual_m)
        conditional=joint_model.condition(prediction,observation,rows=np.arange(72,dtype=np.int64))
        ll=conditional.log_likelihood;prefixes=[]
        for count in libraries.counts_per_mask:
            rows=prefix_indices(library.geometry_ids,count,libraries.probabilities)
            report,weights=summarize(library.geometry_ids[rows],library.physical_coordinates[rows],ll[rows],libraries.probabilities)
            report=dict(seed=library.seed,count_per_mask=count,**report);reports.append(report)
            prefixes.append(dict(count_per_mask=count,indices=_readonly(rows),weights=_readonly(weights),summary=report))
        seeds.append(dict(seed=library.seed,log_likelihood=_readonly(ll),prefixes=prefixes))
        if library.seed==SEEDS[0]:primary=(library,conditional,prefixes[-1])
    library,conditional,largest=primary
    selected=systematic_resample(largest['weights'],PARTICLES,RESAMPLING_SEED)
    rows=largest['indices'][selected]
    selected_conditional=conditional.take(rows)
    empirical=dict(reference_seed=SEEDS[0],count_per_mask=max(libraries.counts_per_mask),
        resampling_seed=RESAMPLING_SEED,reference_indices=_readonly(rows),
        geometry_ids=_readonly(library.geometry_ids[rows]),physical_coordinates=_readonly(library.physical_coordinates[rows]),
        weights=_readonly(np.full(PARTICLES,1./PARTICLES)),base_gp_conditional=selected_conditional)
    return dict(seeds=seeds,summaries=reports,
        stability=_stability(reports,[r.seed for r in libraries.references],libraries.counts_per_mask),
        empirical_posterior=empirical,primary_base_gp_conditional=selected_conditional,
        joint_model_fingerprint=joint_model.fingerprint,summary_record=dict(libraries.summary_record),
        plan_record=dict(libraries.plan_record),old_likelihood_or_posterior_reused=False,
        scope='Known point-pin type, corrected-geometry approximate mechanics and shared Gaussian discrepancy; no calibration claim')
