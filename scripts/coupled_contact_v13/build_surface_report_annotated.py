"""Reporting-only annotations around the frozen builder; no simulation or selection.

Execution retains every original audit/video guard and requires completed,
source-matched auxiliary summaries. Render and visually inspect the resulting
PDF before delivery. This wrapper does not run auxiliary analyses itself.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/coupled-surface-v13'
BUILDER = ROOT/'scripts/coupled_contact_v13/build_surface_report.py'
EXTRAS = ['library-coverage-audit.json', 'primary-cluster-intervals.json']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_repo_path(relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise RuntimeError('Auxiliary evidence path must stay in repository')
    return ROOT/path


def annotations():
    protocol = json.loads((REPORT/'protocol.json').read_text())
    freeze = json.loads((REPORT/'method-freeze.json').read_text())
    expected_builder = freeze['source_sha256'][str(BUILDER.relative_to(ROOT))]
    if digest(BUILDER) != expected_builder:
        raise RuntimeError('Frozen report builder changed')
    coverage = json.loads((REPORT/EXTRAS[0]).read_text())
    if coverage.get('passed') is not True or coverage.get('errors'):
        raise RuntimeError('Completed successful library audit required')
    for name, key in [('protocol.json', 'protocol_sha256'), ('method-freeze.json', 'method_freeze_sha256')]:
        if coverage[key] != digest(REPORT/name):
            raise RuntimeError('Library audit source is stale: '+name)
    if coverage['audit_source_sha256'] != digest(ROOT/'scripts/coupled_contact_v13/library_coverage_audit.py'):
        raise RuntimeError('Library audit script changed')
    if coverage['frozen_source_sha256'] != freeze['source_sha256']:
        raise RuntimeError('Library audit frozen-source inventory differs')
    if not all(value is True for value in coverage['frozen_source_checks'].values()):
        raise RuntimeError('Library source checks failed')
    for relative, expected in coverage['frozen_source_sha256'].items():
        if digest(checked_repo_path(relative)) != expected:
            raise RuntimeError('Frozen source changed: '+relative)
    labels = {'fem': 'nonlinear FEM', 'native': 'adapted PhysTwin spring core', 'rigid': 'rigid contact'}
    descriptions = []
    for family, label in labels.items():
        summary = coverage['families'][family]
        candidates = [row for row in coverage['candidates'] if row['family'] == family]
        if len(candidates) != summary['expected'] or summary['present'] != summary['expected']:
            raise RuntimeError('Incomplete calibration library accounting')
        if len({tuple(row['index']) for row in candidates}) != len(candidates):
            raise RuntimeError('Duplicate library candidates')
        valid = sum(row['valid_completed'] is True for row in candidates)
        failed = sum(row['completed'] is not True for row in candidates)
        if valid != summary['valid_completed'] or failed != summary['failed'] or valid+failed != summary['expected']:
            raise RuntimeError('Calibration counts do not match audited candidates')
        for row in candidates:
            if row['audit_errors']:
                raise RuntimeError('Library candidate audit error')
            for relative, expected in row['file_sha256'].items():
                if digest(checked_repo_path(relative)) != expected:
                    raise RuntimeError('Library candidate changed: '+relative)
        descriptions.append(f"{label}: {valid}/{summary['expected']} valid, {failed} failed")
    native_note = ' Actual calibration-library coverage is '+ '; '.join(descriptions)+'. Failed candidates remain in the accounting. These counts describe the matched adaptation used here.'

    cluster = json.loads((REPORT/EXTRAS[1]).read_text())
    expected_sources = {'protocol.json', 'reserved-results.json', 'ablation-results.json'}
    if set(cluster['source_sha256']) != expected_sources:
        raise RuntimeError('Unexpected cluster-interval input inventory')
    for name, expected in cluster['source_sha256'].items():
        if digest(REPORT/name) != expected:
            raise RuntimeError('Cluster intervals are stale: '+name)
    if cluster['script_sha256'] != digest(ROOT/'scripts/coupled_contact_v13/surface_cluster_intervals.py'):
        raise RuntimeError('Cluster-interval script changed')
    objects = [obj['id'] for obj in protocol['objects'] if obj['split'] == 'reserved']
    if len(objects) != 8 or cluster['objects'] != objects or cluster['primary_scenarios'] != protocol['primary_forecast_scenarios']:
        raise RuntimeError('Cluster interval cohort differs from fixed primary group')
    comparison = cluster['comparisons']['rigid']
    paired = json.loads((REPORT/'summary.json').read_text())['forecast_cohorts']['reserved']['primary']['fem_comparisons'].get('rigid')
    paired_paragraph_present = bool(paired)
    paired_count = paired['pairs'] if paired_paragraph_present else 0
    if comparison['pairs'] != paired_count or comparison['expected_pairs'] != len(objects)*len(cluster['primary_scenarios']):
        raise RuntimeError('Cluster interval paired coverage differs')
    if comparison['pairs']:
        if not math.isclose(comparison['mean_gain_mm'], paired['mean_gain_mm'], rel_tol=1e-10, abs_tol=1e-9):
            raise RuntimeError('Cluster interval mean differs from paired table')
        bounds = comparison['percentile_95_mm']
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
            raise RuntimeError('Invalid descriptive interval')
        interval_text = f"[{bounds[0]:.3f}, {bounds[1]:.3f}] mm"
    else:
        if comparison['percentile_95_mm'] is not None:
            raise RuntimeError('Interval exists without completed pairs')
        interval_text = 'unavailable because there are no completed pairs'
    if comparison['draws'] != 10000 or comparison['seed'] != 130013:
        raise RuntimeError('Declared bootstrap settings changed')
    paired_note = (f' Descriptive 95% object-cluster bootstrap interval for the mean paired reduction: {interval_text}. '
        'The 10,000 resamples draw eight objects with replacement, keeping repeated interactions together. '
        f"{comparison['contributing_objects']} objects contribute completed pairs; {comparison['valid_bootstrap_draws']} resamples contain paired scores. "
        'The interval is conditional on paired completion and is not a population guarantee.')
    if not comparison['pairs']:
        paired_note = ' A descriptive object-cluster interval is unavailable because no FEM-versus-rigid pairs completed. Failed predictions remain in coverage; no missing score is imputed.'
    expected_counts = {'native': 1, 'rigid_pair': int(paired_paragraph_present), 'rigid_unavailable': int(not paired_paragraph_present)}
    return native_note, paired_note, expected_builder, expected_counts


def outcome_notes(summary):
    """Fixed-cohort interpretation only; never filter rows by reference quality."""
    primary = summary['forecast_cohorts']['reserved']['primary']
    stress = summary['forecast_cohorts']['reserved']['stress']
    fem, rigid = (primary['methods'][key] for key in ('fem', 'rigid'))
    shape_note = ''
    if fem['complete'] == rigid['complete'] == primary['expected']:
        fw, rw = fem['coordinate_mean_mm'], rigid['coordinate_mean_mm']
        fs, rs = fem['shape_mean_mm'], rigid['shape_mean_mm']
        if any(not math.isfinite(v) or v < 0 for v in (fw, rw, fs, rs)):
            raise RuntimeError('Invalid interpretation metric')
        shape_note = (f' Across all {primary["expected"]} primary cases, world error is {fw:.6f} mm for FEM versus {rw:.6f} mm for rigid contact; '
            f'aligned shape error is {fs:.6f} versus {rs:.6f} mm, a difference of {rs-fs:.6f} mm. '
            'World-motion gains must not be interpreted as evidence of a substantial deformation-prediction gain.')
    else:
        shape_note = ' Shape and world means have different completion coverage; no aggregate deformation-gain interpretation is made.'
    native = primary['methods']['native']
    paired = primary['fem_comparisons'].get('native')
    native_note = (f' Native primary forecasts complete for {native["complete"]}/{native["expected"]}; its mean is conditional on those cases and must not be directly contrasted with a different FEM cohort. ')
    if paired:
        native_note += (f'On the {paired["pairs"]} completed matched pairs, FEM wins {paired["wins"]}, loses {paired["losses"]} and ties {paired["ties"]}; '
            f'the mean paired world-error reduction is {paired["mean_gain_mm"]:.6f} mm. ')
    else:
        native_note += 'No completed matched native comparison is available. '
    native_note += f'Native stress forecasts complete for {stress["methods"]["native"]["complete"]}/{stress["expected"]}.'
    stress_note = (f' Fresh joint calibration/forecast numerical checks pass for {stress["reference_qualified"]}/{stress["expected"]} stress cases, '
        f'compared with {primary["reference_qualified"]}/{primary["expected"]} primary cases. These flags do not alter either fixed denominator.')
    return {'shape_scope': shape_note, 'native_pairs': native_note, 'stress_quality': stress_note}


def add_outcome_note(value, notes, counts):
    prefixes = {'shape_scope': 'World error measures positions in the shared scene.',
                'native_pairs': "Rigid contact fits friction while suppressing deformation.",
                'stress_quality': 'The drop family remains development-unqualified'}
    for key, prefix in prefixes.items():
        if value.startswith(prefix):
            counts[key] += 1
            value += notes[key]
    return value


def main():
    native_note, paired_note, expected_builder, expected_counts = annotations()
    extra_hashes = {name: digest(REPORT/name) for name in EXTRAS + ['summary.json']}
    summary = json.loads((REPORT/'summary.json').read_text())
    notes = outcome_notes(summary)
    note_counts = {key: 0 for key in notes}
    expected_notes = {'shape_scope': 1, 'native_pairs': 1, 'stress_quality': int(bool(summary['forecast_cohorts']['reserved']['stress']['expected']))}
    wrapper_hash = digest(Path(__file__))
    spec = importlib.util.spec_from_file_location('_surface_report_frozen_private', BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_p = module.p
    original_document = module.SimpleDocTemplate
    counts = {'native': 0, 'rigid_pair': 0, 'rigid_unavailable': 0}

    def annotated_p(text, style='BodyText'):
        value = str(text)
        if "The native baseline uses the authors' unchanged PhysTwin spring-force core" in value:
            counts['native'] += 1
            value += native_note
        if value.startswith('Against rigid contact, inferred FEM wins '):
            counts['rigid_pair'] += 1
            value += paired_note
        if expected_counts['rigid_unavailable'] and value.startswith('World error measures positions in the shared scene.'):
            counts['rigid_unavailable'] += 1
            value += paired_note
        value = add_outcome_note(value, notes, note_counts)
        return original_p(value, style)

    class GuardedDocument(original_document):
        def build(self, *args, **kwargs):
            if counts != expected_counts or note_counts != expected_notes:
                raise RuntimeError('Annotation insertion counts differ: '+repr(counts))
            if digest(BUILDER) != expected_builder:
                raise RuntimeError('Frozen builder changed during preparation')
            return super().build(*args, **kwargs)

    module.p = annotated_p
    module.SimpleDocTemplate = GuardedDocument
    module.main()  # Includes all original audit/video/fixed-cohort validation.
    if counts != expected_counts or note_counts != expected_notes or digest(BUILDER) != expected_builder or digest(Path(__file__)) != wrapper_hash:
        raise RuntimeError('Builder/wrapper provenance changed during report build')
    if any(digest(REPORT/name) != expected for name, expected in extra_hashes.items()):
        raise RuntimeError('Auxiliary report changed during build')
    path = REPORT/'paper-build.json'
    provenance = json.loads(path.read_text())
    if provenance['builder_sha256'] != expected_builder:
        raise RuntimeError('Frozen builder provenance mismatch')
    provenance['annotation_wrapper'] = dict(path=str(Path(__file__).relative_to(ROOT)), sha256=wrapper_hash,
        insertion_counts=counts, expected_insertion_counts=expected_counts, outcome_note_counts=note_counts, expected_outcome_note_counts=expected_notes, outcome_notes=notes, extra_artifact_sha256=extra_hashes,
        auxiliary_script_sha256={name: digest(ROOT/'scripts/coupled_contact_v13'/name) for name in ['library_coverage_audit.py', 'surface_cluster_intervals.py']},
        amendment='Reporting-only actual calibration coverage, descriptive object-cluster interval, world/shape distinction, paired native coverage and fresh stress qualification; no cohort, metric, simulation, or selection change')
    temporary = path.with_suffix('.annotated.tmp')
    temporary.write_text(json.dumps(provenance, indent=2)+'\n')
    temporary.replace(path)


if __name__ == '__main__':
    main()
