"""Summarize ledgers and declared TRAIN audits; never open held-out responses."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def process_live(pid):
    path = Path('/proc')/str(pid)/'stat'
    try:
        return path.read_text().rsplit(')', 1)[1].strip().split()[0] != 'Z'
    except FileNotFoundError:
        return False


def snapshot(campaign, root):
    config = read(campaign)
    jobs = read(root/'launch-ledger.json')
    status = {'campaign': config['campaign'], 'status': 'active',
        'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'campaign_sha256': hashlib.sha256(campaign.read_bytes()).hexdigest(),
        'heldout_plan_sha256': hashlib.sha256(Path(config['heldout_plan']).read_bytes()).hexdigest(),
        'milestone': 'Known-force forward prediction: pilot loss/sampling ablations and additional Isaac controls',
        'jobs': [], 'simulation': {}, 'pilots': None,
        'confirmation_labels_used_for_model_selection': False,
        'qualification': 'Progress snapshot only; no new improvement or convergence claimed. Local ledgers remain the live source.'}
    for job in jobs['jobs'] if jobs else []:
        status['jobs'].append({'name': job['name'], 'gpu': job['gpu'], 'live_at_snapshot': process_live(job['pid'])})
    for name, key in [('controls-train', 'new_train_control_ids'), ('controls-validation', 'new_validation_control_ids')]:
        folder = root/name
        success, failed, partial = [], [], []
        for eid in config[key]:
            private = folder/'private'/eid
            audit, summary = read(private/'validation.json'), read(private/'summary.json')
            if audit and audit.get('passed') and summary and summary.get('completed'):
                success.append(eid)
            elif (private/'failure.json').is_file():
                failed.append(eid)
            elif private.exists():
                partial.append(eid)
        status['simulation'][name] = {'planned_layouts': len(config[key]), 'verified_layouts': success,
            'failed_layouts': failed, 'in_progress_layouts': partial}
    pilots = read(root/'pilots/execution.json')
    if pilots:
        status['pilots'] = {'completed': pilots['completed'], 'arms': []}
        for arm in pilots['arms']:
            history = read(root/'pilots'/arm['name']/'history.json')
            status['pilots']['arms'].append({'name': arm['name'], 'state': arm['state'],
                'last_validation': history[-1] if history else None})
    audit = read(root/'command-audit-v02.json')
    if audit:
        status['command_audit'] = {key: audit[key] for key in ['records', 'passed', 'test_labels_opened', 'validation_labels_opened']}
        status['command_audit']['clock_note'] = audit['clock_note']
    status['nonlinear_comparisons'] = {}
    for directory in sorted(root.glob('nonlinear-*')):
        run = read(directory/'execution.json')
        if not run:
            continue
        rows = []
        for arm in run['arms']:
            history = read(directory/arm['name']/'history.json')
            rows.append({'name': arm['name'], 'state': arm['state'],
                         'last_validation': history[-1] if history else None})
        status['nonlinear_comparisons'][directory.name] = {'wave': run['wave'], 'completed': run['completed'], 'arms': rows}
    zero = read(root/'linearity-diagnostic/execution.json')
    if zero:
        status['zero_force_diagnostic'] = {'completed': zero['completed'],
            'cases': [{'episode_id': c['episode_id'], 'state': c['state']} for c in zero['cases']]}
        status['milestone'] = 'Known-force representability, nonlinear transfer comparison, and expanded Isaac controls'
    rich = read(root/'rich-controls/execution.json')
    if rich:
        status['rich_control_curriculum'] = {'completed': rich['completed'], 'planned_layouts': len(rich['episode_ids']),
            'cases': rich['cases']}
    calibration = read(Path('configs/gaussian_campaign_calibration_v01.json'))
    if calibration:
        verified, partial, failed = [], [], []
        for eid in calibration['fit_ids']+calibration['audit_only_ids']:
            private = Path(calibration['pool'])/'private'/eid
            audit, summary = read(private/'validation.json'), read(private/'summary.json')
            if audit and audit.get('passed') and summary and summary.get('completed'):
                verified.append(eid)
            elif (private/'failure.json').is_file():
                failed.append(eid)
            elif private.exists():
                partial.append(eid)
        status['simulation']['controls-calibration'] = {
            'planned_layouts': len(calibration['fit_ids'])+len(calibration['audit_only_ids']),
            'verified_layouts': verified, 'in_progress_layouts': partial, 'failed_layouts': failed,
            'weight_fitting_allowed': False, 'predictive_audit_opened': False}
    numerical = read(root/'numerical-timestep-v01/analysis.json')
    if numerical:
        status['milestone'] = 'Numerical reliability diagnosis before further fitting or calibration'
        status['numerical_reliability'] = {
            'timestep_gate_passed': False, 'case': 't0001',
            'refinement_pairs': numerical['refinement_pairs'],
            'fresh960_bitwise_replay': numerical['fresh960_vs_original960_bitwise'],
            'protocol': 'configs/gaussian_campaign_numerical_gate_v01.json'}
        stopped = read(root/'calibration-stop-status.json')
        status['calibration_capture_graceful_stop'] = stopped
        status['numerical_jobs'] = {}
        for directory in sorted(root.glob('numerical-*')):
            execution = read(directory/'execution.json')
            if not execution:
                continue
            rows = []
            for cell in execution.get('cells',execution.get('cases',[])):
                rows.append({**{key:cell[key] for key in
                    ['condition','iterations','physics_hz','state','exit_code','gpu'] if key in cell},
                    'worker_live_at_snapshot': process_live(cell['pid']) if 'pid' in cell else False})
            status['numerical_jobs'][directory.name] = {'completed':execution.get('completed'),
                'driver_live_at_snapshot': process_live(execution['driver_pid']) if 'driver_pid' in execution else False,
                'cells':rows}
    force_flag = read(root/'numerical-external-forces-v01/analysis.json')
    if force_flag:
        status['external_force_iteration_flag'] = {
            'case': 'TRAIN t0001', 'physics_hz': 960,
            'native_numerical_gate_passed': False,
            'conclusion': 'Flag alone does not restore numerical reliability; retain the failed 128-iteration cell.',
            'cells': [{**{key: cell[key] for key in [
                'id', 'enabled', 'iterations', 'state',
                'diagonal_coordinate_rms_from_rest_m'] if key in cell},
                **({'failure_time_s': cell['failure']['time_s'],
                    'failure_message': cell['failure']['message']}
                   if isinstance(cell.get('failure'), dict) else {})}
                for cell in force_flag['cells']],
            'comparisons': force_flag['comparisons'],
            'false32_baseline_bitwise': force_flag['false32_baseline_bitwise'],
            'evidence': 'campaign-diagnosis/external-force-iteration-flag-v01.json'}
    lowrate = read(root/'numerical-lowrate-v01/analysis.json')
    if lowrate:
        status['lowrate_numerical_audit'] = {
            'case': 'TRAIN t0001', 'deformable_iterations': 32,
            'native_numerical_gate_passed': False,
            'assessments': lowrate['assessments'],
            'translation_gate_passed': lowrate['translation_gate_passed'],
            'diagonal_responses': [{key: row[key] for key in [
                'condition', 'physics_hz', 'state', 'coordinate_rms_from_rest_m']}
                for row in lowrate['rows'] if row.get('action_id') == 'evaluation'],
            'diagonal_refinement_pairs': [row for row in lowrate['refinement_pairs']
                if row['action_id'] == 'evaluation'],
            'diagonal_translation_pairs': [row for row in lowrate['translation_pairs']
                if row['action_id'] == 'evaluation'],
            'qualification': 'All six cells passed sensor/physics QC; both coordinate conditions fail low-rate refinement and translation agreement also fails. Higher-rate negative evidence is retained.',
            'evidence': 'campaign-diagnosis/lowrate-numerical-v01.json'}
    energy_proxy = read(root/'native-energy-proxy-v01.json')
    if energy_proxy:
        status['continuum_energy_proxy_metadata'] = {
            'case': 'TRAIN t0001', 'physics_hz': 960,
            'conditions': ['original', 'recentered'],
            'native_energy_conservation_assessed': False,
            'qualification': 'Independent declared continuum proxies and held-force translational work; native deformable masses, rigid kinetic and constraint energies are unavailable. No native conservation verdict.',
            'evidence': 'campaign-diagnosis/continuum-energy-proxy-v01.json'}
    linear_fem = read(root/'linear-fem-t0001-v01/results.json')
    if linear_fem:
        status['independent_linear_fem'] = {
            'case': 'TRAIN t0001',
            'temporal_refinement_passed_for_declared_linear_system': linear_fem['refinement_converged'],
            'native_numerical_gate_passed': False,
            'static_coordinate_rms_displacement_m': linear_fem['static_peak_force']['coordinate_rms_displacement_m'],
            'dynamic_coordinate_rms_displacement_m': [
                {key: row[key] for key in ['hz', 'coordinate_rms_displacement_m']}
                for row in linear_fem['dynamics']],
            'refinement_pairs': linear_fem['refinement_pairs'],
            'qualification': 'Independent small-strain linear FEM sanity reference; finite rotations, geometric nonlinearity and native elasticity damping are absent. Temporal convergence is not native or continuum validation.',
            'evidence': 'campaign-diagnosis/linear-fem-t0001-v01.json'}
    nonlinear_fem = read(root/'nonlinear-fem-static-t0001-v02/results.json')
    reactions = read(root/'nonlinear-fem-static-t0001-v02/reaction-balance-audit.json')
    if nonlinear_fem and reactions:
        peak = nonlinear_fem['continuation'][-1]
        status['independent_finite_strain_static_fem'] = {
            'case': 'TRAIN t0001', 'current_revision': 'v02',
            'all_seven_load_scale_residual_gates_passed': nonlinear_fem['all_declared_residual_gates_passed'],
            'independent_nodal_reaction_balance_passed': reactions['all_checks_passed'],
            'peak_coordinate_rms_displacement_m': peak['coordinate_rms_displacement_m'],
            'peak_coordinate_rmse_vs_linear_m': peak['coordinate_rmse_vs_linear_m'],
            'peak_relative_coordinate_rmse_vs_linear': peak['relative_coordinate_rmse_vs_linear'],
            'final_tangent_minimum_eigenvalue': nonlinear_fem['final_nonlinear_equilibrium_tangent']['minimum_eigenvalue'],
            'native_numerical_gate_passed': False,
            'qualification': 'Independent finite-strain static local equilibrium on a fixed mesh; no native constitutive equivalence, dynamic prediction, continuum convergence or global-branch uniqueness claim. Preliminary v01 is preserved.',
            'evidence': 'campaign-diagnosis/nonlinear-fem-static-t0001-v02.json'}
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=Path('configs/gaussian_campaign_v01.json'))
    parser.add_argument('--root', type=Path, default=Path('runs/gaussian-campaign-v01'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = snapshot(args.campaign, args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
