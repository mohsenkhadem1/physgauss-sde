"""Read process/QC ledgers only; never inspect held-out predictive responses."""
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
