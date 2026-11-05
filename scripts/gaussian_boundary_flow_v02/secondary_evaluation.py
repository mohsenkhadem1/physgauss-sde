"""Evaluate immutable rigid-control imports after the primary cohort barrier.

This separate command does not change the frozen primary evaluation protocol.
It preserves failed or interrupted evaluations rather than silently retrying.
"""
import argparse
import json
from pathlib import Path
from scripts.gaussian_boundary_flow_v02.validation_cohort import (
    TERMINAL, cohort_lock, launch, process_identity, require_inactive,
    terminal_gate, verified_evaluation, write, sha,
)
from scripts.gaussian_boundary_flow_v02.import_rigid_contact_secondary import file_manifest
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed


def evaluate(root, protocol, protocol_digest, import_report):
    root, protocol = Path(root), Path(protocol)
    if sha(protocol) != protocol_digest:
        raise ValueError('Protocol digest changed')
    with cohort_lock(root):
        rows = terminal_gate(root, protocol_digest)
        gate = root / 'evaluation-gate.json'
        expected = dict(protocol_sha256=protocol_digest,
                        all12_terminal_before_any_target_read=True, cases=rows)
        if not gate.exists() or json.loads(gate.read_text()) != expected:
            raise PermissionError('Primary evaluator barrier has not been admitted')
        import_digest=sha(import_report)
        import_document=json.loads(Path(import_report).read_text())
        if import_document.get('secondary')is not True or import_document.get('target_access')is not False:raise ValueError('Invalid secondary import report role')
        imports = import_document['cases']
        if len(imports) != 12 or {r['case'] for r in imports} != {r['case'] for r in rows}:
            raise ValueError('Secondary imports do not cover the fixed cohort')
        # Check all imported artifacts before any target read.
        for row in imports:
            if not isinstance(row.get('imported'),bool)or row.get('state')not in TERMINAL:raise ValueError('Invalid terminal import status')
            if row['imported']:
                if row['state']!='complete':raise ValueError('Imported candidate is not complete')
                folder = root / row['case']
                if json.loads((folder / 'rigid_contact-import.json').read_text()) != row:
                    raise ValueError('Secondary import provenance changed')
                destination=folder/'rigid_contact'
                if destination.is_symlink()or Path(row['destination']).resolve()!=destination.resolve():raise ValueError('Secondary import destination mismatch')
                if file_manifest(destination) != row['files_sha256']or file_manifest(row['source'])!=row['files_sha256']:
                    raise ValueError('Secondary source or imported files changed')
                source_plan,_,_,_=load_sealed(destination)
                if source_plan.get('case')!=row['case']or source_plan.get('secondary_addendum_sha256')!=row['secondary_addendum_sha256']or source_plan.get('validation_protocol_sha256')!=protocol_digest:
                    raise ValueError('Secondary sealed import provenance mismatch')
        # A later live orphan must block the entire phase before any earlier case writes/scores.
        for status in root.glob('*/*-evaluation-status.json'):
            old=json.loads(status.read_text())
            if old.get('state')not in TERMINAL:require_inactive(old)
        for row in imports:
            folder=root/row['case'];status=folder/'rigid_contact-evaluation-status.json'
            if status.exists():
                old=json.loads(status.read_text())
                if old.get('cohort_gate_sha256')!=sha(gate)or old.get('import_report_sha256')!=import_digest:raise ValueError('Changed secondary evaluator provenance')
                if old.get('state')=='complete'and not verified_evaluation(folder,'rigid_contact',protocol_digest):raise ValueError('Completed secondary score changed')
        for row in imports:
            if sha(import_report)!=import_digest:raise ValueError('Import report changed during evaluation')
            folder = root / row['case']; stage = 'rigid_contact'
            status = folder / (stage + '-evaluation-status.json')
            if status.exists():
                old = json.loads(status.read_text())
                if old['state'] in TERMINAL:
                    if old['state'] == 'complete' and not verified_evaluation(folder, stage, protocol_digest):
                        raise ValueError('Completed secondary score changed')
                    continue
                require_inactive(old)
                recovered = verified_evaluation(folder, stage, protocol_digest)
                write(status, {**old, 'state': 'complete' if recovered else 'interrupted',
                               'recovered_verified_score': recovered})
                continue
            record = dict(cohort_gate_sha256=sha(gate), import_report_sha256=import_digest, secondary=True)
            if not row['imported']:
                write(status, {**record, 'state': 'blocked', 'import_status': row}); continue
            command = ['runs/phystwin-env/bin/python', '-m',
                       'scripts.gaussian_boundary_flow_v02.softvt_validation_score',
                       '--run', str(folder / stage), '--output', str(folder / (stage + '-scores.json')),
                       '--protocol', str(protocol), '--protocol-sha256', protocol_digest,
                       '--cohort-root', str(root)]
            record.update(state='running', coordinator=process_identity(), command=command)
            write(status, record)
            def started(identity):
                record['child'] = identity; write(status, record)
            try:
                code = launch(command, folder / (stage + '-evaluation.log'), started, 600)
                if code==0 and not verified_evaluation(folder,stage,protocol_digest):raise ValueError('Evaluator exited without a verified score')
                write(status, {**record, 'state': 'complete' if code == 0 else 'failed', 'returncode': code})
            except Exception as error:
                write(status, {**record, 'state': 'failed', 'error': str(error)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'protocol', 'protocol-sha256', 'import-report'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    evaluate(args.root, args.protocol, args.protocol_sha256, args.import_report)
