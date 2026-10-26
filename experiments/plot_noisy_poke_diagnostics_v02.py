"""Presentation of closed, independently audited v02 recovery results only.

No simulation, inference, RNG or raw archive loads. The v01 scientific negative
and its reader remain unchanged. Pictures and result-derived arrays stay local.
"""
import argparse
import json
import os
from pathlib import Path
from experiments import plot_noisy_poke_diagnostics_v01 as original

ROOT = original.ROOT
OUTPUT = ROOT / 'runs/operator-dev/noisy-poke-diagnostics-media-v02'
POLICIES = original.POLICIES
aggregate, draw, read = original.aggregate, original.draw, original.read
SOURCE = 'experiments/plot_noisy_poke_diagnostics_v02.py'
TEST = 'tests/test_plot_noisy_poke_diagnostics_v02.py'


def components():
    from experiments import run_noisy_poke_continuation_v02 as continuous
    from experiments import run_matched_finite_poke_control_v02 as finite
    from experiments import supervise_noisy_poke_recovery_v02 as operator
    from experiments import audit_noisy_poke_continuation_v02 as continuous_audit
    from experiments import audit_matched_finite_poke_control_v02 as finite_audit
    return continuous, finite, operator, continuous_audit, finite_audit


def need(value, message):
    if not value:
        raise ValueError(message)


def terminal_gate():
    """No result rows become readable until both recovery audits are complete."""
    c, f, op, ac, af = components()
    marker = op.LOCAL / 'completed.json'
    done = read(marker)
    need(done.get('completed') is True, 'Completed v02 external operator required')
    need(done.get('noisy_audit') == op.record(c.LOCAL / 'independent-terminal-review.json')
         and done.get('finite_audit') == op.record(f.LOCAL / 'independent-terminal-review.json')
         and done.get('operator_preapproval') == op.record(op.LOCAL / 'root-preapproval.json'),
         'Canonical new audit and authority commitments')
    need(set(done.get('stages', {})) == set(op.CAPS), 'All five recovery stages required')
    c.parent.verify(done)
    approval = op.preapproval(op.sha(op.LOCAL / 'root-preapproval.json'))
    parent_sha = approval['parent_protocol_SHA256']
    specifications = {
        'remaining': (c.__name__, ['run', '--sha', parent_sha], [c.LOCAL / 'execution-remaining.json']),
        'summary': (c.__name__, ['summarize', '--sha', parent_sha], [c.LOCAL / 'summary-LOCAL.json']),
        'continuous_audit': (ac.__name__, ['--sha', parent_sha], [c.LOCAL / 'independent-terminal-review.json']),
        'finite_evaluate': (f.__name__, ['evaluate', '--parent-sha', parent_sha],
                            [f.LOCAL / 'evaluation-results-LOCAL.json', f.LOCAL / 'compact-summary.json']),
        'finite_audit': (af.__name__, ['--parent-sha', parent_sha], [f.LOCAL / 'independent-terminal-review.json']),
    }
    for phase, (module, args, outputs) in specifications.items():
        path = op.LOCAL / 'stages' / phase / 'execution.json'
        need(done['stages'][phase] == op.record(path), 'Canonical completed stage membership')
        state = read(op.exact(done['stages'][phase]))
        op.verify_stage(state, phase, op.command(module, args, op.CAPS[phase][0]), outputs,
                        op.sha(op.LOCAL / 'root-preapproval.json'), parent_sha)
    # These functions verify original choice/checkpoint and new execution roots,
    # prospective source/auditor pins and all transitive hashes without NPZ.
    _, _, _, _, entries = ac.metadata_barrier(parent_sha)
    _, _, _, _, closures = af.metadata_barrier(parent_sha)
    ca = read(op.exact(done['noisy_audit']))
    fa = read(op.exact(done['finite_audit']))
    need(ca.get('audit_passed') is True and fa.get('audit_passed') is True,
         'Both independent v02 audits required')
    need(ca.get('source_SHA256') == ac.sources()
         and ca.get('execution_source_SHA256') == c.sources()
         and ca.get('parent_source_SHA256') == c.parent.sources()
         and ca.get('parent_protocol_SHA256') == parent_sha
         and ca.get('all32_trial_records') == [r['record'] for r in entries]
         and ca.get('source_summary') == op.record(c.LOCAL / 'summary-LOCAL.json')
         and fa.get('source_SHA256') == af.sources()
         and fa.get('source_numerical_SHA256') == f.sources()
         and fa.get('closed_stages') == closures
         and fa.get('source_evaluation') == op.record(f.LOCAL / 'evaluation-results-LOCAL.json'),
         'Exact independent audit procedure, mixed lineage and result bindings')
    need(ca.get('qualified_prefixes') == 32 and ca.get('qualified_joint_posteriors') == 256
         and fa.get('retained_trials_per_model') == 32
         and fa.get('independently_recomputed_posterior_updates') == 512,
         'Complete numerical qualification denominators')
    need(ca.get('summary', {}).get('comparison_blocked') is False
         and fa.get('summary', {}).get('comparison_blocked') is False,
         'Unresolved comparisons stay blocked; no subset figure')
    c.parent.verify(ca)
    c.parent.verify(fa)
    return op, ca, fa, dict(continuous_audit=done['noisy_audit'], finite_audit=done['finite_audit'],
                           operator_complete=op.record(marker),
                           original_protocol=op.record(c.parent.LOCAL / 'protocol.json'),
                           execution_protocol=op.record(c.LOCAL / 'protocol.json'))


def load():
    op, ca, fa, commitments = terminal_gate()
    rows = []
    for rec in ca['all32_trial_records']:
        row = read(op.exact(rec))
        need(row.get('qualified') is True, 'No unresolved continuous trial')
        rows.append(dict(index=row['index'], object_index=row['object_index'], replicate=row['replicate'],
                         metrics=dict(before=row['before']['field'],
                                      **{p: row['policies'][p] for p in POLICIES[1:]})))
    models = {'Continuous linear': aggregate(rows)}
    evaluation = read(op.exact(fa['source_evaluation']))
    for name, title in (('linear', 'Finite linear'), ('NH', 'Finite nonlinear')):
        rows = []
        for row in evaluation['rows']:
            if row['model'] != name:
                continue
            need(row.get('choice_qualified') is True, 'No unresolved finite choice')
            rows.append(dict(index=row['index'], object_index=row['object_index'], replicate=row['replicate'],
                             metrics=row['policies']))
        models[title] = aggregate(rows)
    return dict(models=models, source_commitments=commitments,
                presentation_source_SHA256={SOURCE: op.sha(ROOT / SOURCE), TEST: op.sha(ROOT / TEST)},
                scope='Same static development task: eight fixed objects and four saved noise repeats. '
                      'Original trial0 plus resource-only continuation1..31; original finite64 choices reused. '
                      'Finite models share uniform eight-state prior and known fixture stiffness; continuous '
                      'comparison also changes the prior. Descriptive results, no population/SBC/nominal '
                      'coverage/general neural or dynamic prediction claim.')


def main():
    need(set(os.sched_getaffinity(0)) == {19} and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
         and all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')),
         'Saved-result plotting requires CPU19/noGPU/one-thread')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path, image_path = OUTPUT / 'scalar-summary.json', OUTPUT / 'comparison.png'
    need(not summary_path.exists() and not image_path.exists(), 'No result or presentation overwrite')
    summary = load()
    draw(summary, image_path)
    with summary_path.open('x') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(summary=str(summary_path.relative_to(ROOT)), image=str(image_path.relative_to(ROOT)))))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
