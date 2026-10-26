"""Manufactured metadata/results only; no actual postprocessing inputs."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from experiments import plot_noisy_poke_diagnostics_v02 as plot


@pytest.fixture
def fake(monkeypatch):
    root = Path('/synthetic-presentation')
    rec = lambda p: dict(path=str(p), sha256='manufactured')
    modules = [SimpleNamespace(__name__=name, LOCAL=root / name)
               for name in ('continuous', 'finite', 'operator', 'continuous_audit', 'finite_audit')]
    c, f, op, ac, af = modules
    op.CAPS = dict(remaining=[18], summary=[18], continuous_audit=[16], finite_evaluate=[17], finite_audit=[16])
    op.record = rec
    op.exact = lambda r: Path(r['path'])
    op.sha = lambda p: 'manufactured'
    op.command = lambda module, args, cpu: [cpu, module, *args]
    op.verify_stage = Mock()
    op.preapproval = lambda digest: {'parent_protocol_SHA256': 'original-parent'}
    c.parent = SimpleNamespace(LOCAL=root / 'old', sources=lambda: {'old': 'held'}, verify=Mock())
    c.sources = lambda: {'new': 'held'}
    f.sources = lambda: {'finite': 'held'}
    ac.sources = lambda: {'auditor': 'held'}
    af.sources = lambda: {'finite_auditor': 'held'}
    entries = [dict(index=i, record=rec(root / ('old' if i == 0 else 'continuous') / str(i))) for i in range(32)]
    ac.metadata_barrier = Mock(return_value=({}, {}, [], {}, entries))
    closures = {'original_choose': 'closed', 'new_evaluate': 'closed'}
    af.metadata_barrier = Mock(return_value=({}, {}, {}, {}, closures))
    done = dict(completed=True, noisy_audit=rec(c.LOCAL / 'independent-terminal-review.json'),
                finite_audit=rec(f.LOCAL / 'independent-terminal-review.json'),
                operator_preapproval=rec(op.LOCAL / 'root-preapproval.json'),
                stages={p: rec(op.LOCAL / 'stages' / p / 'execution.json') for p in op.CAPS})
    ca = dict(audit_passed=True, source_SHA256=ac.sources(), execution_source_SHA256=c.sources(),
              parent_source_SHA256=c.parent.sources(), parent_protocol_SHA256='original-parent',
              all32_trial_records=[v['record'] for v in entries],
              source_summary=rec(c.LOCAL / 'summary-LOCAL.json'), qualified_prefixes=32,
              qualified_joint_posteriors=256, summary={'comparison_blocked': False})
    fa = dict(audit_passed=True, source_SHA256=af.sources(), source_numerical_SHA256=f.sources(),
              closed_stages=closures, source_evaluation=rec(f.LOCAL / 'evaluation-results-LOCAL.json'),
              retained_trials_per_model=32, independently_recomputed_posterior_updates=512,
              summary={'comparison_blocked': False})
    data = {op.LOCAL / 'completed.json': done, Path(done['noisy_audit']['path']): ca,
            Path(done['finite_audit']['path']): fa,
            **{Path(v['path']): {'synthetic_stage': name} for name, v in done['stages'].items()}}
    reads = []
    def read(path):
        reads.append(path)
        return data[path]
    monkeypatch.setattr(plot, 'components', lambda: modules)
    monkeypatch.setattr(plot, 'read', read)
    return SimpleNamespace(c=c, f=f, op=op, ac=ac, af=af, done=done, ca=ca, fa=fa,
                           data=data, reads=reads, rec=rec, root=root)


def test_full_new_terminal_gate_and_command_membership(fake):
    op, ca, fa, commitments = plot.terminal_gate()
    assert (op, ca, fa) == (fake.op, fake.ca, fake.fa)
    assert fake.op.verify_stage.call_count == 5
    assert fake.ac.metadata_barrier.call_args.args == ('original-parent',)
    assert fake.af.metadata_barrier.call_count == 1
    assert fake.c.parent.verify.call_count == 3
    assert commitments['original_protocol'] != commitments['execution_protocol']
    calls = fake.op.verify_stage.call_args_list
    assert calls[2].args[2] == [16, 'continuous_audit', '--sha', 'original-parent']
    assert calls[4].args[2] == [16, 'finite_audit', '--parent-sha', 'original-parent']


def test_incomplete_marker_or_old_audit_path_blocks_before_rows(fake):
    fake.done['completed'] = False
    with pytest.raises(ValueError, match='Completed v02'):
        plot.terminal_gate()
    assert fake.op.verify_stage.call_count == 0 and fake.ac.metadata_barrier.call_count == 0
    fake.done['completed'] = True
    fake.done['noisy_audit'] = fake.rec(fake.root / 'old' / 'independent-terminal-review.json')
    with pytest.raises(ValueError, match='Canonical new audit'):
        plot.terminal_gate()
    assert fake.ac.metadata_barrier.call_count == 0


def test_missing_stage_closure_and_source_tamper_fail_closed(fake):
    fake.done['stages'].pop('finite_audit')
    with pytest.raises(ValueError, match='All five'):
        plot.terminal_gate()
    fake.done['stages']['finite_audit'] = fake.rec(fake.op.LOCAL / 'stages/finite_audit/execution.json')
    fake.ac.metadata_barrier.side_effect = ValueError('Missing original checkpoint root closure')
    with pytest.raises(ValueError, match='checkpoint root closure'):
        plot.terminal_gate()
    fake.ac.metadata_barrier.side_effect = None
    fake.ca['parent_source_SHA256'] = {'new': 'falsely old'}
    with pytest.raises(ValueError, match='Exact independent audit'):
        plot.terminal_gate()


def test_numerical_negative_or_missing_denominator_never_subset_plot(fake):
    fake.ca['summary']['comparison_blocked'] = True
    with pytest.raises(ValueError, match='Unresolved comparisons'):
        plot.terminal_gate()
    fake.ca['summary']['comparison_blocked'] = False
    fake.fa['retained_trials_per_model'] = 31
    with pytest.raises(ValueError, match='qualification denominators'):
        plot.terminal_gate()


def test_load_keeps_exact_three_models_all32_and_mean_before_sqrt(monkeypatch):
    root = Path('/synthetic-rows')
    rec = lambda path: {'path': str(path), 'sha256': 'toy'}
    ca = dict(all32_trial_records=[rec(root / str(i)) for i in range(32)])
    fa = dict(source_evaluation=rec(root / 'finite'))
    data = {}
    finite_rows = []
    for i in range(32):
        values = {p: dict(MSE_m2=float((i // 8) ** 2), posterior_variance_m2=float((i % 8) ** 2))
                  for p in plot.POLICIES}
        data[root / str(i)] = dict(qualified=True, index=i, object_index=i % 8, replicate=i // 8,
                                  before={'field': values['before']}, policies={p: values[p] for p in plot.POLICIES[1:]})
        finite_rows += [dict(model=name, index=i, object_index=i % 8, replicate=i // 8,
                             choice_qualified=True, policies=copy.deepcopy(values)) for name in ('linear', 'NH')]
    data[root / 'finite'] = dict(rows=finite_rows)
    op = SimpleNamespace(exact=lambda r: Path(r['path']), sha=lambda p: 'synthetic-source')
    monkeypatch.setattr(plot, 'terminal_gate', lambda: (op, ca, fa, {'proof': 'synthetic'}))
    monkeypatch.setattr(plot, 'read', lambda p: data[p])
    result = plot.load()
    assert list(result['models']) == ['Continuous linear', 'Finite linear', 'Finite nonlinear']
    for model in result['models'].values():
        assert model['all32_retained'] and len(model['objects']) == 8
        assert model['policies']['before']['RMSE_m'] == np.sqrt(3.5)
        assert model['policies']['before']['posterior_RMS_SD_m'] == np.sqrt(17.5)
    assert 'known fixture stiffness' in result['scope'] and 'also changes the prior' in result['scope']
    data[root / 'finite']['rows'].pop()
    with pytest.raises(ValueError, match='Every prescribed'):
        plot.load()


def test_presentation_math_is_frozen_reviewed_pure_code():
    assert plot.aggregate is plot.original.aggregate and plot.draw is plot.original.draw
    assert plot.POLICIES == ('before', 'adaptive', 'fixed', 'uniform_random')
