"""No FE/GPU lifecycle/indicator controls for the prospective refinement."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from experiments import run_long_smooth_refinement_v04 as run
from experiments.campaign_long_trajectory_runtime_v03 import checkpoint, record


def fixture_aggregate(tmp_path, monkeypatch, *, velocity_scale=5., node_q_scale=1., node_v_scale=1.):
    old, out, op = [tmp_path/x for x in ('old', 'out', 'op')]
    op.mkdir()
    monkeypatch.setattr(run, 'OLD', old)
    monkeypatch.setattr(run, 'OUT', out)
    monkeypatch.setattr(run, 'OP', op)
    digest = 'test-only-protocol'
    # q64-q128=4um; q128-q256=1um. Spatial2um/(1.25-1)=8um.
    # The h128 total must be2*1+8=10um, not1+8 or a coarseh128/fineh256 pair.
    for folder, amplitude, hmax, steps in [(old/'fine40_h64', 5e-6, .016, 64),
            (old/'fine40_h128', 1e-6, .016, 128), (out/run.JOB['id'], 0., .016, 256),
            (old/'coarse32_h128', 3e-6, .020, 128)]:
        folder.mkdir(parents=True)
        np.savez(folder/'geometry.npz', rest_m=np.zeros((1, 3)))
        clocks = []
        for i in range(9):
            u = np.array([[[0., 0., amplitude*(i/8)]]])[0]
            np.savez(folder/f'clock-{i:03d}.npz', q=u*node_q_scale,
                     v=u*velocity_scale*node_v_scale, probe_q=u, probe_v=u*velocity_scale)
            clocks.append(dict(time_s=i/8, capture=record(folder/f'clock-{i:03d}.npz')))
        ledger = folder/'accepted-ledger.jsonl'
        ledger.write_text(''.join(json.dumps(dict(step=k, time_s=k/steps, passed=True))+'\n' for k in range(1, steps+1)))
        result = dict(completed=True, passed=True, accepted_steps=steps, protocol_sha256=digest,
                      geometry=record(folder/'geometry.npz'), accepted_ledger=record(ledger),
                      clocks=clocks, hmax_m=hmax)
        if steps == 256:
            # Construct real checkpoint transactions so aggregate's lineage and
            # terminal-state checks are independently exercised.
            for i in range(257):
                checkpoint(folder/'checkpoints', np.zeros((1, 3)), np.zeros((1, 3)), i, i/256, digest, run.JOB['id'])
            result['final_checkpoint_commit'] = record(folder/'checkpoints/COMMIT.json')
        (folder/'result.json').write_text(json.dumps(result))
    return run.aggregate(dict(clocks_s=[i/8 for i in range(9)], qualification=dict(premise='test conditionalp>=1')), digest)


def test_aggregate_qualifies_h128_using_twice_finest_delta_and_same_time_space(tmp_path, monkeypatch):
    result = fixture_aggregate(tmp_path, monkeypatch)
    assert result['conditional_qualification_passed']
    q = result['quantities']['q']
    assert q['63query']['h128_temporal_indicator'] == pytest.approx(2e-6)
    assert q['same_h128_spatial_indicator'] == pytest.approx(8e-6)
    assert q['combined_query_indicator'] == pytest.approx(10e-6)
    assert result['target'] == 'EXISTING fine40_h128'


def test_velocity_only_combined_accuracy_failure_is_not_hidden_by_position_pass(tmp_path, monkeypatch):
    result = fixture_aggregate(tmp_path, monkeypatch, velocity_scale=25.)
    assert result['quantities']['q']['passed']
    assert result['quantities']['v']['allnode']['temporal_passed']
    assert not result['quantities']['v']['passed']
    assert not result['conditional_qualification_passed']


@pytest.mark.parametrize('quantity,scales', [('q', dict(node_q_scale=20.)), ('v', dict(node_v_scale=20.))])
def test_allnode_accuracy_failure_blocks_otherwise_good_queries(tmp_path, monkeypatch, quantity, scales):
    result = fixture_aggregate(tmp_path, monkeypatch, **scales)
    row = result['quantities'][quantity]
    assert row['63query']['temporal_passed']
    assert row['combined_query_indicator'] < row['threshold']
    assert row['allnode']['temporal_contraction_passed']
    assert not row['allnode']['temporal_accuracy_passed']
    assert not row['passed'] and not result['conditional_qualification_passed']


def test_parity_failure_blocks_fine_launch(tmp_path, monkeypatch):
    op = tmp_path/'op'
    op.mkdir()
    out = tmp_path/'out'
    monkeypatch.setattr(run, 'OP', op)
    monkeypatch.setattr(run, 'OUT', out)
    monkeypatch.setattr(run, 'validate', lambda *a, **k: dict(taskset=dict(path='/usr/bin/taskset')))
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *a, **k: '')
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda *a: type('Usage', (), dict(free=100*1024**3))())
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        return dict(exit_code=1, worker_pid=99999999, timeout=False)
    monkeypatch.setattr(run, 'run_owned_command', execute)
    assert not run.controller('frozen-test')
    assert len(calls) == 1 and calls[0][-3] == 'parity'
    execution = json.loads((op/'execution.json').read_text())
    assert execution['terminal'] and not execution.get('completed', False)


@pytest.mark.parametrize('live,groups', [([123], []), ([], [123])])
def test_recovery_refuses_live_pid_or_remaining_group(tmp_path, monkeypatch, live, groups):
    monkeypatch.setattr(run, 'OP', tmp_path)
    monkeypatch.setattr(run, 'inspect', lambda digest: dict(owned_live=live, owned_groups_present=groups))
    with pytest.raises(RuntimeError, match='identities AND groups'):
        run.recover('test-protocol')
    assert not list(tmp_path.glob('recovery-manifest-*'))


@pytest.mark.parametrize('which', ['current_state', 'previous_state', 'previous_metadata'])
def test_recovery_refuses_current_or_previous_transaction_hash_corruption(tmp_path, monkeypatch, which):
    op, out = tmp_path/'op', tmp_path/'out'
    op.mkdir()
    monkeypatch.setattr(run, 'OP', op)
    monkeypatch.setattr(run, 'OUT', out)
    monkeypatch.setattr(run, 'validate', lambda *a, **k: {})
    root = out/run.JOB['id']/'checkpoints'
    digest = 'test-protocol'
    for i in range(3):
        checkpoint(root, np.zeros((1, 3)), np.zeros((1, 3)), i, i/256, digest, run.JOB['id'])
    generation = root/('accepted-00002' if which == 'current_state' else 'accepted-00001')
    bad = generation/('metadata.json' if which == 'previous_metadata' else 'state.npz')
    with bad.open('ab') as f:
        f.write(b'corrupted')
    with pytest.raises(ValueError, match='hash|lineage'):
        run.recover(digest)
    assert not list(op.glob('recovery-manifest-*'))


def test_proc_identity_handles_missing_process_and_retains_own_identity():
    assert run.proc_identity(999999999) is None
    identity = run.proc_identity(os.getpid())
    assert identity['pid'] == os.getpid() and identity['process_start_ticks'] and identity['boot_id']
