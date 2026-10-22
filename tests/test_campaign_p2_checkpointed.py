import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.campaign_p2_exact_quadrature import ExactBulkP2Geometry
from experiments.campaign_p2_streaming import StreamingTwoLevelP2
from experiments.campaign_p2_checkpointed import (CheckpointedStaticSolveMixin,
    AcceptedStaticCheckpoint,checkpoint_identity,load_accepted_checkpoint,ROOT)


class Durable(CheckpointedStaticSolveMixin,StreamingTwoLevelP2):pass


def identity(m,t):
    p='experiments/campaign_p2_checkpointed.py'
    return checkpoint_identity(m,t,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()},protocol_sha256='a'*64)


def test_callback_clone_and_numerical_sequence_exact_parity():
    torch.set_num_threads(2);m=ExactBulkP2Geometry((1,1,1));old=StreamingTwoLevelP2(m);new=Durable(m)
    def callback(s):s['displacement'][:]=123;s['history'][0]['residual_N']=-1
    a=old.solve();b=new.solve(on_accepted=callback)
    torch.testing.assert_close(a['displacement'],b['displacement'],rtol=0,atol=0)
    assert a['history']==b['history'] and old.linear_records==new.linear_records


def test_atomic_accepted_checkpoint_resume_budget_and_tamper(tmp_path):
    m=ExactBulkP2Geometry((1,1,1));t=Durable(m);expected=identity(m,t);writer=AcceptedStaticCheckpoint(tmp_path,expected)
    class Stop(Exception):pass
    def stop_after_accepted(s):
        writer(s)
        if s['iteration']==1:raise Stop
    with pytest.raises(Stop):t.solve(on_accepted=stop_after_accepted)
    config=dict(force_tolerance_N=1e-10,maximum_iterations=25,maximum_CG_iterations=4000,quadrature_chunk_size=8)
    loaded=load_accepted_checkpoint(tmp_path,expected,config)
    assert loaded['remaining_Newton_iterations']==24 and loaded['metadata']['iteration']==1
    fresh=Durable(m);resumed=fresh.solve(initial=loaded['initial'],maximum_iterations=24);full=Durable(m).solve()
    torch.testing.assert_close(resumed['displacement'],full['displacement'],rtol=0,atol=0)
    assert float(torch.linalg.vector_norm(resumed['residual']))<=1e-10
    wrong=dict(expected);wrong['material']={**wrong['material'],'young':42}
    with pytest.raises(ValueError,match='identity'):load_accepted_checkpoint(tmp_path,wrong,config)
    with pytest.raises(ValueError,match='identity'):load_accepted_checkpoint(tmp_path,expected,{**config,'force_tolerance_N':1e-8})
    file=tmp_path/loaded['metadata']['data_file'];file.write_bytes(file.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='array'):load_accepted_checkpoint(tmp_path,expected,config)


def test_identity_force_change_and_explicit_static_scope():
    m=ExactBulkP2Geometry((1,1,1));t=Durable(m);a=identity(m,t);t.force=t.force*2;b=identity(m,t)
    assert a['actual_nodal_force']!=b['actual_nodal_force']
    with pytest.raises(ValueError,match='static'):t.solve(target=torch.zeros_like(t.rest),effective_dt=.01)
