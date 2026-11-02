import json
from pathlib import Path
from scripts.compositional_neural_mechanics.reference_posterior_extension import copy_proposal,sha


def test_exact_existing_recovery_proposal_metadata_copy():
    source=Path('runs/compositional-neural-mechanics-v16/reference-posterior-recovery/proposal.json')
    previous=json.loads(source.read_text());original=json.loads(source.read_text())
    assert 'reused_proposal_sha256' in previous
    result=copy_proposal(previous,'newplan',sha(source))
    assert result['center']==previous['center'] and result['covariance']==previous['covariance']
    assert result['plan_sha256']=='newplan' and result['reused_proposal_sha256']==sha(source)
    assert result['ancestor_proposal_sha256']==previous['reused_proposal_sha256']
    assert previous==original
