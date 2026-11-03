import pytest
from scripts.complex_scenes_v18.held_protocol import draft
from scripts.complex_scenes_v18.freeze_protocol import check


def test_freeze_requires_every_supported_development_slot_to_be_terminal():
    protocol=draft();rows=[]
    for family in protocol['families']:
        for v in (0,1):
            for method,settings in protocol['methods'].items():rows.append(dict(case=family['family'],variant=v,method=method,state='complete'if family['family']in settings['families']else'unsupported'))
    decision=dict(status='reviewed',rationale='Toy decision for validation test',methods={m:dict(calibration='fixed test budget',dt_s=.000025)for m in protocol['methods']},terminal_controls=[dict(path='test')])
    check(protocol,dict(rows=rows),decision)
    rows[0]['state']='pending'
    with pytest.raises(ValueError,match='pending'):check(protocol,dict(rows=rows),decision)
    rows[0]['state']='unsupported'
    with pytest.raises(ValueError,match='coverage'):check(protocol,dict(rows=rows),decision)
