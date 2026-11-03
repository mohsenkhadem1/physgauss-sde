import pytest
from scripts.complex_scenes_v18.held_protocol import draft
from scripts.complex_scenes_v18.freeze_protocol import check


def test_freeze_requires_every_supported_development_slot_to_be_terminal():
    protocol=draft();rows=[]
    for family in protocol['families']:
        for v in (0,1):
            for method,settings in protocol['methods'].items():rows.append(dict(case=family['family'],variant=v,method=method,state='complete'if family['family']in settings['families']else'unsupported'))
    decision=dict(status='reviewed',rationale='Toy decision for validation test',methods={m:dict(calibration='fixed test budget',dt_s=protocol['methods'][m]['dt_s'])for m in protocol['methods']},terminal_controls=[dict(path='test')])
    check(protocol,dict(rows=rows),decision)
    rows[0]['state']='pending'
    with pytest.raises(ValueError,match='pending'):check(protocol,dict(rows=rows),decision)
    rows[0]['state']='unsupported'
    with pytest.raises(ValueError,match='coverage'):check(protocol,dict(rows=rows),decision)


def test_freeze_rejects_timestep_that_workers_would_ignore():
    from copy import deepcopy
    p=draft();rows=[dict(case=f['family'],variant=v,method=m,state='complete'if f['family']in s['families']else'unsupported')for f in p['families']for v in (0,1)for m,s in p['methods'].items()]
    decision=dict(status='reviewed',rationale='test',methods={m:dict(calibration='test',dt_s=s['dt_s'])for m,s in p['methods'].items()},terminal_controls=[dict(path='test')])
    decision['methods']['ours']['dt_s']=.000025
    with pytest.raises(ValueError,match='not implemented'):check(p,dict(rows=rows),decision)
