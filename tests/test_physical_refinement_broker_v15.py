import json
import numpy as np
import pytest
from scripts.physical_refinement_v15.measurement_broker import MeasurementBroker,digest


def make(tmp_path):
    packets={}
    for name in ('initial','a','b','unselected'):
        p=tmp_path/(name+'.npz');np.savez(p,observation=np.array([1.,2.]))
        packets[name]=dict(path=p.name,sha256=digest(p),dimension=2)
    manifest=tmp_path/'packets.json'
    manifest.write_text(json.dumps(dict(role='synthetic_observed_packets',packets=packets)))
    return manifest


def test_only_selected_packets_are_read_and_budget_is_enforced(tmp_path):
    manifest=make(tmp_path);broker=MeasurementBroker(manifest,digest(manifest),packet_root=tmp_path)
    # A nonexistent unselected outcome must not be loaded eagerly.
    (tmp_path/'unselected.npz').unlink()
    with pytest.raises(ValueError):broker.request('a')
    broker.request('initial');broker.request('a')
    with pytest.raises(ValueError):broker.request('a')
    broker.request('b')
    with pytest.raises(ValueError):broker.request('unselected')
    assert broker.receipt()['consumed']==['initial','a','b']


def test_changed_packet_rejected_without_consuming_budget(tmp_path):
    manifest=make(tmp_path);broker=MeasurementBroker(manifest,digest(manifest),packet_root=tmp_path)
    np.savez(tmp_path/'initial.npz',observation=np.array([3.,4.]))
    with pytest.raises(ValueError):broker.request('initial')
    assert broker.history==[]
