import numpy as np
import pytest
from experiments import run_campaign_c38_prediction_evaluation as e


def test_truth_member_extraction_never_materializes_probe_measurements():
    keys=set()
    values=dict(object_id=np.asarray('0'*32),private_configuration_index=np.asarray(0,dtype=np.int64),
                repeat_index=np.asarray(0,dtype=np.int64),young_Pa=np.asarray(100000.),
                support_category=np.asarray('both_ends'),canonical_target_Fz_N=np.asarray(1.),
                canonical_target_point_ids=np.arange(63),canonical_target_positions_m=e.metrics.POINTS.copy())
    class PoisonedFuture:
        def __getitem__(self,k):
            keys.add(k)
            if k not in e.metrics.TRUTH_KEYS: raise AssertionError('Future outcomes leaked')
            return values[k]
    truth=e.extract_truth_members(PoisonedFuture())
    assert keys==e.metrics.TRUTH_KEYS and truth['private_configuration_index']==0


def test_choice_gate_requires_all64_and_exact_initial_seal():
    ids={f'{i:032x}' for i in range(64)};initial=dict(path='initial.json',sha256='a'*64)
    rows=[dict(object_id=t,choices=dict(information=-2.,target_risk=2.,fixed_down=-2.,fixed_up=2.,random=2.)) for t in ids]
    seal=dict(completed=True,count=64,private_probe_outcomes_access=False,initial_prediction_seal=initial,choices=rows)
    assert len(e.verify_choices(seal,initial,ids))==64
    with pytest.raises(ValueError):e.verify_choices(seal,dict(path='different',sha256='b'*64),ids)
    seal['choices']=rows[:-1]
    with pytest.raises(ValueError):e.verify_choices(seal,initial,ids)


def test_hash_bound_saved_arrays_roundtrip_and_tamper_failure(tmp_path):
    d=tmp_path/'prediction';d.mkdir()
    expected=dict(completed=True,methods=dict(adaptive=dict(mean=np.arange(6).reshape(2,3))))
    e.io.save_full_result(d,expected)
    files={n:e.io.record(tmp_path,d/n) for n in ('posterior-arrays.npz','posterior-structure.json','numerics.json')}
    actual=e.restore_prediction(tmp_path,files)
    np.testing.assert_array_equal(actual['methods']['adaptive']['mean'],expected['methods']['adaptive']['mean'])
    (d/'posterior-structure.json').write_text('{}')
    with pytest.raises(ValueError,match='Changed input'):e.restore_prediction(tmp_path,files)
