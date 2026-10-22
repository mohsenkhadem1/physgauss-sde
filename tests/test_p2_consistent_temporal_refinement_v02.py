import numpy as np
import pytest
from experiments.run_p2_consistent_temporal_refinement_v02 import verify_record,eligible_larger_case
from experiments.run_static_p2_axial_continuation import record

def test_binary_reference_hash_validation_does_not_parse_NPZ_as_JSON(tmp_path):
 p=tmp_path/'reference.npz';np.savez(p,field=np.arange(3));r=record(p)
 verify_record(r)
 p.write_bytes(p.read_bytes()+b'changed')
 with pytest.raises(ValueError,match='Changed pinned input'):verify_record(r)

def test_resource_condition_preserved():
 assert eligible_larger_case([30e-6,10e-6],40.,230.)
 assert not eligible_larger_case([30e-6,21e-6],40.,230.)
 assert not eligible_larger_case([5e-6,10e-6],40.,230.)
 assert not eligible_larger_case([30e-6,10e-6],65.,230.)
