import numpy as np
from scripts.compositional_neural_mechanics_reporting.stable_zero_io_recovery import mkdir_npz,implementation,OUT,original

def test_real_nested_output_atomic_write(tmp_path):
 path=tmp_path/'prediction'/'new-scene'/'new-command'/'model.npz'
 assert not path.parent.exists()
 q=np.arange(12,dtype=float).reshape(4,3);mkdir_npz(path,material_query=q,displacement=q*.01,wrenches=np.ones((2,6)))
 assert path.exists()and not path.with_name(path.name+'.pending.npz').exists()
 with np.load(path,allow_pickle=False)as saved:
  assert np.array_equal(saved['material_query'],q)
  assert np.array_equal(saved['displacement'],q*.01)
  assert np.array_equal(saved['wrenches'],np.ones((2,6)))

def test_private_clone_preserves_original_paths_and_predictor():
 original_out=original.OUT;original_writer=original.npz;copy=implementation()
 assert copy.OUT==OUT and copy.COMMIT==OUT/'predictions-commit.json'
 assert original.OUT==original_out and original.npz is original_writer
 assert copy.direct_forward is original.direct_forward
 assert copy.predict.__code__.co_code==original.predict.__code__.co_code
