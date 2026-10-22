import copy,json,hashlib,ast
from pathlib import Path
import numpy as np
import pytest
from experiments import isaac_solver_ablation_v02 as study

def capture(delta=0.):
 return dict(rest_m=np.zeros((2,3)),tets=np.array([[0,1,1,1]]),faces=np.array([[0,1,1]]),fixed_node_ids=np.array([0]),
  times_s=np.arange(19)/30,positions_m=np.full((19,2,3),delta))

def test_exact_factorial_and_comparison():
 assert [(c['hz'],c['iterations']) for c in study.cases()]==[(240,32),(240,128),(480,32),(480,128)]
 data={(h,i):capture((h==480)*1e-4+(i==32)*1e-3) for h in (240,480) for i in (32,128)}
 r=study.compare(data)
 assert len(r['comparisons'])==4 and not r['all_pairs_pass']
 np.testing.assert_allclose([x['maximum_material_query_difference_m'] for x in r['comparisons']],np.sqrt(3)*np.array([1e-3,1e-3,1e-4,1e-4]))
 with pytest.raises(ValueError):study.compare({(240,32):capture()})
 data[(480,128)]['times_s'][1]+=.001
 with pytest.raises(AssertionError):study.compare(data)

def test_worker_guard_rejects_undeclared_settings(tmp_path,monkeypatch):
 stage={'source_sha256':{'source':'hash'}};base={'output':'old','physics_hz':240,'cases':[],'E':300000}
 r={'path':'stage','sha256':'sha'};case=study.cases()[0]
 cell=study.cell_protocol(stage,base,case,r)
 monkeypatch.setattr(study,'ROOT',tmp_path);monkeypatch.setattr(study,'validate_stage',lambda *args:(stage,base))
 p=tmp_path/'cell.json'
 def check(value):
  p.write_text(json.dumps(value));digest=hashlib.sha256(p.read_bytes()).hexdigest()
  return study.validate_worker_protocol('cell.json',digest)
 assert check(cell)==cell
 for mutate in (lambda x:x.update(physics_hz=960),lambda x:x.update(gpu=0),lambda x:x.update(E=300001),lambda x:x['cases'][0].update(iterations=255)):
  wrong=copy.deepcopy(cell);mutate(wrong)
  with pytest.raises(ValueError):check(wrong)

def test_worker_only_changes_declared_gpu_and_iterations():
 source=Path(study.ROOT/'experiments/isaac_solver_ablation_worker_v02.py').read_text();ast.parse(source)
 assert 'active_cuda_gpus=[0],physics_gpu=0,limit_cpu_threads=2' in source and "device='cuda:0'" in source
 assert "solverPositionIterationCount=case['iterations']" in source
 assert "if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-c49707f6-24fd-38e4-3973-3543ec644ceb'" in source
 assert "row.split(',')[0].strip()=='1'" in source
 assert "Worker compute must be solely physicalGPU1" in source
 assert 'linearDamping=0.' in source and "'physxDeformableMaterial:elasticityDamping':0." in source

def test_nonfinite_or_incomplete_trajectory_rejected():
 data={(h,i):capture() for h in (240,480) for i in (32,128)}
 data[(480,128)]['positions_m'][1,1,1]=np.nan
 with pytest.raises(ValueError):study.compare(data)
