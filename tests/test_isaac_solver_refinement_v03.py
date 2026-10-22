import copy,json,hashlib,ast
from pathlib import Path
import numpy as np
import pytest
from experiments import isaac_solver_refinement_v03 as study

def capture(delta=0.):
 return dict(rest_m=np.zeros((2,3)),tets=np.array([[0,1,1,1]]),faces=np.array([[0,1,1]]),fixed_node_ids=np.array([0]),times_s=np.arange(19)/30,positions_m=np.full((19,2,3),delta))

def test_exact_scope_and_separate_temporal_iteration_gates():
 assert [(c['hz'],c['iterations']) for c in study.cases()]==[(960,128),(1920,128),(3840,128),(3840,255)]
 data={(480,128):capture(.01),**{(c['hz'],c['iterations']):capture() for c in study.cases()}}
 r=study.compare(data)
 assert not r['comparisons'][0]['passed'] and r['finest_declared_numerical_checks_passed']
 data[(3840,255)]=capture(.001)
 assert not study.compare(data)['finest_declared_numerical_checks_passed']
 data[(3840,255)]['times_s'][1]+=.0001
 with pytest.raises(AssertionError):study.compare(data)

def test_strict_worker_protocol_no_physical_changes(tmp_path,monkeypatch):
 stage={'source_sha256':{'file':'hash'}};base={'cases':[],'physics_hz':240,'rho':625};ref={'path':'stage','sha256':'hash'}
 monkeypatch.setattr(study,'ROOT',tmp_path);monkeypatch.setattr(study,'validate_stage',lambda *a:(stage,base))
 p=tmp_path/'cell.json';cell=study.cell_protocol(stage,base,study.cases()[0],ref)
 def check(v):
  p.write_text(json.dumps(v));return study.validate_worker_protocol('cell.json',hashlib.sha256(p.read_bytes()).hexdigest())
 assert check(cell)==cell
 for mutate in (lambda d:d.update(rho=626),lambda d:d.update(physics_hz=480),lambda d:d['cases'][0].update(iterations=256),lambda d:d.update(gpu=0)):
  d=copy.deepcopy(cell);mutate(d)
  with pytest.raises(ValueError):check(d)

def test_worker_reuses_exact_v02_physics_resource_guard():
 source=Path(study.ROOT/'experiments/isaac_solver_refinement_worker_v03.py').read_text();old=Path(study.ROOT/'experiments/isaac_solver_ablation_worker_v02.py').read_text()
 assert source==old.replace('from isaac_solver_ablation_v02 import','from isaac_solver_refinement_v03 import')
 ast.parse(source)
 assert 'active_cuda_gpus=[0],physics_gpu=0,limit_cpu_threads=2' in source
