import pytest
from scripts.compositional_dynamics_v17.fine_refinement import normalized,tasks,analyze

def test_failed_worker_row_retained_and_unqualified():
 t=tasks('release',6)[0];r=normalized(dict(task=t,task_id=t['id'],completed=False,error='Synthetic initialization failure',path='runs/nonexistent-fixture'),t)
 assert r['case']=='release'and r['n']==6
 a=analyze([r],'release',4,6)
 assert not a['passed']and len(a['checks'])==5
 assert all(not c['passed']for c in a['checks'])

def test_conflicting_identity_rejected():
 t=tasks('drop',6)[0]
 with pytest.raises(ValueError):normalized(dict(task_id=t['id'],case='release'),t)
