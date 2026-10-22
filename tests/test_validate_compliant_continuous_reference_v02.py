import json
import numpy as np
from experiments import validate_compliant_continuous_reference_v02 as continuation


def test_numpy_scalar_nested_reports_are_json_safe(tmp_path):
 path=tmp_path/'report.json'
 continuation.write(path,{'passed':np.bool_(True),'nested':[np.int64(4),np.float64(.2)],'array':np.array([1.,2.]),'_private':np.zeros(3)})
 assert json.loads(path.read_text())=={'passed':True,'nested':[4,.2],'array':[1.,2.]}


def test_failure_metadata_remains_durable_with_numpy_progress(tmp_path):
 path=tmp_path/'failure.json'
 continuation.save_failure(path,TypeError('original reporting exception'),completed_sweeps=np.int64(42),numerical_check=np.bool_(False))
 value=json.loads(path.read_text())
 assert value['completed'] is False and value['error_type']=='TypeError' and value['completed_sweeps']==42
 assert value['numerical_check'] is False
