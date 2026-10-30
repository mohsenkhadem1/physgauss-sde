"""Poison hidden target channels in memory and verify identical predictor inputs."""
from pathlib import Path
import sys,pickle,json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,h5py
from scripts.external_benchmark.data import softvt,phystwin,predictor_payload
from scripts.external_benchmark.common import REPORT,write_json

def identical(a,b):
    assert a.keys()==b.keys()
    for key,value in a.items():
        if isinstance(value,np.ndarray):assert np.array_equal(value,b[key],equal_nan=True),key
        else:assert value==b[key],key

class Dataset:
    def __init__(self,dataset,start):self.dataset=dataset;self.start=start
    def __getitem__(self,key):
        x=self.dataset[key].copy();x[self.start:]=1e6
        return x
class Group:
    def __init__(self,group,start):self.group=group;self.start=start;self.attrs=group.attrs
    def __getitem__(self,key):
        item=self.group[key]
        return Dataset(item,self.start) if key.endswith('/nodal_pos_w') else item
class File:
    def __init__(self,file,start):self.file=file;self.start=start
    def __enter__(self):return self
    def __exit__(self,*args):self.file.close()
    def __getitem__(self,key):return Group(self.file[key],self.start)

rows=[]
for subset in ['object-soft','spatial-soft']:
    original=softvt(subset,0,1);real_file=h5py.File
    with patch('h5py.File',lambda *args,**kwargs:File(real_file(*args,**kwargs),original['train_end'])):
        poisoned=softvt(subset,0,1)
    assert not np.array_equal(original['evaluation']['points'],poisoned['evaluation']['points'])
    identical(predictor_payload(original),predictor_payload(poisoned))
    rows.append(dict(dataset='SoftVTBench',name=original['name'],check='All future nodal truth replaced by 1e6; predictor payload exactly unchanged',passed=True))
original=phystwin('single_lift_sloth',300);real_load=pickle.load

def poison(file,*args,**kwargs):
    value=real_load(file,*args,**kwargs)
    if str(file.name).endswith('final_data.pkl'):
        value['object_points'][original['train_end']:]=1e6
    elif str(file.name).endswith('gt_track_3d.pkl'):value[:]=1e6
    return value
with patch('pickle.load',poison):poisoned=phystwin('single_lift_sloth',300)
identical(predictor_payload(original),predictor_payload(poisoned))
rows.append(dict(dataset='PhysTwin',name=original['name'],check='Future object tracks and all manual marker targets replaced by 1e6; predictor payload exactly unchanged',passed=True))
from scripts.external_benchmark.contact_adapter import softvt as local_softvt
from scripts.external_benchmark.calibrated_baseline import calibrated_rigid
for adapter,label in [(softvt,'original'),(local_softvt,'local-contact development candidate')]:
    for subset in ['object-soft','spatial-soft']:
        original=adapter(subset,0,1);real_file=h5py.File
        with patch('h5py.File',lambda *args,**kwargs:File(real_file(*args,**kwargs),original['train_end'])):
            poisoned=adapter(subset,0,1)
        identical(predictor_payload(original),predictor_payload(poisoned))
        assert np.array_equal(calibrated_rigid(original),calibrated_rigid(poisoned))
        rows.append(dict(dataset='SoftVTBench',name=original['name'],adapter=label,check='Future target poisoning leaves predictor payload and prefix-calibrated rigid prediction exactly unchanged',passed=True))
write_json(REPORT/'input-isolation-audit.json',rows)
print(json.dumps(rows,indent=2))
