"""Native GIC entry with explicit held-view/time observation partition."""
import sys,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[2]
vendor=R/'runs/gap-crossmethod-v02/vendor/gic'
sys.path.insert(0,str(vendor))
from scene import dataset_readers
original=dataset_readers.readPACNeRFInfo
def partition(*args,**kwargs):
 info=original(*args,**kwargs)
 train=[c for c in info.train_cameras if c.uid!=0 and c.fid<8/24-1e-8]
 test=[c for c in info.test_cameras if c.uid==0]
 if kwargs.get('read_cam',True):assert len(train)==80 and len(test)==14,(len(train),len(test))
 return info._replace(train_cameras=train,test_cameras=test)
dataset_readers.sceneLoadTypeCallbacks['PacNeRF']=partition
if __name__=='__main__':
 runpy.run_path(str(vendor/'train_dynamic.py'),run_name='__main__')
