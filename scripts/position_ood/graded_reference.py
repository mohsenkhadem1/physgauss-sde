"""Independent graded-mesh reference track; never overwrites primary references."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood import study
from scripts.position_ood.graded import GradedMaterialFixture
from scripts.phystwin.tensor_fixture import basis

def configure():
    original=study.protocol();base=study.BASE;uniform_interpolate=study.interpolate_samples
    study.BASE=base/'graded-reference';study.BASE.mkdir(exist_ok=True)
    study.protocol=lambda:original
    study.MaterialPositionFixture=GradedMaterialFixture
    def interpolate(field,query,n=1):
        if n==1:return uniform_interpolate(field,query,n)
        s=np.linspace(-1,1,n+1);axes=[.12*(1-np.cos(np.pi*np.linspace(0,1,4*n+1))),-.04*np.cos(np.pi*np.linspace(0,1,n+1)),np.sign(s)*.04*(1-np.cos(np.pi*abs(s)))/2]
        cell=np.stack([np.clip(np.searchsorted(a,query[:,i],side='right')-1,0,len(a)-2) for i,a in enumerate(axes)],axis=1)
        lo=np.stack([a[cell[:,i]] for i,a in enumerate(axes)],axis=1);width=np.stack([np.diff(a)[cell[:,i]] for i,a in enumerate(axes)],axis=1);N,_=basis(2*(query-lo)/width-1)
        offset=np.stack(np.meshgrid(*[np.arange(3)]*3,indexing='ij'),-1).reshape(-1,3);ids=np.ravel_multi_index((2*cell[:,None]+offset).transpose(2,0,1),(8*n+1,2*n+1,2*n+1))
        return np.einsum('qa,qai->qi',N,field[ids])
    study.interpolate_samples=interpolate
    # The alternate geometry is recorded independently of primary cache hashes.
    import hashlib
    stamp={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('graded.py')]};path=study.BASE/'geometry-source.json'
    if path.exists():assert json.loads(path.read_text())==stamp,'Alternate geometry changed'
    else:path.write_text(json.dumps(stamp,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--object',type=int,default=3);a=p.parse_args();configure();study.reference(a.split,a.object)
