"""Independent-element diagnostic only; does not qualify or replace campaign truth."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.tensor_fixture import TensorFixture
from scripts.phystwin.consistent_fixture import query
from scripts.phystwin.qualified_reference import BASE,protocol,release,solve,checked_compare
from scripts.phystwin.qualify_fixture import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);p.add_argument('--x-power',type=float,choices=[1.5,2.],default=1.5);p.add_argument('--levels',type=int,nargs='+',default=[4,6,8,12,16,20]);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    spec=protocol();r=spec['development'][a.case]
    out=BASE/('tensor-diagnostic' if a.x_power==1.5 else 'tensor-power2-diagnostic')/f'development-{a.case}';out.mkdir(parents=True,exist_ok=True)
    previous=None;old=None
    for n in a.levels:
        release(previous);m=TensorFixture(n,r['E'],r['K'],r['layout'],device=a.device,x_power=a.x_power);m.progress=True
        f=out/f'n{n}-initial.npz';started=time.time()
        if f.exists():
            with np.load(f) as z:row={k:z[k] for k in z.files}
        else:
            u=query(previous,old['u'],m.rest) if old is not None else None
            state=solve(m,'initial',u);u=state['displacement_local_m'];row=dict(u=u,**metrics(m,u,state))
            tmp=f.with_suffix('.tmp.npz');np.savez_compressed(tmp,**row);tmp.replace(f)
        if old is not None:print('COMPARE',n,json.dumps(checked_compare(old,row,spec,r,'initial')),flush=True)
        print('FINISHED',n,time.time()-started,flush=True);old=row;previous=m
    release(previous)

if __name__=='__main__':main()
