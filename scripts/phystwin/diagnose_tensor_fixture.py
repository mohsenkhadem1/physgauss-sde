"""Independent-element diagnostic only; does not qualify or replace campaign truth."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.tensor_fixture import TensorFixture
from scripts.phystwin.consistent_fixture import query
from scripts.phystwin.qualified_reference import BASE,ACTIONS,protocol,release,solve,checked_compare
from scripts.phystwin.qualify_fixture import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);p.add_argument('--x-power',type=float,choices=[1.5,2.],default=1.5);p.add_argument('--levels',type=int,nargs='+',default=[4,6,8,12,16,20]);p.add_argument('--device',default='cuda:0');p.add_argument('--edge-grading',action='store_true');p.add_argument('--actions',choices=ACTIONS,nargs='+',default=['initial']);a=p.parse_args()
    spec=protocol();r=spec['development'][a.case]
    tag='tensor-edges-diagnostic' if a.edge_grading else ('tensor-diagnostic' if a.x_power==1.5 else 'tensor-power2-diagnostic')
    out=BASE/tag/f'development-{a.case}';out.mkdir(parents=True,exist_ok=True)
    previous=None;old=[]
    for n in a.levels:
        release(previous);m=TensorFixture(n,r['E'],r['K'],r['layout'],device=a.device,x_power=2. if a.edge_grading else a.x_power,y_grading=a.edge_grading,z_refinement=4 if a.edge_grading else 1);m.progress=True;rows=[]
        for j,action in enumerate(a.actions):
            f=out/f'n{n}-{action}.npz';started=time.time()
            if f.exists():
                with np.load(f) as z:row={k:z[k] for k in z.files}
            else:
                u=query(previous,old[j]['u'],m.rest) if old else None
                state=solve(m,action,u);u=state['displacement_local_m'];row=dict(u=u,**metrics(m,u,state))
                tmp=f.with_suffix('.tmp.npz');np.savez_compressed(tmp,**row);tmp.replace(f)
            if old:print('COMPARE',n,action,json.dumps(checked_compare(old[j],row,spec,r,action)),flush=True)
            print('FINISHED',n,action,time.time()-started,flush=True);rows.append(row)
        old=rows;previous=m
    release(previous)

if __name__=='__main__':main()
