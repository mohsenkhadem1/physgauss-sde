"""Predeclared finite spatial-hypothesis mechanics dictionary (raw arrays stay local)."""
from pathlib import Path
import argparse,itertools,json,time,hashlib
import numpy as np
import torch
from .mechanics import LinearGaussianBody,PARAMETERS
ROOT=Path(__file__).resolve().parents[2]
TIMES=np.linspace(0,1,8)
NOISE=.00002
FORCE_STD=.08
PROBE_FORCE=.08
BOUNDS=np.array([[np.log(8000),np.log(32000)],[-.8,.8],[-.025,.025],[np.log(4000),np.log(16000)],[-.04,.04]])

def points():
    p=list(itertools.product([-.045,-.015,.015,.045],[-.025,.025],[.015,.025]))
    p+=list(itertools.product([-.045,-.015],[-.025,.025],[.05]))
    return np.array(p)

def grid(levels=5):
    return np.array(list(itertools.product(*[np.linspace(a,b,levels) for a,b in BOUNDS])))

def load_cov(scale=1.):
    d=np.abs(TIMES[:,None]-TIMES[None,:]);r=np.sqrt(3)*d/.30
    return FORCE_STD**2*scale**2*np.kron((1+r)*np.exp(-r),np.eye(2))

def probe_specs():
    return [dict(x=x,mode=mode) for x in [-.04,0.,.04,-.02,.02] for mode in ['top','side']]

def build(shape,theta,cells=(8,4,4),device='cpu',batch=16):
    body=LinearGaussianBody(shape,cells=cells,device=device);p=points();interp=body.interpolation(p)
    units=[];probes=[];materials=[];start=time.time()
    for i in range(0,len(theta),batch):
        th=torch.tensor(theta[i:i+batch],dtype=torch.float64,device=device)
        with torch.no_grad():
            out=body.solve(th,probe_specs=probe_specs())
            units.append(torch.einsum('pn,bncl->bpcl',interp,out['unit_response']).cpu().numpy())
            probes.append(torch.einsum('pn,bncl->bpcl',interp,out['probe_response']).cpu().numpy())
            materials.append(body.modulus_at_points(th,p).cpu().numpy())
        if i%(batch*20)==0:print(f'{shape} {i}/{len(theta)} elapsed {time.time()-start:.1f}s',flush=True)
    return dict(theta=theta,unit=np.concatenate(units),probe=np.concatenate(probes),material=np.concatenate(materials),points=p,cells=np.array(cells)),dict(shape=shape,cells=list(cells),nodes=body.n,elements=len(body.elements),seconds=time.time()-start,hypotheses=len(theta),device=device)

def maps(data):
    unit,probe=data['unit'],data['probe'];n=len(unit);nf=16
    H=[]
    for t in [0,3,7]:
      for p in [1,5,9,13]:
       for axis in [0,2]:
        h=np.zeros((n,nf));h[:,2*t:2*t+2]=unit[:,p,axis];H.append(h)
    H=np.stack(H,1)
    hidden=np.array([p for p in np.flatnonzero(data['points'][:,1]>0) if p not in (3,11)])
    T=np.zeros((n,8*len(hidden)*3+4*16*3,nf));offset=np.zeros(T.shape[:2]);j=0
    for t in range(8):
      for p in hidden:
       for axis in range(3):
        T[:,j,2*t:2*t+2]=unit[:,p,axis];j+=1
    split=j
    for q in range(6,10):
      for p in range(16):
       for axis in range(3):
        offset[:,j]=PROBE_FORCE*probe[:,p,axis,q];j+=1
    actions=[]
    for t in [1,4,6]:
      for p in [3,11]:
       h=np.zeros((n,nf));h[:,2*t:2*t+2]=unit[:,p,1]
       actions.append(dict(name=f'view_y_t{t}_p{p}',kind='view',h=h,offset=np.zeros(n),time=t,point=p,axis=1))
    for t in [2,5]:
      for p in [1,13]:
       h=np.zeros((n,nf));h[:,2*t:2*t+2]=unit[:,p,2]
       actions.append(dict(name=f'view_z_t{t}_p{p}',kind='view',h=h,offset=np.zeros(n),time=t,point=p,axis=2))
    for q,spec in enumerate(probe_specs()[:6]):
       axis=2 if spec['mode']=='top' else 1
       actions.append(dict(name=f'probe_{q}',kind='probe',h=np.zeros((n,nf)),offset=PROBE_FORCE*probe[:,13,axis,q],probe=q,point=13,axis=axis))
    return dict(H=H,T=T,offset=offset,split=split,actions=actions)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shape',choices=['block','step'],required=True);ap.add_argument('--levels',type=int,default=5);ap.add_argument('--cells',type=int,nargs=3,default=[8,4,4]);ap.add_argument('--device',default='cpu');ap.add_argument('--batch',type=int,default=16);ap.add_argument('--output',required=True);args=ap.parse_args()
    torch.set_num_threads(1);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    data,meta=build(args.shape,grid(args.levels),tuple(args.cells),args.device,args.batch);np.savez(out,**data)
    meta.update(parameters=PARAMETERS,bounds=BOUNDS.tolist(),levels=args.levels,noise_m=NOISE,force_prior_std_N=FORCE_STD,probe_force_N=PROBE_FORCE,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
if __name__=='__main__':main()
