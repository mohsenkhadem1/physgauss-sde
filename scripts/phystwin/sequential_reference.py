"""Resumable v06 multi-support reference and continuous-surrogate bank."""
from pathlib import Path
import argparse,gc,json,os,sys,time,itertools
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.sequential_fixture import SequentialFixture,ACTIONS,LAYOUTS,points
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/phystwin-sequential-support-v06';OUT=ROOT/'reports/phystwin-sequential-support-v06'
BOUNDS=np.log([[1500,10000],[80,4000],[80,4000]])
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def protocol():
 path=BASE/'protocol.json'
 if path.exists():return json.loads(path.read_text())
 spec=dict(version=6,layouts=LAYOUTS,actions=ACTIONS,parameter_bounds_log=BOUNDS.tolist(),bank_grid=4,bank_meshes=[2,4],reference_levels=[4,6,8,10],noise_sigma_m=.00015,noise_repeats=3,posterior_levels=[17,33,65],utility_samples=[64,256],seed=280610,
   reference_gate=dict(displacement_rms_m=2e-5,maximum_coordinate_m=1e-4,residual_N=1e-9,positive_J=.1),
   scope='Two independent end-patch stiffnesses, three spatial support layouts, six reset force probes, two sequential observations, known weak gravity. Static nonlinear Q2 FEM. Numerical errors and scientific improvements reported separately; v05 is unchanged.',
   interpretation='Pilot evidence graded by effect size, paired object intervals and failure cases. No superiority or physical-calibration claim from a single threshold. Reference qualification here concerns sampled displacement, equilibrium and positive Jacobian; traction/strain differences remain diagnostics, not v05 full qualification.')
 for split,count,seed in [('development',6,610),('calibration',12,611),('test',18,612)]:
  rng=np.random.default_rng(seed);spec[split]=[dict(layout=LAYOUTS[i%3],parameters=np.exp(rng.uniform(BOUNDS[:,0],BOUNDS[:,1])).tolist()) for i in range(count)]
 write(path,spec);return spec

def jobs(split):
 s=protocol()
 if split=='bank':
  axes=[np.linspace(*b,s['bank_grid']) for b in BOUNDS]
  return [(f'bank-{l}-{i}-{j}-{k}',dict(layout=l,parameters=np.exp([axes[0][i],axes[1][j],axes[2][k]]).tolist())) for l in LAYOUTS for i,j,k in itertools.product(range(s['bank_grid']),repeat=3)]
 return [(f'{split}-{i}',r) for i,r in enumerate(s[split])]

def solve_level(name,r,n,device,order=3):
 path=BASE/'fields'/f'{name}-n{n}-q{order}.npz'
 if path.exists():
  with np.load(path) as z:return {k:z[k] for k in z.files}
 m=SequentialFixture(n,*r['parameters'],r['layout'],device=device,order=order);rows=[];started=time.time()
 try:
  for action in ACTIONS:
   checkpoint=BASE/'checkpoints'/name/f'n{n}-q{order}-{action}.npz'
   if checkpoint.exists():
    with np.load(checkpoint) as z:row={k:z[k] for k in z.files}
   else:
    m.set_action(action);state=m.solve();row=m.sampled(state)
    assert row['residual_N']<=1e-9 and row['minimum_J']>.1 and row['reaction_balance_N']<=1e-8
    checkpoint.parent.mkdir(parents=True,exist_ok=True);tmp=checkpoint.with_suffix('.tmp.npz');np.savez_compressed(tmp,**row);tmp.replace(checkpoint)
   rows.append(row)
  data={k:np.stack([r[k] for r in rows]) for k in rows[0]};data['elapsed_seconds']=time.time()-started
  path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);tmp.replace(path)
  print('FIELDS',name,n,order,round(time.time()-started,2),flush=True);return data
 finally:
  m.clear_linear_cache();m.engine=None;del m;gc.collect()
  import torch
  if torch.cuda.is_available():torch.cuda.empty_cache()

def compare(a,b):
 delta=a['field']-b['field'];rms=np.sqrt(np.mean(delta**2,axis=(1,2)));maximum=abs(delta).max(axis=(1,2));strain=np.linalg.norm((a['strain']-b['strain']).reshape(8,-1),axis=1)/np.maximum(np.linalg.norm(b['strain'].reshape(8,-1),axis=1),1e-12)
 return dict(rms_m=rms.tolist(),maximum_coordinate_m=maximum.tolist(),strain_relative=strain.tolist(),passed=bool(rms.max()<=2e-5 and maximum.max()<=1e-4))
def reference(name,r,device):
 final=BASE/'fields'/f'{name}-reference.npz';audit=BASE/'qualification'/f'{name}.json'
 if final.exists() and audit.exists():return
 old=None;checks=[];accepted=False
 for n in protocol()['reference_levels']:
  data=solve_level(name,r,n,device)
  if old is not None:
   check=compare(old,data);checks.append(dict(coarse=previous,fine=n,**check))
   if check['passed']:
    quad=solve_level(name,r,n,device,4);qcheck=compare(data,quad)
    if qcheck['passed']:accepted=True;break
  old=data;previous=n
 write(audit,dict(case=r,mesh=n,displacement_qualified=accepted,spatial=checks,quadrature=qcheck if accepted else None))
 if not accepted:raise RuntimeError(f'{name}: displacement reference unresolved; keep diagnostics and refine')
 tmp=final.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);tmp.replace(final)
 print('REFERENCE',name,n,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['bank','development','calibration','test'],required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--mesh',type=int,default=4);p.add_argument('--device',default='cuda:0');a=p.parse_args();protocol()
 for i,(name,r) in enumerate(jobs(a.split)):
  if i%a.workers!=a.worker:continue
  if a.split=='bank':solve_level(name,r,a.mesh,a.device)
  else:reference(name,r,a.device)
if __name__=='__main__':main()
