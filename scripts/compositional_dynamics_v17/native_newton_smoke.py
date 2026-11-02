"""CPU-only upstream Newton API smoke; no contact or accuracy qualification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,importlib.metadata,time
from pathlib import Path
import numpy as np
import warp as wp
import newton
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/compositional-dynamics-v17/newton-smoke';REPORT=ROOT/'reports/compositional-dynamics-v17/native-newton-smoke.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if REPORT.exists():raise ValueError('Preserve previous smoke')
 OUT.mkdir(parents=True,exist_ok=True);wp.config.kernel_cache_dir=str(OUT/'warp-cache');wp.init();start=time.monotonic()
 b=newton.ModelBuilder(gravity=(0.,0.,-9.81));X=np.array([[0,0,.1],[.01,0,.1],[0,.01,.1],[0,0,.11]],float)
 for x in X:b.add_particle(tuple(x),(0.,0.,0.),.0001,radius=.0001)
 b.add_tetrahedron(0,1,2,3,k_mu=1000.,k_lambda=1000.,k_damp=0.);b.color();m=b.finalize(device='cpu');solver=newton.solvers.SolverVBD(m,iterations=8,particle_enable_self_contact=False,particle_enable_tile_solve=False,rigid_compliant_alm=True);a=m.state();z=m.state();control=m.control();q=[a.particle_q.numpy().copy()]
 for _ in range(8):a.clear_forces();solver.step(a,z,control,None,.001);a,z=z,a;q.append(a.particle_q.numpy().copy())
 q=np.array(q);Dm=(X[1:]-X[0]).T;J=[float(np.linalg.det((x[1:]-x[0]).T@np.linalg.inv(Dm)))for x in q];passed=bool(np.isfinite(q).all()and min(J)>0 and q[-1,:,2].mean()<q[0,:,2].mean());np.savez_compressed(OUT/'trajectory.npz',times_s=np.arange(9)*.001,positions=q,rest=X)
 site=Path(newton.__file__).parent;files=[site/'_src/solvers/vbd/solver_vbd.py',site/'_src/solvers/vbd/particle_vbd_kernels.py',site/'_src/sim/builder.py'];result=dict(passed=passed,scope='Native CPU VBD API execution only, one freefall tetrahedron/eight1mssteps; no contact, constitutive matching or impact qualification',versions={n:importlib.metadata.version(n)for n in ('newton','warp-lang','numpy')},device='cpu',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],nodes=4,tets=1,minimum_J=min(J),center_drop_m=float(q[0,:,2].mean()-q[-1,:,2].mean()),seconds=time.monotonic()-start,trajectory_sha256=sha(OUT/'trajectory.npz'),source_sha256={str(p):sha(p)for p in files+[Path(__file__)]});REPORT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
