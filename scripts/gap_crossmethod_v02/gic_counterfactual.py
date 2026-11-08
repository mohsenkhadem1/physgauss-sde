"""Conditional native GIC solver reference on the same reconstructed particles.
No new-interaction measured ground truth, and no claim of OOD benchmark accuracy.
Each case runs in its own process: Taichi does not release simulator fields, and six
simulators in one process exhaust the 11GB device (first attempt aborted, log retained).
Usage: gic_counterfactual.py            -> runs all cases as subprocesses, then combines
       gic_counterfactual.py CASE       -> runs one case (condition_material)
"""
import sys,json,copy,subprocess,os
from pathlib import Path
from argparse import Namespace
import numpy as np
R=Path(__file__).resolve().parents[2]
MODEL=R/'runs/gap-crossmethod-v02/gic-output/torus';OUT=MODEL/'counterfactual'
CONDITIONS=('original','half_gravity','transverse_velocity');MATERIALS=('inferred','paper_reference')

def run_case(case):
 sys.path.insert(0,str(R))
 from scripts.gap_crossmethod_v02 import gic_native_entry
 import torch,taichi as ti
 from simulator import Simulator
 from new_trajectory import load_pcd_file,gen_xyz_list
 condition,material=case.rsplit('_',1) if not case.endswith('paper_reference') else (case[:-len('_paper_reference')],'paper_reference')
 cfg=json.loads((MODEL/'0-pred.json').read_text());p=copy.deepcopy(cfg)
 if condition=='half_gravity':p['gravity']=[0,-4.9,0]
 if condition=='transverse_velocity':p['vel']=(np.asarray(p['vel'])+np.array([.5,0,0])).tolist()
 if material=='paper_reference':p['mat_params'].update(E=1e6,nu=.3)
 ti.init(arch=ti.cuda,debug=False,fast_math=False,device_memory_fraction=.5)
 vol=load_pcd_file(str(MODEL),40000);sim=Simulator(Namespace(**p),vol);seq=gen_xyz_list(sim,14,diff=False)
 if not sim.succeed() or not all(torch.isfinite(x).all() for x in seq):raise RuntimeError(f'Native CFL/nonfinite failure retained: {case}')
 OUT.mkdir(exist_ok=True);np.savez_compressed(OUT/f'{case}.npz',xyz=np.stack([x.cpu().numpy() for x in seq]),settings=json.dumps(p))

def main():
 if len(sys.argv)>1:return run_case(sys.argv[1])
 cwd=R/'runs/gap-crossmethod-v02/vendor/gic'
 for c in CONDITIONS:
  for m in MATERIALS:
   case=f'{c}_{m}'
   if (OUT/f'{case}.npz').exists():continue
   q=subprocess.run([sys.executable,'-u',__file__,case],cwd=cwd,env=os.environ.copy())
   if q.returncode:raise SystemExit(f'case {case} failed rc={q.returncode}')
 cfg=json.loads((MODEL/'0-pred.json').read_text());arrays={};rows=[]
 for c in CONDITIONS:
  a=np.load(OUT/f'{c}_inferred.npz')['xyz'];b=np.load(OUT/f'{c}_paper_reference.npz')['xyz'];arrays[c+'_inferred']=a;arrays[c+'_paper_reference']=b
  assert np.array_equal(a[0],b[0])
  err=np.linalg.norm(a-b,axis=-1)
  rows.append(dict(condition=c,fit_prefix_particle_rms_m=float(np.sqrt(np.mean(err[:8]**2))),future_particle_rms_m=float(np.sqrt(np.mean(err[8:]**2))),final_particle_rms_m=float(np.sqrt(np.mean(err[-1]**2)))))
 np.savez_compressed(MODEL/'conditional-counterfactual.npz',**arrays)
 report=dict(scope='Conditional native-solver reference, not independent OOD ground truth. Both materials use identical reconstructed particles, initial state except specified intervention, configured density, contact law and solver.',paper_reference=dict(E_Pa=1e6,nu=.3,source='https://arxiv.org/html/2303.05512',location='Table1 Torus ground truth; not initialization or numberedGT_table2 samples'),configured_density=cfg['rho'],inferred=cfg['mat_params'],rows=rows,interventions='Native gravity acceleration and initial velocity only; no calibrated-force or position-control claim')
 (R/'reports/gap-crossmethod-v02/gic-counterfactual.json').write_text(json.dumps(report,indent=2));print(json.dumps(rows,indent=1))
if __name__=='__main__':main()
