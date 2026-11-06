"""Verify actual learned PIDG appearance survives the initial mechanical binding."""
import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pidg_comparison.run import configure,EXTERNAL
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();configure(a.gpu);sys.path.insert(0,str(EXTERNAL/'upstream'))
import torch,numpy as np
from scripts.pidg_solver_forecast.common import BASE,DATA,load,camera,native
from scripts.pidg_solver_forecast.model import SceneFEM
torch.set_num_threads(2);out=BASE/a.scene/'solver';g,d=load(a.scene);f=json.loads((DATA/a.scene/'fit.json').read_text())['frames'][0];cam,_,_=camera(a.scene,f);m=SceneFEM(np.load(out/'adapter.npz'),torch.load(out/'appearance-initial.pt',weights_only=False));m.set_binding_tets()
with torch.no_grad():
 ref=native(g,d,cam,f['time'])['render'].clamp(0,1);pred=m.image(m.rest,cam);mse=float((pred-ref).square().mean());maximum=float((pred-ref).abs().max());psnr=-10*np.log10(max(mse,1e-16))
report=dict(passed=bool(psnr>60),scene=a.scene,psnr_between_native_and_initial_bound=psnr,max_rgb_difference=maximum,mse=mse,scope='Actual prefix checkpoint, first fitting camera/time, initial appearance before static refinement. This verifies binding/covariance/SH implementation, not moving-state reconstruction fidelity.')
(ROOT/'reports/pidg-solver-forecast-v01'/a.scene/'identity.json').write_text(json.dumps(report,indent=2));print(report);assert report['passed']
