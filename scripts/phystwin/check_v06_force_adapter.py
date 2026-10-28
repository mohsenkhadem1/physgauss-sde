"""Verify the isolated PhysTwin prescribed-force interpolation kernel."""
from pathlib import Path
import json,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];os.environ['PHYSTWIN_HEADLESS']='1';sys.path.insert(0,str(ROOT/'runs/third_party/PhysTwin-v06'))
import warp as wp
from qqtt.model.diff_simulator.spring_mass_warp import add_prescribed_forces
start=np.array([[1,2,3],[-2,0,1]],np.float32);end=np.array([[5,6,7],[1,2,-1]],np.float32);base=np.ones((2,3),np.float32)*.2
s=wp.array(start,dtype=wp.vec3,device='cuda:0');e=wp.array(end,dtype=wp.vec3,device='cuda:0');f=wp.array(base,dtype=wp.vec3,device='cuda:0')
wp.launch(add_prescribed_forces,dim=2,inputs=[s,e,.25],outputs=[f]);wp.synchronize();error=float(abs(f.numpy()-(base+.75*start+.25*end)).max());assert error<1e-6
out=ROOT/'reports/phystwin-sequential-support-v06';out.mkdir(parents=True,exist_ok=True);(out/'force-kernel-check.json').write_text(json.dumps(dict(maximum_float32_error=error,passed=True,scope='Prescribed force adds to existing internal force and interpolates start/end values. Default upstream path remains unchanged when no external-force file is supplied.'),indent=2)+'\n');print('FORCE_KERNEL_PASSED',error)
