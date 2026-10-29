"""Development-only native MPM stability under smooth attachment footprints."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.physgaussian import PositionMPM
from scripts.position_ood.study import BASE,OUT,protocol,within_deadline
from scripts.phystwin.sequential_reference import write
case=protocol()['development'][1];ref=np.load(BASE/'fields/development-1.npz');rows=[];fields={}
for name,width,rpic in [('thin_apic',0.,0.),('thin_pic',0.,-1.),('layer_pic',.006,-1.)]:
 m=PositionMPM(2,64,.0001,rpic=rpic,attachment_width=width,ramp_seconds=1.);m.parameters(**case)
 for seconds in [4.,8.]:
  for action in [0,4,8]:
   within_deadline()
   try:
    s=m.solve(ref['poses'][action],seconds=seconds,block=50);u=s.pop('field');fields[f'{name}-{seconds}-{action}']=u;s.update(action=action,configuration=name,seconds=seconds,numerically_valid=True,reference_rmse_mm=float(1000*np.sqrt(np.mean((u-ref['field'][action])**2))))
    if seconds==8.:s['doubling_change_rms_m']=float(np.sqrt(np.mean((u-fields[f'{name}-4.0-{action}'])**2)))
   except RuntimeError as e:s=dict(action=action,configuration=name,seconds=seconds,numerically_valid=False,error=repr(e))
   rows.append(s);write(OUT/'mpm-smooth-numerical-audit.json',dict(scope='Development oracle physics parameters are supplied only to isolate numerical behavior; native fixed-corotated MPM and Neo-Hookean reference have different constitutive laws. These are not inverse benchmark scores.',rows=rows));np.savez_compressed(BASE/'mpm-smooth-audit.npz',**fields);print('SMOOTH_MPM',s,flush=True)
 del m
