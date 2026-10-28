"""Validate a measured pilot bundle before fitting; no hardware dependency.

NPZ: times_s (T), tracked_points_m (T,N,3), visibility (T,N),
forces_N (T,3), force_application_m (T,3), camera_to_world (T,4,4).
JSON: measured_geometry, nominal_mass_kg, force_sigma_N, episodes.
Each episode: start_frame, end_frame (exclusive), role, object_id, action.
"""
from pathlib import Path
import argparse,json
import numpy as np

def validate(data,manifest):
 t=np.asarray(data['times_s']);x=np.asarray(data['tracked_points_m']);v=np.asarray(data['visibility'])
 assert t.ndim==1 and len(t)>1 and np.isfinite(t).all() and np.all(np.diff(t)>0),'timestamps must strictly increase'
 assert x.ndim==3 and x.shape[0]==len(t) and x.shape[2]==3 and v.shape==x.shape[:2],'track shapes'
 assert v.dtype==bool and np.isfinite(x[v]).all(),'visible tracks must be finite'
 assert float(manifest['nominal_mass_kg'])>0 and float(manifest['force_sigma_N'])>0
 assert manifest['measured_geometry'] is True,'measure rest geometry before fitting'
 for key in ['forces_N','force_application_m']:
  a=np.asarray(data[key]);assert a.shape==(len(t),3) and np.isfinite(a).all(),key
 camera=np.asarray(data['camera_to_world']);assert camera.shape==(len(t),4,4) and np.isfinite(camera).all()
 np.testing.assert_allclose(camera[:,3,:],np.broadcast_to([0,0,0,1],(len(t),4)),atol=1e-6)
 np.testing.assert_allclose(camera[:,:3,:3]@camera[:,:3,:3].transpose(0,2,1),np.broadcast_to(np.eye(3),(len(t),3,3)),atol=1e-5)
 np.testing.assert_allclose(np.linalg.det(camera[:,:3,:3]),1,atol=1e-5)
 used=np.zeros(len(t),bool);roles=set()
 for e in manifest['episodes']:
  a,b=e['start_frame'],e['end_frame'];assert 0<=a<b<=len(t) and not used[a:b].any(),'episodes overlap or leave file bounds'
  assert e['role'] in ['fit','calibration','held_out'];assert e['object_id'] and e['action'];used[a:b]=True;roles.add(e['role'])
 assert {'fit','held_out'}<=roles,'need separate fit and held-out episodes'
 return dict(frames=len(t),points=x.shape[1],episodes=len(manifest['episodes']),visible_fraction=float(v.mean()),scope='Structural validation only: does not establish camera calibration, force accuracy, or physical prediction quality.')
def main():
 p=argparse.ArgumentParser();p.add_argument('npz',type=Path);p.add_argument('manifest',type=Path);a=p.parse_args()
 with np.load(a.npz) as z:result=validate(z,json.loads(a.manifest.read_text()))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
