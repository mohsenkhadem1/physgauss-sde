"""New visualization camera on unchanged saved native Gaussian trajectories."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.gap_first_v01.drop_run import geometry,render_utils,BASE,REPORT,OUT,sha
from scripts.gap_first_v01.drop_protocol import verify
import torch,numpy as np,utils3d
@torch.no_grad()
def main():
 p=verify();camera=json.loads((REPORT/'illustration-camera-v2.json').read_text());extr=utils3d.torch.extrinsics_look_at(torch.tensor(camera['origin_world'],device='cuda'),torch.tensor(camera['target_world'],device='cuda'),torch.tensor(camera['up_world'],device='cuda',dtype=torch.float32));fov=torch.tensor(np.deg2rad(camera['fov_degrees']),device='cuda',dtype=torch.float32);intr=utils3d.torch.intrinsics_from_fov_xy(fov,fov);records=[]
 for asset in p['assets']:
  g,_=geometry(asset);renderer=render_utils.get_renderer(g,resolution=camera['resolution'],bg_color=tuple(camera['background']),near=camera['near'],far=camera['far'])
  for case in ['passive-A','passive-B','changed-A','changed-B']:
   folder=BASE/'cases'/asset/case;status=json.loads((folder/'result.json').read_text());assert status['complete'];destination=folder/'illustration-camera-v2.npz';frames=[];sourcehashes={}
   if destination.exists():raise RuntimeError('Illustration already exists; do not overwrite')
   for frame in range(30):
    path=folder/f'gaussians-{frame:03d}.npy';sourcehashes[path.name]=sha(path);g._xyz=torch.from_numpy(np.load(path)).cuda();rgb=renderer.render(g,extr,intr,colors_overwrite=None)['color'].permute(1,2,0).clamp(0,1);frames.append((rgb.cpu().numpy()*255).astype(np.uint8))
   np.savez_compressed(destination,rgb=np.array(frames),extrinsics=extr.cpu().numpy(),intrinsics=intr.cpu().numpy());records.append(dict(asset=asset,case=case,output_sha256=sha(destination),source_position_sha256=sourcehashes));print(asset,case,flush=True)
 (OUT/'illustration-render-v2-manifest.json').write_text(json.dumps(dict(camera=camera,camera_sha256=sha(REPORT/'illustration-camera-v2.json'),source_sha256=sha(__file__),native_protocol_sha256=sha(REPORT/'protocol.json'),records=records),indent=2))
if __name__=='__main__':main()
