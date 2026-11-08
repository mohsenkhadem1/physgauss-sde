"""Conditional native-solver reference, NOT independent OOD ground truth.
Replays fitted vs published Torus E/nu with identical learned appearance/geometry.
"""
from pathlib import Path
import json,hashlib,os,time
import numpy as np,torch,mmcv
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/gap-crossmethod-v02';REPORT=ROOT/'reports/gap-crossmethod-v02';os.chdir(BASE/'pacnerf-work')
from lib import PACNeRF,constraint,constraint_inv
import imageio.v2 as imageio
from plyfile import PlyData
P=REPORT/'pacnerf-counterfactual-protocol.json';p=json.loads(P.read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert sha(__file__)==p['source_sha256'];assert json.loads((REPORT/'pacnerf-prefix-status.json').read_text())['phase']=='complete'
checkpoint=BASE/'pacnerf-work/checkpoint/torus-prefix-native/model/train_dynamic.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False);data=torch.load(BASE/'pacnerf-work/data/torus/data.pt',map_location='cpu',weights_only=False)
root=BASE/'pacnerf-counterfactual';root.mkdir(exist_ok=True);results=[]
for condition,gravity in p['gravity'].items():
 for params in ['fitted','paper_parameters']:
  key=condition+'-'+params;out=root/key
  if (out/'complete.json').exists():results.append(json.loads((out/'complete.json').read_text()));continue
  torch.manual_seed(123);np.random.seed(123);torch.set_num_threads(4)
  cfg=mmcv.Config.fromfile(str(BASE/'vendor/pac-nerf/configs/torus.py'))['cfg'];cfg.update(saved['cfg']);cfg.update(base_dir=str(out),gravity=gravity,dt=p['dt'],taichi_cuda_memory=.45,nerf_bs=16384,write_out=True)
  model=PACNeRF(**cfg).cuda();model.load_state_dict(saved['model_state_dict'],strict=False)
  if params=='paper_parameters':
   with torch.no_grad():model.global_E.fill_(6.);model.global_nu.copy_(constraint_inv(torch.tensor([.3],device='cuda'),model.nu_bound))
  actual=dict(E=float(10**model.global_E.item()),nu=float(constraint(model.global_nu,model.nu_bound).item()),rho=float(10**model.global_rho.item()),v=model.global_v.detach().cpu().tolist())
  rgb,_,_=model.render_sequence(14,H=800,W=800,rays_o=data['rays_o'][0],rays_d=data['rays_d'][0],viewdirs=data['viewdirs'][0],savedir=str(out/'images'),dump_images=True)
  assert model.dynamic_observer.succeed() and np.isfinite(rgb).all()
  pointframes=[]
  for frame in range(14):
   v=PlyData.read(out/'simulation'/f'{frame}.ply')['vertex'];pointframes.append(np.stack([v[x] for x in ['x','y','z']],-1))
  xyz=np.stack(pointframes);assert np.isfinite(xyz).all();np.savez_compressed(out/'predictions.npz',rgb=rgb,xyz=xyz);imageio.mimwrite(out/'native.mp4',(np.clip(rgb,0,1)*255).astype(np.uint8),fps=24,quality=8)
  d=dict(key=key,complete=True,parameters=actual,gravity=gravity,dt=p['dt'],xyz_shape=list(xyz.shape),protocol_sha256=sha(P),checkpoint_sha256=sha(checkpoint),archive_sha256=sha(out/'predictions.npz'))
  (out/'complete.json').write_text(json.dumps(d,indent=2));results.append(d);del model;torch.cuda.empty_cache()
summary=[]
for condition in p['gravity']:
 a=np.load(root/(condition+'-fitted')/'predictions.npz');b=np.load(root/(condition+'-paper_parameters')/'predictions.npz');np.testing.assert_allclose(a['xyz'][0],b['xyz'][0],atol=0,rtol=0)
 rms=np.sqrt(np.mean(np.sum((a['xyz']-b['xyz'])**2,-1),-1));mse=np.mean((a['rgb']-b['rgb'])**2,axis=(1,2,3))
 summary.append(dict(condition=condition,framewise_particle_point_rms_m=rms.tolist(),framewise_render_mse=mse.tolist(),scope='Disagreement against a conditional native-solver reference sharing learned geometry, configured density, fitted initial velocity and native contact; not measured changed-force ground truth'))
(REPORT/'pacnerf-counterfactual-results.json').write_text(json.dumps(dict(complete=True,cases=results,comparison=summary,protocol_sha256=sha(P)),indent=2))
