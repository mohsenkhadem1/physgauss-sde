"""Run released ReconPhys predictor unmodified on lexicographic released videos."""
from pathlib import Path
import sys,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/gap-first-v01/drop';sys.path.insert(0,str(BASE/'vendor/ReconPhys'))
import torch
from internvit_predictor import VideoPhysicsPredictor_internvit300m_temporalattn
from release_utils import load_video_as_tensor
ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();torch.set_num_threads(8);torch.manual_seed(123)
model=VideoPhysicsPredictor_internvit300m_temporalattn(internvit_model_name=str(BASE/'internvit')).to(a.device)
state=torch.load(BASE/'ckpt.pt',map_location='cpu',weights_only=False)['model'];model.load_state_dict(state,strict=True);model.eval();del state
out=ROOT/'reports/gap-first-v01/drop';out.mkdir(parents=True,exist_ok=True)
for asset in sorted((BASE/'assets').iterdir()):
 video=asset/'sample_00000/view0.mp4';v=load_video_as_tensor(str(video),num_frames=30,resize=(512,512))[None,:,:20].to(a.device);start=time.time()
 with torch.no_grad():result=model(v)
 d=dict(asset=asset.name,source='Official released video sample00000; no heldout split claim.',device=a.device,params={k:float(x.cpu()) for k,x in result.items()},seconds=time.time()-start,checkpoint_sha256=hashlib.sha256((BASE/'ckpt.pt').read_bytes()).hexdigest(),native_architecture=True,strict_weights=True,backbone_initialization='Same checkpoint backbone extracted to local pretrained file; full checkpoint then loaded strictly. No weight/architecture change.')
 (out/(asset.name+'-native-predictor.json')).write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
