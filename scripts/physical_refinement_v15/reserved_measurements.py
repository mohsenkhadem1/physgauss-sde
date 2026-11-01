"""Offline committed synthetic packets. Never imported by acquisition policies.

No clean features, latent state, or alternative outcomes reach the broker API.
Execution is gated by explicit frozen protocol/source commitments.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.physical_refinement_v15 import banks
from scripts.physical_refinement_v15.render_bank import render_geometry,CAMERAS,write
from scripts.physical_refinement_v15.query_design import menu
from scripts.physical_refinement_v15.study import development_actions
ROOT=Path(__file__).resolve().parents[2]
RESERVED=ROOT/'runs/physical-refinement-v15/reserved'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def guard(protocol_path,protocol_sha,freeze_path,freeze_sha):
 if sha(protocol_path)!=protocol_sha or sha(freeze_path)!=freeze_sha:raise ValueError('Frozen commitment changed')
 p=json.loads(Path(protocol_path).read_text());f=json.loads(Path(freeze_path).read_text())
 if f['protocol_sha256']!=protocol_sha:raise ValueError('Freeze/protocol mismatch')
 for name,digest in f['artifact_sha256'].items():
  if sha(ROOT/name)!=digest:raise ValueError('Frozen artifact changed: '+name)
 sources=f['source_sha256']
 own=str(Path(__file__).resolve().relative_to(ROOT))
 if own not in sources:raise ValueError('Packet generator is not frozen')
 for name,digest in sources.items():
  if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
 worlds=p['reserved_worlds']
 if [w['name']for w in worlds]!=[f'reserved-{i:02d}'for i in range(12)]:raise ValueError('Exact declared twelve-world order required')
 if any(w['initial_camera']not in ('left','oblique')for w in worlds):raise ValueError('Undeclared initial camera')
 if p['measurement_seed']!=931127 or p['noise_repeats']!=4:raise ValueError('Undeclared measurement repetition/seed')
 return p,f

def packet(mean,C,*,world_index,repeat,action,camera,seed=931127):
 """Identical first64 Gaussian draws for V/VF; query coupling across policies."""
 mean=np.asarray(mean);C=np.asarray(C)
 if mean.shape not in ((64,),(70,)) or C.shape!=(len(mean),len(mean)):raise ValueError('Invalid modality')
 if not np.isfinite(mean).all()or not np.isfinite(C).all():raise ValueError('Nonfinite sensor input')
 query_seed=int.from_bytes(hashlib.sha256((action+'@'+camera).encode()).digest()[:4],'little')
 rng=np.random.default_rng(np.random.SeedSequence([seed,world_index,repeat,query_seed]))
 visual=mean[:64]+np.linalg.cholesky(C[:64,:64])@rng.normal(size=64)
 if len(mean)==64:return visual
 if np.any(C[:64,64:])or np.any(C[64:,:64]):raise ValueError('Declared visual/wrench covariance must be block diagonal')
 return np.r_[visual,mean[64:]+np.linalg.cholesky(C[64:,64:])@rng.normal(size=6)]

def nominal_bank(evidence):
 actions=list(development_actions());covariances=[]
 from scripts.physical_refinement_v15.feature_likelihood import covariance
 from scripts.physical_refinement_v15.sensor import SensorSettings
 settings=SensorSettings()
 # Only five observation actions are admitted; no target-response arrays read.
 actions=actions[:5]
 for action in actions:
  row=[]
  for camera in banks.CAMERAS:
   data,hashes=banks.checked_npz(banks.BASE/'render-bank/parallel_lower-g2-r0'/action,camera+'.npz');evidence.update(hashes)
   with data:row.append(covariance(data['rgb'],data['depth'],settings.depth_sd_m,settings.rgb_sd))
  covariances.append(row)
 return dict(actions=actions,cameras=banks.CAMERAS,features=np.zeros((1,5,3,64)),wrenches=np.zeros((1,5,6)),covariances=np.asarray(covariances))

def committed(folder,filename,evidence):
 record_path=folder/'complete.json';path=folder/filename;record=json.loads(record_path.read_text())
 if record.get('state') not in (None,'complete'):raise ValueError('Incomplete mechanics input')
 if record['artifact_sha256'][filename]!=sha(path):raise ValueError('Mechanics input changed')
 evidence.update({str(p.relative_to(ROOT)):sha(p)for p in (record_path,path)})
 return path,record

def world_packets(world,world_index,nominal,output,*,repeat_count=4,seed=931127):
 evidence={};clean={};wrenches={};actions=development_actions()
 requests=[('initial','initial',world['initial_camera'])]+menu(world['initial_camera'])
 for name,action,camera in requests:
  index=list(actions).index(action)
  if index not in range(5):raise ValueError('Response action cannot be measured')
  folder=RESERVED/'worlds'/world['name']/'n6'/f'pose-{index:03d}'
  path,record=committed(folder,'sensor-geometry.npz',evidence)
  if record['pose_index']!=index or record['pose_m_rad']!=actions[action].tolist():raise ValueError('Observation pose mismatch')
  _,clean[name]=render_geometry(path,CAMERAS[camera])
  if action.startswith('probe_'):
   path,_=committed(folder,'reaction-wrench.npz',evidence)
   with np.load(path,allow_pickle=False)as data:
    forces=data['forces_N'];torques=data['torques_Nm'];origins=data['origins_world_m']
    expected_origin=np.array([.24,0.,0.])+actions[action][1,:3]
    if origins.shape!=(2,3) or not np.isfinite(origins).all() or not np.allclose(origins[1],expected_origin,rtol=0,atol=1e-12):raise ValueError('Right-tool wrench origin mismatch')
    if forces.shape!=(2,3)or torques.shape!=(2,3):raise ValueError('Right-tool wrench shape mismatch')
    wrenches[name]=np.r_[forces[1],torques[1]]
 manifests=[]
 for repeat in range(repeat_count):
  for track in ('V','VF'):
   folder=output/world['name']/f'repeat-{repeat}'/track;entries={}
   for name,action,camera in requests:
    wrench=track=='VF'and action.startswith('probe_')
    mean=np.r_[clean[name],wrenches[name]]if wrench else clean[name]
    q=banks.query(nominal,action,camera,include_wrench=wrench,name=name)
    value=packet(mean,q.variance,world_index=world_index,repeat=repeat,action=action,camera=camera,seed=seed)
    path=folder/f'{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('Preserve existing measurement commitments')
    np.savez_compressed(path,observation=value)
    entries[name]=dict(path=str(path.relative_to(output)),sha256=sha(path),dimension=len(value))
   manifest=folder/'manifest.json'
   write(manifest,dict(role='synthetic_observed_packets',world=world['name'],repeat=repeat,track=track,packets=entries))
   manifests.append(dict(path=str(manifest.relative_to(output)),sha256=sha(manifest)))
 return dict(world=world['name'],input_sha256=evidence,manifests=manifests)

def generate(protocol_path,protocol_sha,freeze_path,freeze_sha):
 p,f=guard(protocol_path,protocol_sha,freeze_path,freeze_sha);output=RESERVED/'measurements'
 if output.exists():raise ValueError('Measurement directory already exists; no implicit overwrite/resume')
 evidence={};nominal=nominal_bank(evidence);results=[]
 for i,world in enumerate(p['reserved_worlds']):
  guard(protocol_path,protocol_sha,freeze_path,freeze_sha)
  results.append(world_packets(world,i,nominal,output,repeat_count=p['noise_repeats'],seed=p['measurement_seed']))
 guard(protocol_path,protocol_sha,freeze_path,freeze_sha)
 for r in results:evidence.update(r['input_sha256'])
 if any(sha(ROOT/name)!=digest for name,digest in evidence.items()):raise ValueError('Observation input changed during generation')
 result=dict(role='synthetic_packet_generation_provenance',protocol_sha256=protocol_sha,method_freeze_sha256=freeze_sha,source_sha256=f['source_sha256'],input_sha256=evidence,worlds=results,seed=p['measurement_seed'],noise_repeats=p['noise_repeats'],visual_noise_coupling='Same world/repeat/action/camera seed for V and VF and every policy; wrench appended only to VF probes.',scope='Original factor1 RGBD rendering; known synthetic feature noise; numerical observation mismatch remains. No target responses or private mechanics fields read.')
 write(output/'complete.json',result)
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--freeze-sha256',required=True);a=parser.parse_args()
 guard(a.protocol,a.protocol_sha256,a.freeze,a.freeze_sha256)
 if a.execute:generate(a.protocol,a.protocol_sha256,a.freeze,a.freeze_sha256)
if __name__=='__main__':main()
