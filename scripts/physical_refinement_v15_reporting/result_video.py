"""Post-score reporting only. No execution is permitted before root authorization.

Four independent settled states with a moving display camera, not a dynamical
forecast or interpolated trajectory. Original runtime and banks stay unchanged.
"""
import argparse,json,math,subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
from scripts.physical_refinement_v15.runtime_contract import verify
from scripts.physical_refinement_v15.banks import ROOT,sha
from scripts.physical_refinement_v15.reserved_scores import committed
from scripts.physical_refinement_v15.quadrature import hypotheses
from scripts.physical_refinement_v15.response_actions import response_actions
from scripts.physical_refinement_v15.render_bank import render_geometry,write
from scripts.physical_refinement_v15.observations import Camera
BASE=ROOT/'runs/physical-refinement-v15'
REPORT=ROOT/'reports/physical-refinement-v15'

def select_cases(rows,worlds):
 selected=[r for r in rows if r['track']=='VF' and r['repeat']==0 and r['policy']=='physical_risk' and r['budget']in (0,2)]
 pairs=[]
 for world in worlds:
  pair={b:[r for r in selected if r['world']==world and r['budget']==b]for b in (0,2)}
  if any(len(v)!=1 for v in pair.values()):raise ValueError('Exactly one before/after row per declared world required')
  before,after=pair[0][0],pair[2][0]
  x=before['highest_weight_metrics']['normalized_design_mse'];y=after['highest_weight_metrics']['normalized_design_mse']
  if not np.isfinite([x,y]).all()or min(x,y)<0:raise ValueError('Invalid reported error')
  pairs.append(dict(world=world,improvement=x-y,after_error=y,before=before,after=after))
 ordered=sorted(pairs,key=lambda x:(x['improvement'],x['world']))
 median=ordered[(len(ordered)-1)//2];worst=sorted(pairs,key=lambda x:(-x['after_error'],x['world']))[0]
 return [('median_improvement',median),('worst_after_error',worst)]

def read_state(folder,expected_pose,evidence):
 cp=folder/'complete.json';c=json.loads(cp.read_text())
 if c['pose_m_rad']!=expected_pose.tolist():raise ValueError('Display action mismatch')
 files=[cp]
 for name in ('sensor-geometry.npz','reaction-wrench.npz'):
  p=folder/name
  if sha(p)!=c['artifact_sha256'][name]:raise ValueError('Display state changed')
  files.append(p)
 evidence.update({str(p.relative_to(ROOT)):sha(p)for p in files})
 with np.load(folder/'reaction-wrench.npz',allow_pickle=False)as z:
  origin=np.array([.24,0,0])+expected_pose[1,:3]
  if not np.allclose(z['origins_world_m'][1],origin,atol=1e-12,rtol=0):raise ValueError('Wrench origin mismatch')
  wrench=np.r_[z['forces_N'][1],z['torques_Nm'][1]]
 if not np.isfinite(wrench).all():raise ValueError('Unavailable wrench; do not hide failure')
 return folder/'sensor-geometry.npz',wrench

def render_panel(job):
 geometry,camera=job
 return render_geometry(geometry,camera)[0]['rgb']

def run(args):
 if not args.execute_after_root_authorization:raise ValueError('Explicit post-score root authorization required')
 protocol,freeze=verify(args.protocol,args.protocol_sha256,args.freeze,args.freeze_sha256,artifacts=True)
 scorepath=REPORT/'reserved-scores.json';scores=json.loads(scorepath.read_text())
 if scores['protocol_sha256']!=args.protocol_sha256 or scores['method_freeze_sha256']!=args.freeze_sha256:raise ValueError('Score commitment mismatch')
 # Verify all forecast commitments and all scored evidence before any geometry reads.
 committed(BASE/'reserved/forecasts',protocol,args.protocol_sha256,args.freeze_sha256)
 for name,digest in scores['evidence_sha256'].items():
  if sha(ROOT/name)!=digest:raise ValueError('Scored evidence changed')
 chosen=select_cases(scores['rows'],[w['name']for w in protocol['reserved_worlds']])
 hypotheses_rows,_=hypotheses(protocol['gradient_count'],protocol['scale_count'])
 evidence={str(scorepath.relative_to(ROOT)):sha(scorepath)};outputs=[]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
 pool=ProcessPoolExecutor(max_workers=4)
 for role,pair in chosen:
  world=pair['world'];before=pair['before'];after=pair['after']
  nodes=[hypotheses_rows[r['highest_weight_index']]for r in (before,after)]
  target=REPORT/'videos'/f'{role}-{world}.mp4'
  if target.exists():raise ValueError('Preserve prior video')
  frames=BASE/'reporting-video-frames'/f'{role}-{world}';frames.mkdir(parents=True,exist_ok=True)
  recovery_path=REPORT/'videos/frame-recovery.json'
  recovery=json.loads(recovery_path.read_text())if recovery_path.exists()else {}
  if recovery:evidence[str(recovery_path.relative_to(ROOT))]=sha(recovery_path)
  frame_index=0
  for action_index,(action,pose)in enumerate(response_actions().items()):
   folders=[BASE/'reserved/worlds'/world/'n6'/f'pose-{action_index+5:03d}']+[BASE/'response-predictions/hypotheses'/node['base_name']/'n4'/f'pose-{action_index+1:03d}'for node in nodes]
   states=[read_state(folder,pose,evidence)for folder in folders]
   scales=[1.]+[node['scale']for node in nodes]
   for k in range(24):
    existing=frames/f'{frame_index:04d}.png'
    if existing.exists():
     if recovery.get(str(existing.relative_to(ROOT)))!=sha(existing):raise ValueError('Uncommitted recovery frame')
     frame_index+=1
     continue
    angle=.22*math.sin(2*math.pi*k/24)
    camera=Camera((.12+.45*math.sin(angle),-.55*math.cos(angle),.24),(.12,0,0),360,272,390.)
    canvas=Image.new('RGB',(1080,382),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,5),f'{world} | {role.replace("_"," ")} | held settled state {action_index+1}/4: {action}',fill='black',font=font)
    labels=['Actual simulated state','Initial highest-weight hypothesis','After 2 physical-risk queries (VF)']
    rendered=list(pool.map(render_panel,[(geometry,camera)for geometry,_ in states]))
    for panel,((geometry,wrench),scale,label)in enumerate(zip(states,scales,labels)):
     rgb=np.uint8(np.clip(rendered[panel],0,1)*255);canvas.paste(Image.fromarray(rgb),(panel*360,52))
     draw.text((panel*360+8,31),label,fill='black',font=font)
     w=wrench*scale
     draw.text((panel*360+8,325),'Right F [N]: '+', '.join(f'{v:+.3f}'for v in w[:3]),fill='black',font=small)
     draw.text((panel*360+8,344),'Right T [Nm]: '+', '.join(f'{v:+.4f}'for v in w[3:]),fill='black',font=small)
    draw.text((8,365),'Independent settled states; camera orbit only. No dynamic forecast/interpolation. Exploratory; numerical observation mismatch.',fill='#8b2222',font=small)
    canvas.save(frames/f'{frame_index:04d}.png');frame_index+=1
  target.parent.mkdir(parents=True,exist_ok=True)
  subprocess.run([FFMPEG,'-v','error','-framerate','12','-i',str(frames/'%04d.png'),'-c:v','libx264','-crf','23','-pix_fmt','yuv420p',str(target)],check=True)
  subprocess.run([FFMPEG,'-v','error','-i',str(target),'-f','null','-'],check=True)
  outputs.append(dict(role=role,world=world,path=str(target.relative_to(ROOT)),sha256=sha(target),frames=frame_index,fps=12,highest_weight_indices=[before['highest_weight_index'],after['highest_weight_index']],hypotheses=nodes,sequence=after['sequence'],ranking_improvement=pair['improvement'],ranking_after_error=pair['after_error']))
 pool.shutdown()
 for name,digest in evidence.items():
  if sha(ROOT/name)!=digest:raise ValueError('Video input changed')
 write(REPORT/'videos/provenance.json',dict(source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__))},input_sha256=evidence,protocol_sha256=args.protocol_sha256,method_freeze_sha256=args.freeze_sha256,outputs=outputs,selection='VF physical_risk repeat0, highest-weight normalized_design_mse; lower median improvement and greatest after-error across all12 worlds, ties lexical. Posthoc transparent visualization, not performance estimation.',scope='Four settled responses, common orbit camera, common scale affects wrench only. Highest-weight quadrature node, not posterior mean or continuous MAP. Visual review still required.'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-after-root-authorization',action='store_true');p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True);run(p.parse_args())
if __name__=='__main__':main()
