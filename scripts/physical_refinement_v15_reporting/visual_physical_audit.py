"""Prospective secondary rendered/physical consistency audit; post-release only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,datetime,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scripts.physical_refinement_v15.banks import ROOT,sha
from scripts.physical_refinement_v15.render_bank import CAMERAS,render_geometry,write
from scripts.physical_refinement_v15.runtime_contract import verify
from scripts.physical_refinement_v15.reserved_scores import committed
from scripts.physical_refinement_v15.quadrature import hypotheses
BASE=ROOT/'runs/physical-refinement-v15';REPORT=ROOT/'reports/physical-refinement-v15'
PLAN=REPORT/'SECONDARY-VISUAL-PLAN.json'

def image_metrics(actual,predicted):
 a,b=actual['depth'],predicted['depth'];ma=np.isfinite(a);mb=np.isfinite(b);joint=ma&mb;union=ma|mb
 return dict(foreground_union_rgb_rmse=float(np.sqrt(np.mean((actual['rgb'][union]-predicted['rgb'][union])**2)))if union.any()else None,missing_foreground_pixels=int(np.count_nonzero(ma&~mb)),extra_foreground_pixels=int(np.count_nonzero(mb&~ma)),rgb_rmse=float(np.sqrt(np.mean((actual['rgb']-predicted['rgb'])**2))),joint_foreground_depth_rmse_m=float(np.sqrt(np.mean((a[joint]-b[joint])**2)))if joint.any()else None,silhouette_iou=float(joint.sum()/union.sum())if union.any()else 1.,joint_foreground_pixels=int(joint.sum()),union_foreground_pixels=int(union.sum()))

def prepare():
 if PLAN.exists():raise ValueError('Preserve prospective plan')
 write(PLAN,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),role='prospective_secondary_reporting_diagnostic',worlds=[f'reserved-{i:02d}'for i in range(12)],repeat=0,tracks=['V','VF'],policy='physical_risk',budgets=[0,2],states=['initial','held_response_0','held_response_1','held_response_2','held_response_3'],camera='Initial observed camera from protocol; all held states fixed oblique bank camera256x192/f250',metrics=['rgb_rmse','joint_foreground_depth_rmse_m','silhouette_iou'],physical='Exact same highest-weight quadrature node as highest_weight_metrics; common scale changes wrench, not rendered geometry',selection='Every declared case/track/budget; no filtering on qualification, success or error. Missing artifacts abort complete audit and remain explicit.',plots=['initial_image_fit_vs_held_force_and_torque_error','before_after_held_image_and_mechanical_error'],workers=4,limitations='Secondary reporting outside frozen primary scores. Known appearance/geometry, not learned reconstruction. RGB-D raster mismatch remains; highest-weight scene metrics are not posterior mean metrics.',source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__))}))

def render_job(job):
 folder,camera=job;folder=ROOT/folder;cp=folder/'complete.json';record=json.loads(cp.read_text());gp=folder/'sensor-geometry.npz'
 if sha(gp)!=record['artifact_sha256']['sensor-geometry.npz']:raise ValueError('Geometry artifact mismatch')
 key=sha(gp)+'-'+camera;out=BASE/'reporting-visual-audit-cache'/f'{key}.npz';out.parent.mkdir(parents=True,exist_ok=True)
 image,_=render_geometry(gp,CAMERAS[camera]);np.savez_compressed(out,rgb=image['rgb'],depth=image['depth'])
 return dict(job=job,path=str(out.relative_to(ROOT)),sha256=sha(out),input_sha256={str(p.relative_to(ROOT)):sha(p)for p in (cp,gp)})

def run(args):
 if not args.execute_after_root_release:raise ValueError('Root release required')
 plan=json.loads(PLAN.read_text())
 if sha(PLAN)!=args.plan_sha256:raise ValueError('Prospective plan changed')
 reporting_source_sha=sha(Path(__file__))
 p,f=verify(args.protocol,args.protocol_sha256,args.freeze,args.freeze_sha256,artifacts=True)
 scorepath=REPORT/'reserved-scores.json';score=json.loads(scorepath.read_text())
 if score['protocol_sha256']!=args.protocol_sha256 or score['method_freeze_sha256']!=args.freeze_sha256:raise ValueError('Score commitments differ')
 committed(BASE/'reserved/forecasts',p,args.protocol_sha256,args.freeze_sha256)
 for path,digest in score['evidence_sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Scored evidence changed')
 h,_=hypotheses(p['gradient_count'],p['scale_count']);jobs=set();rows=[]
 for world in p['reserved_worlds']:
  for track in ('V','VF'):
   for budget in (0,2):
    matches=[r for r in score['rows']if r['world']==world['name']and r['track']==track and r['repeat']==0 and r['policy']=='physical_risk'and r['budget']==budget]
    if len(matches)!=1:raise ValueError('Missing or duplicate fixed-cohort row')
    row=matches[0];hyp=h[row['highest_weight_index']];states=[]
    for state in range(5):
     camera=world['initial_camera']if state==0 else 'oblique'
     true=BASE/'reserved/worlds'/world['name']/'n6'/f'pose-{0 if state==0 else state+4:03d}'
     model=(BASE/'development/hypotheses'/hyp['base_name']/'n4'/'pose-000')if state==0 else BASE/'response-predictions/hypotheses'/hyp['base_name']/'n4'/f'pose-{state:03d}'
     pair=[(str(folder.relative_to(ROOT)),camera)for folder in (true,model)];jobs.update(pair);states.append(pair)
    rows.append(dict(world=world['name'],track=track,budget=budget,highest_weight_index=row['highest_weight_index'],hypothesis=hyp,highest_weight_metrics=row['highest_weight_metrics'],states=states))
 outputs={};evidence={str(PLAN.relative_to(ROOT)):sha(PLAN),str(scorepath.relative_to(ROOT)):sha(scorepath)}
 with ProcessPoolExecutor(max_workers=4)as pool:
  for result in pool.map(render_job,sorted(jobs)):
   outputs[tuple(result['job'])]=result;evidence.update(result['input_sha256']);evidence[result['path']]=result['sha256']
 for row in rows:
  metrics=[]
  for pair in row.pop('states'):
   images=[]
   for job in pair:
    with np.load(ROOT/outputs[job]['path'])as z:images.append({k:z[k].copy()for k in ('rgb','depth')})
   metrics.append(image_metrics(*images))
  row['initial_image_metrics']=metrics[0];row['held_image_metrics']=metrics[1:]
  row['held_image_mean']={k:float(np.mean([m[k]for m in metrics[1:]]))if all(m[k]is not None for m in metrics[1:])else None for k in metrics[0]}
 for path,digest in evidence.items():
  if sha(ROOT/path)!=digest:raise ValueError('Reporting evidence changed')
 result=dict(scope=plan['scope'],timing_disclosure=plan['timing_disclosure'],reporting_source_sha256=reporting_source_sha,plan_sha256=sha(PLAN),protocol_sha256=args.protocol_sha256,method_freeze_sha256=args.freeze_sha256,evidence_sha256=evidence,expected_rows=48,actual_rows=len(rows),unique_renders=len(jobs),rows=rows)
 output=REPORT/'visual-physical-audit.json'
 if output.exists():raise ValueError('Preserve completed audit')
 write(output,result);plot(result,output)

def plot(result,jsonpath):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
 for track,color in [('V','#2975ad'),('VF','#b86032')]:
  for budget,marker in [(0,'o'),(2,'x')]:
   rows=[r for r in result['rows']if r['track']==track and r['budget']==budget]
   x=[r['initial_image_metrics']['foreground_union_rgb_rmse']for r in rows]
   for ax,key in zip(axes[0,:2],('reaction_force_rmse_N','reaction_torque_rmse_Nm')):
    ax.scatter(x,[r['highest_weight_metrics'][key]for r in rows],color=color,marker=marker,label=f'{track}, budget{budget}')
  for world in sorted({r['world']for r in result['rows']}):
   pair=sorted([r for r in result['rows']if r['track']==track and r['world']==world],key=lambda r:r['budget'])
   values=[('held_image_mean','foreground_union_rgb_rmse',1),('held_image_mean','joint_foreground_depth_rmse_m',1000),('highest_weight_metrics','normalized_design_mse',1),('held_image_mean','silhouette_iou',1)]
   for ax,(group,key,scale)in zip([axes[0,2],*axes[1]],values):
    a,b=[r[group][key]for r in pair]
    if a is not None and b is not None:ax.scatter(a*scale,b*scale,color=color,s=22)
 for ax,title in zip(axes[0,:2],('Held force RMSE (N)','Held torque RMSE (Nm)')):ax.set_xlabel('Initial-state foreground-union RGB RMSE');ax.set_ylabel(title);ax.legend(fontsize=8)
 for ax,title in zip([axes[0,2],*axes[1]],('Held foreground-union RGB RMSE','Held joint-depth RMSE (mm)','Held normalized physical error','Held silhouette IoU')):
  lo,hi=ax.get_xlim();a,b=ax.get_ylim();lo=min(lo,a);hi=max(hi,b);ax.plot([lo,hi],[lo,hi],'k--',alpha=.4);ax.set_xlabel('Initial hypothesis');ax.set_ylabel('After2 physical-risk queries');ax.set_title(title)
 fig.suptitle('Secondary diagnostic: all12 worlds, repeat0, same highest-weight physical scenes\nKnown appearance; numerical observation mismatch remains. These are not posterior mean forecasts.')
 for ext in ('png','svg'):fig.savefig(REPORT/f'visual-physical-audit.{ext}',dpi=160)
 plt.close(fig)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute-after-root-release',action='store_true');parser.add_argument('--plan-sha256');parser.add_argument('--protocol',type=Path);parser.add_argument('--protocol-sha256');parser.add_argument('--freeze',type=Path);parser.add_argument('--freeze-sha256');args=parser.parse_args()
 if args.prepare:raise ValueError('Root-owned plan already declared; do not overwrite')
 else:run(args)
if __name__=='__main__':main()
