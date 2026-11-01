"""Development RGB-D prediction bank from committed geometry-only artifacts.
No private mechanics state, reaction labels or actual truth sensor outcomes read.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,datetime,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from scripts.physical_refinement_v15.observations import Camera,render_scene
from scripts.physical_refinement_v15.feature_likelihood import features
from scripts.physical_refinement_v15.study import base_cases,development_actions
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/physical-refinement-v15/development'
REPORT=ROOT/'reports/physical-refinement-v15'
CAMERAS={name:Camera(eye,(.12,0,0),256,192,250.)for name,eye in {'left':(-.30,0,0),'right':(.54,0,0),'oblique':(.12,-.48,.20)}.items()}
SOURCES=['scripts/physical_refinement_v15/render_bank.py','scripts/physical_refinement_v15/observations.py','scripts/physical_refinement_v15/feature_likelihood.py','scripts/physical_refinement_v15/study.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)
def sources():return {p:sha(ROOT/p)for p in SOURCES}
def check_sources(expected):
 if sources()!=expected:raise RuntimeError('Renderer source inventory changed')
def seconds(deadline):return (datetime.datetime.fromisoformat(deadline.replace('Z','+00:00'))-datetime.datetime.now(datetime.timezone.utc)).total_seconds()

def render_geometry(geometry_path,camera):
 """Reusable hypothesis/reference interface; no path-derived material branching."""
 with np.load(geometry_path,allow_pickle=False)as archive:
  names=archive.files;indices=sorted({int(k.split('_')[0][4:])for k in names if k.startswith('mesh')and k.endswith('_vertices')})
  expected={f'mesh{i}_{key}'for i in indices for key in ('vertices','triangles','color')}
  if not indices or set(names)!=expected:raise ValueError('Only complete geometry/appearance meshes permitted')
  meshes=[{key:archive[f'mesh{i}_{key}'].copy()for key in ('vertices','triangles','color')}for i in indices]
 for mesh in meshes:
  if mesh['vertices'].ndim!=2 or mesh['vertices'].shape[1]!=3 or not np.isfinite(mesh['vertices']).all():raise ValueError('Invalid geometric vertices')
  if not np.isfinite(mesh['color']).all():raise ValueError('Invalid appearance')
 image=render_scene(meshes,camera)
 return image,features(image['rgb'],image['depth'])

def render_job(job,expected_sources,deadline):
 check_sources(expected_sources)
 started=time.monotonic()
 if seconds(deadline)<=0:raise TimeoutError('Rendering deadline reached')
 case,action,index,pose=job;folder=BASE/'hypotheses'/case/'n4'/f'pose-{index:03d}';record_path=folder/'complete.json';geometry=folder/'sensor-geometry.npz';output=BASE/'render-bank'/case/action
 record=json.loads(record_path.read_text());input_sha={str(p.relative_to(ROOT)):sha(p)for p in (record_path,geometry)}
 if record['pose_index']!=index or record['pose_m_rad']!=pose or record['artifact_sha256']['sensor-geometry.npz']!=sha(geometry):raise ValueError('Geometry completion/pose mismatch')
 completion=output/'complete.json'
 if completion.exists():
  old=json.loads(completion.read_text())
  if old['input_sha256']!=input_sha or old['source_sha256']!=expected_sources or any(sha(output/name)!=digest for name,digest in old['artifact_sha256'].items()):raise ValueError('Changed render cache evidence')
  return dict(case=case,action=action,state='complete',cached=True)
 output.mkdir(parents=True,exist_ok=True)
 try:
  artifacts={};cameras={}
  for name,cam in CAMERAS.items():
   if seconds(deadline)<=0 or time.monotonic()-started>600:raise TimeoutError('Rendering stage/job deadline reached')
   image,mean=render_geometry(geometry,cam);target=output/f'{name}.npz';temp=output/f'{name}.pending.npz';np.savez_compressed(temp,rgb=image['rgb'],depth=image['depth'],features=mean);temp.replace(target);artifacts[target.name]=sha(target);cameras[name]=image['camera']
  check_sources(expected_sources)
  if any(sha(ROOT/p)!=digest for p,digest in input_sha.items()):raise RuntimeError('Geometry changed during rendering')
  write(completion,dict(state='complete',case=case,action=action,pose_index=index,source_sha256=expected_sources,input_sha256=input_sha,artifact_sha256=artifacts,cameras=cameras,feature_count=64,scope='Clean predicted renderedRGBDfeatures only; noactualtruthmeasurement/noise, nohiddenlabel channel'))
  return dict(case=case,action=action,state='complete',cached=False)
 except Exception as error:
  write(output/'status.json',dict(state='failed',case=case,action=action,error=repr(error),input_sha256=input_sha,source_sha256=expected_sources));raise

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--wait',action='store_true');parser.add_argument('--workers',type=int,default=4);parser.add_argument('--deadline-utc',required=True);args=parser.parse_args()
 if not 1<=args.workers<=4:raise ValueError('At most4singlethread workers')
 if seconds(args.deadline_utc)<=0:raise TimeoutError('Deadline passed')
 names=[c['name']for c in base_cases()];actions=development_actions();jobs=[(name,action,i,pose.tolist())for name in names for i,(action,pose)in enumerate(actions.items())];expected=sources()
 plan=dict(jobs=[dict(case=c,action=a,index=i,pose=p)for c,a,i,p in jobs],source_sha256=expected,deadline_utc=args.deadline_utc,workers=args.workers,cameras={k:dict(eye=list(v.eye),target=list(v.target),width=v.width,height=v.height,focal=v.focal)for k,v in CAMERAS.items()},nominal_covariance_case='parallel_lower-g2-r0',scope='Development20basecases×8publicactions×3fixedcameras; no truthsensoroutcomes orprivate-label reads')
 plan_path=REPORT/'development-render-plan.json'
 if plan_path.exists()and json.loads(plan_path.read_text())!=plan:raise ValueError('Existing render plan differs; preserve it and explicitly amend')
 write(plan_path,plan)
 if not args.execute:return
 pending=list(jobs);running={};results=[]
 with ProcessPoolExecutor(max_workers=args.workers)as pool:
  while pending or running:
   check_sources(expected)
   if seconds(args.deadline_utc)<=0:raise TimeoutError('Bank incomplete atdeadline')
   for job in list(pending):
    if len(running)>=args.workers:break
    path=BASE/'hypotheses'/job[0]/'n4'/f'pose-{job[2]:03d}'/'complete.json'
    if path.exists():running[pool.submit(render_job,job,expected,args.deadline_utc)]=job;pending.remove(job)
   if running:
    done,_=wait(running,timeout=min(10,max(.1,seconds(args.deadline_utc))),return_when=FIRST_COMPLETED)
    for future in done:
     job=running.pop(future)
     try:results.append(future.result())
     except Exception as error:results.append(dict(case=job[0],action=job[1],state='failed',error=repr(error)))
    write(BASE/'render-bank-status.json',dict(expected=len(jobs),completed=sum(x['state']=='complete'for x in results),failed=sum(x['state']=='failed'for x in results),pending=len(pending),running=len(running),results=results))
   elif pending:
    if not args.wait:break
    time.sleep(min(10,max(.1,seconds(args.deadline_utc))))
 write(BASE/'render-bank-status.json',dict(expected=len(jobs),completed=sum(x['state']=='complete'for x in results),failed=sum(x['state']=='failed'for x in results),pending=len(pending),running=0,results=results))
 print('render bank',len(results),'terminal /',len(jobs),'expected',flush=True)
if __name__=='__main__':main()
