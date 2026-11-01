"""Separate four declared development worlds; never reserved test generation."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.physical_refinement_v15 import development_mechanics_queue as worker
from scripts.physical_refinement_v15.mechanics_adapter import specification
from scripts.physical_refinement_v15.mechanics_pilot import BASE,REPORT,write,sha
PLAN=REPORT/'development-truth-plan.json';OUT=BASE/'development/worlds-queue';MANIFEST=OUT/'manifest.json'

def prepare():
 plan=json.loads(PLAN.read_text());assert len(plan['scenes'])==4 and all(s['name']==f'development-{i}' for i,s in enumerate(plan['scenes']));actions=plan['actions'];poses=np.asarray(list(actions.values()));jobs=[];sources={str(PLAN.relative_to(ROOT)):sha(PLAN)}
 for scene in plan['scenes']:
  name=scene['name'];path=OUT/'specs'/f'{name}.json';write(path,dict(case=scene['case'],poses=poses.tolist(),action_names=list(actions)))
  for n in (4,6):
   sources.update(specification(scene['case'],poses,n,3)['source_sha256']);jobs.append(dict(id=f'{name}-n{n}',name=name,n=n,spec=str(path.relative_to(ROOT)),out=str((BASE/'development/worlds'/name/f'n{n}').relative_to(ROOT)),budget_seconds=600))
 for p in (Path(__file__),ROOT/'scripts/physical_refinement_v15/development_mechanics_queue.py',ROOT/'scripts/physical_refinement_v15/study.py'):sources[str(p.relative_to(ROOT))]=sha(p)
 m=dict(scope=plan['role'],truth_plan_sha256=sha(PLAN),action_names=list(actions),poses_m_rad=poses.tolist(),jobs=jobs,source_sha256=sources,spec_sha256={j['spec']:sha(ROOT/j['spec']) for j in jobs},gates=worker.GATES,workers=3,per_job_seconds=600,conditional_refinement='n8 only when development4→6 checks fail; no threshold change')
 if MANIFEST.exists():
  if json.loads(MANIFEST.read_text())!=m:raise ValueError('Changed development-world manifest')
 else:write(MANIFEST,m)
 return m

def compare(name,a,b,m):
 rows=[]
 for i,action in enumerate(m['action_names']):
  dirs=[BASE/'development/worlds'/name/f'n{n}'/f'pose-{i:03d}' for n in (a,b)]
  if not all((p/'complete.json').exists() for p in dirs):rows.append(dict(action=action,passed=False,reason='Missing pose completion'));continue
  for p in dirs:
   rec=json.loads((p/'complete.json').read_text())
   if any(sha(p/key)!=h for key,h in rec['artifact_sha256'].items()):raise ValueError('Changed mechanics pose')
  x,y=[np.load(p/'private-state.npz') for p in dirs];sx,sy=[np.load(p/'surface.npz')['vertices'] for p in dirs];wx,wy=[np.load(p/'reaction-wrench.npz') for p in dirs];d=x['field']-y['field'];s=sx-sy;f=np.linalg.norm(wx['forces_N']-wy['forces_N'],axis=1);t=np.linalg.norm(wx['torques_Nm']-wy['torques_Nm'],axis=1)
  metrics=dict(field_rms_m=float(np.sqrt(np.mean(d*d))),field_max_m=float(abs(d).max()),surface_rms_m=float(np.sqrt(np.mean(s*s))),surface_max_m=float(abs(s).max()),force_difference_N=f.tolist(),torque_difference_Nm=t.tolist());passed=all(metrics[k]<=m['gates'][k] for k in ('field_rms_m','field_max_m','surface_rms_m','surface_max_m')) and bool((f<=1e-4+.005*np.linalg.norm(wy['forces_N'],axis=1)).all() and (t<=1e-5+.005*np.linalg.norm(wy['torques_Nm'],axis=1)).all());rows.append(dict(action=action,passed=passed,**metrics))
 return dict(name=name,mesh_pair=[a,b],passed=all(r['passed'] for r in rows),rows=rows)

def main():
 m=prepare();worker.OUT=OUT;worker.MANIFEST=MANIFEST;worker.phase(m['jobs'],m);initial=[compare(f'development-{i}',4,6,m) for i in range(4)];write(REPORT/'development-worlds-initial-qualification.json',dict(manifest_sha256=sha(MANIFEST),comparisons=initial));extra=[]
 for c in initial:
  if not c['passed']:
   j=next(x for x in m['jobs'] if x['name']==c['name'] and x['n']==6).copy();j.update(id=f"{c['name']}-n8",n=8,out=str((BASE/'development/worlds'/c['name']/'n8').relative_to(ROOT)));extra.append(j)
 write(OUT/'conditional-n8-jobs.json',dict(jobs=extra,manifest_sha256=sha(MANIFEST)))
 if extra:worker.phase(extra,m)
 final=[compare(j['name'],6,8,m) for j in extra];states=[json.loads((OUT/'jobs'/f"{j['id']}.json").read_text()) for j in m['jobs']+extra]
 write(REPORT/'development-worlds-mechanics.json',dict(scope=m['scope'],manifest_sha256=sha(MANIFEST),initial_comparisons=initial,refined_comparisons=final,jobs=states,failures=[s['job']['id'] for s in states if s['state']!='complete'],all_spatial_gates_pass=all(c['passed'] or any(f['name']==c['name'] and f['passed'] for f in final) for c in initial)))
if __name__=='__main__':prepare() if '--prepare-only' in sys.argv else main()
