"""Declared g9→g17 public-development prior refinement; no new truth data."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.physical_refinement_v15 import development_mechanics_queue as worker
from scripts.physical_refinement_v15.quadrature import geometry_cases
from scripts.physical_refinement_v15.study import base_cases,development_actions
from scripts.physical_refinement_v15.mechanics_adapter import specification
from scripts.physical_refinement_v15.mechanics_pilot import BASE,REPORT,write,sha
OUT=BASE/'development/gradient17-queue';MANIFEST=OUT/'manifest.json'
def prepare():
 original={c['name'] for c in geometry_cases(9)};cases=geometry_cases(17);new=[c for c in cases if c['name'] not in original];assert len(cases)==68 and len(new)==32
 poses=np.asarray(list(development_actions().values()));sources={};jobs=[];reused={}
 for name in sorted(original):
  p=BASE/'development/hypotheses'/name/'n4/complete.json';reused[str(p.relative_to(ROOT))]=sha(p)
 for c in new:
  name=c['name'];physical={k:v for k,v in c.items() if k!='name'};sources.update(specification(physical,poses,4,3)['source_sha256']);path=OUT/'specs'/f'{name}.json';write(path,dict(case=physical,poses=poses.tolist()));jobs.append(dict(id=f'{name}-n4',name=name,n=4,spec=str(path.relative_to(ROOT)),out=str((BASE/'development/hypotheses'/name/'n4').relative_to(ROOT)),budget_seconds=600))
 for p in (Path(__file__),ROOT/'scripts/physical_refinement_v15/quadrature.py',ROOT/'scripts/physical_refinement_v15/study.py',ROOT/'scripts/physical_refinement_v15/development_mechanics_queue.py'):sources[str(p.relative_to(ROOT))]=sha(p)
 sources.update(reused);m=dict(scope='Development prior gradient refinement5→9 only;36existingkinematiccases reused,32newinteriorgradients atn4/q3;no newtruths orreserveddata',gradient_count=17,jobs=jobs,cases=cases,action_names=list(development_actions()),source_sha256=sources,spec_sha256={j['spec']:sha(ROOT/j['spec']) for j in jobs},reused_completion_sha256=reused,per_job_seconds=600,workers=3,spatial_qualification='New interior gradients are inside previously qualified gradient/ratio range; no fresh n6 companion per new node, so range assumption is not per-node qualification')
 if MANIFEST.exists():
  if json.loads(MANIFEST.read_text())!=m:raise ValueError('Changed g17 development manifest')
 else:write(MANIFEST,m)
 return m

def main():
 m=prepare();worker.OUT=OUT;worker.MANIFEST=MANIFEST;worker.phase(m['jobs'],m);states=[json.loads((OUT/'jobs'/f"{j['id']}.json").read_text()) for j in m['jobs']];write(REPORT/'development-gradient17-mechanics.json',dict(scope=m['scope'],manifest_sha256=sha(MANIFEST),gradient_count=17,total_kinematic_cases=68,reused_cases=36,new_jobs=states,failed_jobs=[s['job']['id'] for s in states if s['state']!='complete'],all_new_jobs_complete=all(s['state']=='complete' for s in states),spatial_qualification=m['spatial_qualification']))
if __name__=='__main__':prepare() if '--prepare-only' in sys.argv else main()
