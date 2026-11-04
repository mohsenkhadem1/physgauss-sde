from pathlib import Path
import json,zipfile
from collections import Counter
import numpy as np


def schema(path):
    out={}
    with zipfile.ZipFile(path)as z:
        for name in z.namelist():
            with z.open(name)as f:
                version=np.lib.format.read_magic(f)
                shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
                out[name[:-4]]={'shape':list(shape),'dtype':str(dtype)}
    return out

rows=[]
for root in map(Path,['runs/isaac-boundary-orbit-v01','runs/gaussian-boundary-neural-v01']):
    episodes=sorted((root/'episodes').iterdir());checks=[];gravity=Counter();hz=Counter();convergence=Counter()
    for episode in episodes:
        s=json.loads((episode/'summary.json').read_text());v=json.loads((episode/'validation.json').read_text())
        checks.append(bool(v.get('passed')));gravity[str(s.get('gravity_m_s2'))]+=1;hz[str(s.get('physics_hz'))]+=1;convergence[str(s.get('numerical_convergence_established'))]+=1
    ep=episodes[0];c=np.load(ep/'observations/cameras.npz');ids=c['frame_ids'].tolist();times=c['times_s'].tolist()
    rows.append({'root':str(root),'episodes':len(episodes),'validation_passed':sum(checks),'gravity_summary':dict(gravity),'physics_hz':dict(hz),'convergence_established':dict(convergence),'example_episode':ep.name,'reference_schema':schema(ep/'reference.npz'),'camera_schema':schema(ep/'observations/cameras.npz'),'observed_frame_ids':ids,'observed_times_s':times,'private_trajectory_schema_headers_only':schema(ep/'truth/trajectory.npz')})
root=Path('runs/complex-scenes-v18/held-campaign-01/prefixes');complexrows=[]
for family in ('clamp_release','curved_push','container_drop','rigid_impact','deformable_pair','heterogeneous'):
    ep=root/f'{family}-world00';meta=json.loads((ep/'metadata.json').read_text());a=np.load(ep/'observations.npz')
    complexrows.append({'family':family,'example_prefix':str(ep),'schema':schema(ep/'observations.npz'),'times_s_first_last':[float(a['times_s'][0]),float(a['times_s'][-1])],'metadata':meta})
r={'scope':'Read-only schema inventory; no private trajectory array decoded, no raw RGB/depth decoded, no Isaac launched, no dataset modified','isaac272':rows,'v18_prefixes':complexrows}
Path('reports/gaussian-boundary-flow-v02/isaac-integration-inventory.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:[{q:v for q,v in x.items()if q in ('root','episodes','validation_passed','gravity_summary','convergence_established','family','schema','observed_frame_ids')}for x in r[k]]for k in('isaac272','v18_prefixes')},indent=2))
