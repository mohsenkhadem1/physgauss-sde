"""Read-only comparison of declared consistent-mass native development runs."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.campaign import write
REPORT=ROOT/'reports/coupled-surface-v13';BASE=ROOT/'runs/coupled-surface-v13/development/native-timestep'

def main():
    tags=['consistent-engineered-candidate4','consistent-engineered-failed625','consistent-engineered-stable625','consistent-engineered-stable3125']
    rows=[];checks={}
    for tag in tags:
        path=REPORT/f'native-timestep-{tag}.json'
        if not path.exists():continue
        data=json.loads(path.read_text());rows+=data['rows']
        checks[tag]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in data['source_sha256'].items()}
    comparisons=[]
    for scene in ('squeeze','curved'):
        for scale in (1e5,1e6):
            group=sorted([r for r in rows if r['spec']['scene']==scene and r['spec']['scale']==scale],key=lambda r:-r['spec']['dt'])
            for a,b in zip(group,group[1:]):
                with np.load(BASE/a['spec']['tag']/a['spec']['name']/'trajectory.npz') as aa,np.load(BASE/b['spec']['tag']/b['spec']['name']/'trajectory.npz') as bb:
                    n=min(len(aa['times']),len(bb['times']))
                    if not np.allclose(aa['times'][:n],bb['times'][:n],atol=1e-10,rtol=0):raise ValueError('Incompatible saved times')
                    delta=aa['frames'][:n]-bb['frames'][:n]
                    comparisons.append(dict(scene=scene,scale=scale,base_dt=a['spec']['dt'],fine_dt=b['spec']['dt'],base_success=a['success'],fine_success=b['success'],base_last_time=a['last_successful_time_s'],fine_last_time=b['last_successful_time_s'],shared_saved_prefix_end_s=float(aa['times'][n-1]),shared_saved_prefix_coordinate_rms_m=float(np.sqrt(np.mean(delta**2))),shared_saved_prefix_final_coordinate_rms_m=float(np.sqrt(np.mean(delta[-1]**2))),full_trajectory_comparison=bool(a['success'] and b['success'])))
    write(REPORT/'native-consistent-engineered-audit.json',dict(source_hash_checks=checks,comparisons=comparisons,scope='Development only; failed pairs report common saved prefixes, not completed forecasts. No physical-collapse claim follows from repeated element-guard failures.'))
    print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
