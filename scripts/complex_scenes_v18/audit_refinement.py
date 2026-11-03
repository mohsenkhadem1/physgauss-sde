"""Development sensitivity only; never promote an incomplete grid to truth."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.complex_scenes_v18.audit_development import ROOT


def compare(coarse,fine,families=None):
    roots=[ROOT/'runs/complex-scenes-v18'/x for x in (coarse,fine)]
    jobs=([dict(family=f,variant=v)for f in families for v in (0,1)]if families else json.loads((roots[0]/'supervisor.json').read_text())['planned']);rows=[]
    for job in jobs:
        row=dict(**job);paths=[list(r.glob(f"{job['family']}-v{job['variant']}-*/complete.json"))for r in roots]
        if any(len(p)!=1 for p in paths):row['state']='pending';rows.append(row);continue
        status=[json.loads(p[0].read_text())for p in paths]
        if any(not s['completed']for s in status):row['state']='incomplete';rows.append(row);continue
        for p,s in zip(paths,status):
            if hashlib.sha256((p[0].parent/'trajectory.npz').read_bytes()).hexdigest()!=s['trajectory_sha256']:raise ValueError('Refinement trajectory integrity failure')
        a,b=[np.load(p[0].parent/'trajectory.npz')for p in paths]
        assert np.allclose(a['times_s'],b['times_s'],atol=1e-10,rtol=0)
        assert np.array_equal(a['material_queries'],b['material_queries'])
        d=a['surface_positions_m']-b['surface_positions_m']
        rms=float(np.sqrt(np.mean(np.sum(d[1:]**2,axis=-1))))
        peak=float(np.sqrt(np.mean(np.sum(d**2,axis=-1),axis=(1,2))).max())
        row.update(state='complete',vector_rms_mm=1000*rms,peak_frame_vector_rms_mm=1000*peak,
                   development_target_pass=rms<=.00025 and peak<=.0005,
                   penetration_difference_mm=1000*abs(status[0]['maximum_penetration_m']-status[1]['maximum_penetration_m']),
                   trajectory_sha256=[s['trajectory_sha256']for s in status])
        rows.append(row)
    result=dict(coarse=coarse,fine=fine,scope='Development sensitivity; thresholds 0.25 mm mean and 0.5 mm peak; contact sampling and spatial checks separately required',cases=rows)
    out=ROOT/'reports/complex-scenes-v18'/f'{coarse}-versus-{fine}.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('coarse');p.add_argument('fine');p.add_argument('--families',nargs='+');a=p.parse_args();compare(a.coarse,a.fine,a.families)
