"""Count every planned case and quantify deformation after rigid alignment."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


def shape_rms(rest,current):
    a=rest-rest.mean(0);b=current-current.mean(0)
    U,_,Vh=np.linalg.svd(a.T@b);D=np.eye(3);D[-1,-1]=np.linalg.det(U@Vh)
    return np.sqrt(np.mean(np.sum((a@U@D@Vh-b)**2,axis=-1)))


def audit(label):
    folder=ROOT/'runs/complex-scenes-v18'/label
    supervisor=json.loads((folder/'supervisor.json').read_text());rows=[]
    for job in supervisor['planned']:
        paths=list(folder.glob(f"{job['family']}-v{job['variant']}-*/complete.json"))
        if not paths:rows.append(dict(**job,state='pending'));continue
        if len(paths)!=1:raise ValueError('Ambiguous case')
        p=paths[0];s=json.loads(p.read_text());row=dict(**job,state=s['state'],elapsed_s=s['elapsed_s'])
        for k in ['minimum_J','maximum_penetration_m','maximum_energy_minus_work_drift_J','error']: 
            if k in s:row[k]=s[k]
        trajectory=p.parent/'trajectory.npz'
        if trajectory.exists():
            assert hashlib.sha256(trajectory.read_bytes()).hexdigest()==s['trajectory_sha256']
            a=np.load(trajectory);x=a['surface_positions_m'];rest=a['rest_surface']
            deformation=np.array([[shape_rms(r,y)for r,y in zip(rest,frame)]for frame in x])
            motion=np.sqrt(np.mean(np.sum((x-rest[None])**2,axis=-1),axis=-1))
            row.update(peak_shape_rms_mm=float(deformation.max()*1000),peak_motion_rms_mm=float(motion.max()*1000),peak_shape_rms_per_body_mm=(deformation.max(0)*1000).tolist(),trajectory_sha256=s['trajectory_sha256'])
        rows.append(row)
    result=dict(label=label,development_only=True,reference_qualified=False,planned=len(rows),completed=sum(r['state']=='complete'for r in rows),cases=rows)
    out=ROOT/'reports/complex-scenes-v18'/f'{label}-audit.json';out.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:print(r['family'],r['variant'],r['state'],'shape mm',round(r.get('peak_shape_rms_mm',float('nan')),3),'penetration mm',round(1000*r.get('maximum_penetration_m',float('nan')),3))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args();audit(a.label)
