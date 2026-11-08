"""Independent state and force-budget audit of twelve native shared-shape runs."""
import json,hashlib,datetime
from pathlib import Path
import numpy as np
R=Path('reports/gap-crossmethod-v02');U=Path('runs/gap-crossmethod-v02/mphygs-shared-shapes')
def aligned(a,b):
 a=a-a.mean(0);b=b-b.mean(0);u,_,vt=np.linalg.svd(a.T@b);d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);return np.sqrt(np.mean(np.sum((a@u@d@vt-b)**2,axis=1)))
def main():
 status=json.loads((R/'mphygs-shared-shapes-status.json').read_text());assert len(status)==12;assert all(r['status']=='complete' for r in status)
 protocol=json.loads((R/'mphygs-shared-shapes-protocol.json').read_text())
 for p,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 rows=[];checks=0
 for shape in ['torus','ellipsoid']:
  for action in ['passive','calibration10x','held20z']:
   archives=[]
   for scale in [1,2]:
    p=U/f'{shape}-scale{scale}-{action}.npz';a=dict(np.load(p));m=json.loads(p.with_suffix('.json').read_text());assert a['x'].shape[0]==31;assert np.isfinite(a['x']).all();assert np.all(a['min_J']>0)
    expected=np.load(f'runs/gap-crossmethod-v02/masiv/contact-v02/{shape}-scale1-gaussian0x.npz')['x'][0];assert np.array_equal(a['x'][0],expected)
    assert np.allclose(np.array(m['per_selected_particle_force_N'])*m['selected_particles'],m['total_force_N']);assert np.isclose(m['mass_kg'],len(expected)*.018**3*1000*scale);checks+=6;archives.append(a)
   a,b=[v['x'].astype(float) for v in archives];err=np.linalg.norm(a-b,axis=-1)
   rows.append(dict(shape=shape,action=action,pair_trajectory_point_RMS_mm=float(np.sqrt(np.mean(err**2))*1000),pair_final_point_RMS_mm=float(np.sqrt(np.mean(err[-1]**2))*1000),pair_final_rigid_aligned_RMS_mm=float(aligned(a[-1],b[-1])*1000),worldA_deformation_after_rigid_removal_mm=float(aligned(a[0],a[-1])*1000),min_saved_J=min(float(z['min_J'].min()) for z in archives)))
 out=dict(passed=True,reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,completed_trajectories=12,rows=rows,scope='Native supplied-parameter mechanics; shared synthetic geometry with MASIV but native box versus Gaussian force support and different constitutive laws. Not a fitted pipeline or cross-method accuracy ranking.',protocol_sha256=hashlib.sha256((R/'mphygs-shared-shapes-protocol.json').read_bytes()).hexdigest())
 (R/'mphygs-shared-shapes-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
