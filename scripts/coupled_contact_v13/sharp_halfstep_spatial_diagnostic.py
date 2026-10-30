"""Re-score existing sharp12/16 trajectories at their common12.5us step."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.coupled_contact_v13.engineered_mesh_refinement import compare,dense_compare
from scripts.coupled_contact_v13.campaign import REPORT,write

def main():
 evidence={};rows=[];hashes={}
 for cells in [12,16]:
  matches=[]
  for p in REPORT.glob('consistent-reference-time-*.json'):
   d=json.loads(p.read_text())
   if 'checks' in d and d['plan']['reference_map']=={'edge_drop':cells} and d['plan']['dt']==2.5e-5:matches.append((p,d))
  if len(matches)!=1:raise RuntimeError(f'Require completed unique{cells} temporal report')
  p,d=matches[0];evidence[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
  for path,digest in d['plan']['source_sha256'].items():
   if path in hashes and hashes[path]!=digest:raise RuntimeError('Conflicting source hashes')
   hashes[path]=digest
  rows.extend(r for r in d['runs'] if r['spec'][4]==1.25e-5)
 checks=[]
 for E,mu in [(6000.,.35),(3000.,.1)]:
  a,b=[next(r for r in rows if r['spec'][1:4]==[E,mu,n]) for n in [12,16]]
  c=compare(a,b,True);dense=dense_compare(a,b);c.update(young=E,friction=mu,cells=[12,16],dense_surface=dense)
  c['passed']=c['complete'] and dense['complete'] and c['world_rms_m']<=.001 and c['shape_rms_m']<=.0005 and dense['world_rms_m']<=.001 and dense['shape_rms_m']<=.0005 and c['body_positions_finite'] and c['body_center_rms_m']<=.001 and all(r['success'] and r['min_j']>.02 and r['max_penetration_m']<=.001 for r in [a,b]);checks.append(c)
 hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
 write(REPORT/'sharp-halfstep-spatial-diagnostic.json',dict(scope='Diagnostic within existing pair; does not replace failed25us evidence or independently establish timestep convergence',source_sha256=hashes,evidence_sha256=evidence,dt=1.25e-5,runs=rows,checks=checks,source_unchanged=unchanged,spatial_gates_pass=unchanged and all(c['passed'] for c in checks),all_gates_pass=False))
 print(json.dumps(checks))
if __name__=='__main__':main()
