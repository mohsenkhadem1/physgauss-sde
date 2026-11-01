"""Read-only development final-geometry audit, separate from fitting/selection."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.contact_benchmark_v12.data import load,predictor_payload
from scripts.contact_benchmark_v12.feasibility import inherited_elements,jacobians

def main():
 data=predictor_payload(load({'case':'single_lift_sloth'}));elements=inherited_elements(data);base=ROOT/'runs/benchmark-improvement-v14/phystwin-attachment-development'
 sources=dict(attachment_fem=(base/'full-prediction.npz','predicted'),attachment_adapted=(base/'selected-prediction.npz','predicted'),original_adapted=(ROOT/'runs/contact-benchmark-v12/evaluation/development/single_lift_sloth/predictions.npz','object_frame'))
 out=dict(scope='Same inherited tetrahedra on final saved frames; orientation only, not continuous collision/equilibrium guarantee',element_count=len(elements),cutoff=.03+1e-8,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),methods={})
 for name,(path,key) in sources.items():
  with np.load(path) as f:pred=f[key]
  rows=[]
  for i,x in enumerate(pred):
   if not np.isfinite(x).all():rows.append(dict(frame=i,failed=True));continue
   J=jacobians(data['nodes'],x,elements);rows.append(dict(frame=i,failed=False,min_J=float(J.min()),inverted_elements=int((J<=0).sum()),below_cutoff_elements=int((J<=out['cutoff']).sum())))
  out['methods'][name]=dict(prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=rows,failed_frames=sum(r['failed'] for r in rows),inverted_frames=sum(r.get('inverted_elements',0)>0 for r in rows),below_cutoff_frames=sum(r.get('below_cutoff_elements',0)>0 for r in rows),min_J=min((r['min_J'] for r in rows if not r['failed']),default=None))
 (ROOT/'reports/benchmark-improvement-v14/phystwin-attachment-topology.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:{a:b for a,b in v.items() if a!='frames'} for k,v in out['methods'].items()}))
if __name__=='__main__':main()
