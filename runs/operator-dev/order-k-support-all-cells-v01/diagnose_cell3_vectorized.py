from fractions import Fraction as Q
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from experiments.campaign_hypothesis_inputs import make_prior,realize_hypothesis
from experiments.campaign_physical_prior import FACE_BASES,FACE_NORMALS,PriorState
from experiments.campaign_support_mask_prior import cell_bounds,rational
from experiments.campaign_order_k_support_geometry import affine_scores,enumerate_order_k_cell
started=time.monotonic();out=Path('runs/operator-dev/order-k-support-all-cells-v01/cell3-representability.json')
public=json.loads(Path('runs/gaussian-campaign-v01/hypothesis-inputs-v01/public-H0.json').read_text());prior=make_prior(public);face=0;cell=3
nodes=np.flatnonzero(np.isclose(prior.reference_points[:,0],prior.minimum[0],atol=prior.surface_tolerance_m,rtol=0));origin=prior.center+FACE_NORMALS[face]*prior.size/2;B=FACE_BASES[face].T*prior.size[:,None];scores=affine_scores(prior.reference_points[nodes],origin,B);lower,upper=cell_bounds(prior,cell);result=enumerate_order_k_cell(scores,nodes.tolist(),16,lower,upper)
initial=json.loads(Path('runs/operator-dev/order-k-support-all-cells-v01/cell3-mismatch-diagnosis.json').read_text());rows=[];full_checks=0

def floats_inside(a,b):
 x=float(a)
 if Q.from_float(x)<=a:x=np.nextafter(x,np.inf)
 values=[]
 while Q.from_float(float(x))<b:
  values.append(float(x));x=np.nextafter(x,np.inf)
  if len(values)>10000:raise RuntimeError('Axis bound exceeded')
 return values

for row,r in zip(initial['rows'],result['regions']):
 if not row['mismatch']:continue
 if time.monotonic()-started>55:raise TimeoutError('55s narrow diagnostic bound')
 polygon=[tuple((u-a)/(b-a) for u,a,b in zip(point,lower,upper)) for point in r['polygon']]
 bbox=[(min(v[j] for v in polygon),max(v[j] for v in polygon)) for j in range(2)]
 lattice=[floats_inside(a,b) for a,b in bbox]
 count=len(lattice[0])*len(lattice[1])
 if count>200000:raise RuntimeError('Candidate bound exceeded')
 # Exact orientation via integer linear forms, avoiding repeated rationalgcd.
 lines=[]
 for a,b in zip(polygon,polygon[1:]+polygon[:1]):
  if a==b:continue
  A=a[1]-b[1];B=b[0]-a[0];C=(b[1]-a[1])*a[0]-(b[0]-a[0])*a[1]
  den=math.lcm(A.denominator,B.denominator,C.denominator);lines.append(tuple(int(v*den) for v in [A,B,C]))
 def exact_inside(x0,x1):
  n0,d0=float(x0).as_integer_ratio();n1,d1=float(x1).as_integer_ratio();D=max(d0,d1)
  X=n0*(D//d0);Y=n1*(D//d1)
  return all(A*X+B*Y+C*D>0 for A,B,C in lines)
 values=np.asarray([(a,b) for a in lattice[0] for b in lattice[1] if exact_inside(a,b)],dtype=float).reshape(-1,2)
 rounded_inside=exact_inside(*row['local_binary64'])
 matching=[]
 for offset in range(0,len(values),1024):
  local=values[offset:offset+1024]
  uv=-prior.support_uv_limit+(np.array([cell//prior.grid,cell%prior.grid])[None]+local)*(2*prior.support_uv_limit/prior.grid)
  world=origin+np.einsum('ni,ij->nj',uv,FACE_BASES[face])*prior.size
  distances=np.linalg.norm(prior.reference_points[nodes][None]-world[:,None],axis=-1)
  selected=np.sort(nodes[np.argsort(distances,axis=1,kind='stable')[:,:16]],axis=1)
  hits=np.flatnonzero(np.all(selected==np.array(r['node_ids'])[None],axis=1))
  matching.extend((offset+hits).tolist())
 witness_index=matching[0] if matching else (0 if len(values) else None);full=None
 if witness_index is not None and full_checks<10:
  x=(prior.lower+prior.upper)/2;x[4:6]=values[witness_index]
  state=PriorState(x[None],np.array([1]),np.array([0]),np.array([0]),np.array([[cell,0]]),np.array([[int(f[0]) for f in prior.incident_faces]]))
  actual=realize_hypothesis(public,state,prior)['fixture_nodes'];full_checks+=1;full={'local':values[witness_index].tolist(),'actual_fixture':actual,'matches_expected':tuple(actual)==r['node_ids']}
  if full['matches_expected']!=bool(matching):raise ValueError('Vectorized selector disagrees with fullrealizer witness')
 rows.append(dict(expected_fixture=list(r['node_ids']),area_fraction_of_cell=row['area_fraction_of_cell'],exact_gap_m2=row['exact_gap_float'],rounded_centroid_local_is_strict_exact_interior=rounded_inside,binary64_bbox_axis_counts=[len(v) for v in lattice],binary64_bbox_candidate_count=count,binary64_strict_exact_interior_points=len(values),vectorized_canonical_selector_matches=len(matching),full_realizer_witness=full,uv_polygon_exact=[[str(v) for v in u] for u in r['polygon']]))
 report=dict(completed=len(rows)==6,scope='Narrow offending-cell geometry only, no bulkretry or massdrop',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started,full_realizer_checks=full_checks,rows=rows,prior_mass_policy='No area removed, normalized or qualified; original301cell attempt remainsfailed',interrupted_slow_diagnostic='diagnose_cell3.py stoppedSIGINT/reaped130; itslog preserved')
 out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}));print(json.dumps([{k:r[k] for k in ['binary64_bbox_axis_counts','binary64_strict_exact_interior_points','vectorized_canonical_selector_matches','rounded_centroid_local_is_strict_exact_interior','full_realizer_witness']} for r in rows]))
