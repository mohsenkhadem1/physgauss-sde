from fractions import Fraction as Q
import json
from pathlib import Path
import numpy as np
from experiments.campaign_hypothesis_inputs import make_prior,realize_hypothesis
from experiments.campaign_physical_prior import FACE_BASES,FACE_NORMALS,PriorState
from experiments.campaign_support_mask_prior import cell_bounds,rational
from experiments.campaign_order_k_support_geometry import affine_scores,enumerate_order_k_cell
path=Path('runs/operator-dev/order-k-support-all-cells-v01/cell3-mismatch-diagnosis.json');report=json.loads(path.read_text())
public=json.loads(Path('runs/gaussian-campaign-v01/hypothesis-inputs-v01/public-H0.json').read_text());prior=make_prior(public);face=0;cell=3
nodes=np.flatnonzero(np.isclose(prior.reference_points[:,0],prior.minimum[0],atol=prior.surface_tolerance_m,rtol=0));origin=prior.center+FACE_NORMALS[face]*prior.size/2;B=FACE_BASES[face].T*prior.size[:,None];scores=affine_scores(prior.reference_points[nodes],origin,B);lower,upper=cell_bounds(prior,cell);result=enumerate_order_k_cell(scores,nodes.tolist(),16,lower,upper)

def inside(p,u):
 return all((b[0]-a[0])*(u[1]-a[1])-(b[1]-a[1])*(u[0]-a[0])>0 for a,b in zip(p,p[1:]+p[:1]) if a!=b)

def floats_strictly_between(a,b):
 x=float(a)
 if Q.from_float(x)<=a:x=np.nextafter(x,np.inf)
 values=[]
 while Q.from_float(float(x))<b:
  values.append(float(x));x=np.nextafter(x,np.inf)
  if len(values)>10000:raise RuntimeError('Narrow binary-lattice axis bound exceeded')
 return values

for row,r in zip(report['rows'],result['regions']):
 if not row['mismatch']:continue
 localp=[tuple((u-a)/(b-a) for u,a,b in zip(point,lower,upper)) for point in r['polygon']]
 bbox=[(min(v[j] for v in localp),max(v[j] for v in localp)) for j in range(2)]
 lattice=[floats_strictly_between(a,b) for a,b in bbox]
 count=len(lattice[0])*len(lattice[1])
 if count>200000:raise RuntimeError('Narrow binary-lattice product bound exceeded')
 rounded=tuple(map(Q.from_float,row['local_binary64']));row['rounded_centroid_local_is_strict_exact_interior']=inside(localp,rounded)
 row['local_polygon_bbox_exact']=[[str(x) for x in bb] for bb in bbox]
 row['binary64_bbox_axis_counts']=[len(v) for v in lattice];row['binary64_bbox_candidate_count']=count
 exact_inside=0;checks=0;matches=[]
 for x0 in lattice[0]:
  for x1 in lattice[1]:
   if not inside(localp,(Q.from_float(x0),Q.from_float(x1))):continue
   exact_inside+=1;x=(prior.lower+prior.upper)/2;x[4:6]=[x0,x1]
   state=PriorState(x[None],np.array([1]),np.array([0]),np.array([0]),np.array([[3,0]]),np.array([[int(f[0]) for f in prior.incident_faces]]))
   actual=realize_hypothesis(public,state,prior)['fixture_nodes'];checks+=1
   if tuple(actual)==r['node_ids'] and len(matches)<4:matches.append([x0,x1])
 row['binary64_strict_exact_interior_points']=exact_inside;row['canonical_realizer_checks_all_interior_points']=checks;row['matching_interior_witness_examples']=matches;row['matching_interior_witness_exists']=bool(matches)
report['mismatch_exact_area_sum']=str(sum((r['area'] for row,r in zip(report['rows'],result['regions']) if row['mismatch']),Q(0)))
report['mismatch_conditional_cell_mass_float']=sum(row['area_fraction_of_cell'] for row in report['rows'] if row['mismatch'])
report['mismatch_joint_BC1_prior_mass_float']=report['mismatch_conditional_cell_mass_float']/len(prior.feasible_faces[1])/len(prior.feasible_cells[1,0])
report['point_atlas_unchanged']=True
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'mismatch_conditional_cell_mass':report['mismatch_conditional_cell_mass_float'],'mismatch_joint_BC1_prior_mass':report['mismatch_joint_BC1_prior_mass_float'],'regions':[{k:row[k] for k in ['rounded_centroid_local_is_strict_exact_interior','binary64_bbox_axis_counts','binary64_strict_exact_interior_points','canonical_realizer_checks_all_interior_points','matching_interior_witness_exists']} for row in report['rows'] if row['mismatch']]}))
