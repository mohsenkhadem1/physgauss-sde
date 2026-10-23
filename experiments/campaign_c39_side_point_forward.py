"""Typed SIDE hypothetical forward library, composed with immutable C38 fields.

Physical grid labels are hypotheses, never observed-object truth. No files,
measurements, target labels or randomness. Static ideal q63 scope only.
"""
import numpy as np
from experiments.campaign_c38_table_validation import POINTS
ACTIONS={'gravity':(0.,0.,0.),'middle_down_2N':(0.,0.,-2.),'canonical_up_1N':(0.,0.,1.),'middle_side_2N':(0.,2.,0.)}
SIDE='middle_side_2N'
CATEGORIES=('both_ends','separated_bottom_patches')
YOUNG=np.array([75000.,np.sqrt(75000.*150000.),150000.])
MESHES=((32,16,16),(40,20,20))
ARCHIVE_KEYS={'source_index','case_ids','material_use','information_role','cells','support','young_Pa','action_id','force_vector_N','material_probes_m','probe_positions_m'}


def side_catalog():
 out=[]
 for mid,f in [('Elo',0.),('ElogQ1',.25),('Emid',.5),('ElogQ3',.75),('Ehi',1.)]:
  for b in CATEGORIES:out.append(dict(id='C39__'+mid+'__'+b+'__side2',young_Pa=float(np.sqrt(75000.*150000.))if f==.5 else float(75000.*2**f),support=b,material_use='TABLE_NODE'if f in(0.,.5,1.)else'WITHHELD_INTERPOLATION_VALIDATION',action_id=SIDE,force_vector_N=[0.,2.,0.]))
 return out


def ids_checked(ids):
 ids=np.asarray(ids)
 if ids.ndim!=1 or not len(ids)or ids.dtype.kind not in'iu'or len(np.unique(ids))!=len(ids)or np.any(ids<0)or np.any(ids>=63):raise ValueError('Exact unique q63 integer pointIDs required')
 return ids


class SidePointForward:
 def __init__(self,archive,*,cells=(40,20,20)):
  if set(archive)!=ARCHIVE_KEYS or tuple(cells)not in MESHES:raise ValueError('ExactSIDEcandidate archive/schema/mesh')
  a={k:np.asarray(v)for k,v in archive.items()}
  if a['information_role'].shape!=()or str(a['information_role'])!='C39_SIDE_CANDIDATE_TABLE_NODE_NOT_ADMITTED'or a['action_id'].shape!=()or str(a['action_id'])!=SIDE or not np.array_equal(a['force_vector_N'],np.array([0.,2.,0.])):raise ValueError('TypedSIDEcandidate role/worldvector required; noFz relabelling')
  if not np.array_equal(a['material_probes_m'],POINTS)or a['probe_positions_m'].shape!=(12,63,3)or not np.isfinite(a['probe_positions_m']).all():raise ValueError('Exact finiteSIDE12tablefields/q63 required')
  for k in ('source_index','case_ids','material_use','support','young_Pa'):
   if a[k].shape!=(12,):raise ValueError('SIDE12 source/hypothesis rows required')
  if a['cells'].shape!=(12,3)or a['source_index'].dtype.kind not in'iu':raise ValueError('ExactSIDEsourceindex/mesh')
  expected={2*i+j:(c,n)for i,c in enumerate(side_catalog())if c['material_use']=='TABLE_NODE'for j,n in enumerate(MESHES)}
  if len(set(a['source_index'].tolist()))!=12 or set(a['source_index'].tolist())!=set(expected):raise ValueError('OnlyallTABLE_NODErows; no withheldpromotion')
  joined={}
  for j,index in enumerate(a['source_index']):
   c,n=expected[int(index)]
   if str(a['case_ids'][j])!=c['id']or str(a['support'][j])!=c['support']or a['young_Pa'][j]!=c['young_Pa']or str(a['material_use'][j])!='TABLE_NODE'or tuple(a['cells'][j])!=n:raise ValueError('SIDEexactcase/source/material/BC/mesh identity')
   if n==tuple(cells):joined[(c['support'],c['young_Pa'])]=a['probe_positions_m'][j]-POINTS
  self.categories=CATEGORIES;self.young_nodes=YOUNG.copy();self.coefficients={};self.cells=tuple(cells)
  x=75000./YOUNG;V=np.stack([np.ones(3),x,x*x],axis=1)
  for b in CATEGORIES:
   q=np.stack([joined[(b,E)]for E in YOUNG]);coef=np.linalg.solve(V,q.reshape(3,-1)).reshape(3,63,3);coef.setflags(write=False);self.coefficients[b]=coef
  self.young_nodes.setflags(write=False)
 def observation_coefficients(self,category,action_id,point_ids):
  if category not in self.categories or action_id!=SIDE:raise ValueError('SIDEtyped action/support required')
  ids=ids_checked(point_ids);return self.coefficients[category][:,ids,:].reshape(3,-1).copy()


class LinearSidePointForward:
 def __init__(self,rows,*,cells=(40,20,20)):
  if len(rows)!=4 or tuple(cells)not in MESHES:raise ValueError('Exact4 fine linearSIDErows required')
  joined={}
  for r in rows:
   b=r['support'];n=tuple(r['cells']);key=(b,n)
   if b not in CATEGORIES or n not in MESHES or key in joined or r['young_Pa']!=75000. or r['action_id']!=SIDE or not np.array_equal(r['force_vector_N'],[0.,2.,0.])or not np.array_equal(r['material_probes_m'],POINTS):raise ValueError('FineLINEARtypedSIDE Eref75k/source/mesh/point identity')
   u=np.asarray(r['displacement63_m'])
   if u.shape!=(63,3)or not np.isfinite(u).all():raise ValueError('LINEARdisplacements notpositions required')
   joined[key]=u.copy()
  if set(joined)!={(b,n)for b in CATEGORIES for n in MESHES}:raise ValueError('BothBCs/bothmeshes mandatory')
  self.categories=CATEGORIES;self.young_nodes=YOUNG.copy();self.coefficients={};self.cells=tuple(cells)
  for b in CATEGORIES:
   c=np.zeros((3,63,3));c[1]=joined[(b,self.cells)];c.setflags(write=False);self.coefficients[b]=c
  self.young_nodes.setflags(write=False)
 def observation_coefficients(self,category,action_id,point_ids):
  if category not in self.categories or action_id!=SIDE:raise ValueError('LINEARtypedSIDE action/support required')
  ids=ids_checked(point_ids);return self.coefficients[category][:,ids,:].reshape(3,-1).copy()


class CompositePointForward:
 def __init__(self,initial_candidate,side_candidate):
  if tuple(initial_candidate.categories)!=CATEGORIES or tuple(side_candidate.categories)!=CATEGORIES or not np.array_equal(initial_candidate.young_nodes,YOUNG)or not np.array_equal(side_candidate.young_nodes,YOUNG):raise ValueError('Sameprior/BC hypotheses in both immutablelibraries required')
  self.initial_candidate=initial_candidate;self.side_candidate=side_candidate;self.categories=CATEGORIES;self.young_nodes=YOUNG.copy();self.actions=tuple(ACTIONS)
 def observation_coefficients(self,category,action_id,point_ids):
  ids=ids_checked(point_ids)
  if action_id not in ACTIONS:raise ValueError('Declaredtyped C39action only')
  if action_id==SIDE:return self.side_candidate.observation_coefficients(category,action_id,ids)
  return self.initial_candidate.observation_coefficients(category,{'gravity':0.,'middle_down_2N':-2.,'canonical_up_1N':1.}[action_id],ids)
 def predict_displacement(self,category,action_id,young_Pa,point_ids):
  if not np.isfinite(young_Pa)or not YOUNG[0]<=young_Pa<=YOUNG[-1]:raise ValueError('No material extrapolation')
  c=self.observation_coefficients(category,action_id,point_ids);x=75000./young_Pa;return((c[2]*x+c[1])*x+c[0]).reshape(-1,3)
