"""Existing-trajectory oracle discretization diagnostic, never parameter selection."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.coupled_contact_v13.engineered_scenarios import build
from scripts.coupled_contact_v13.mesh_diagnostic import interpolation,shape
from scripts.coupled_contact_v13.engineered_mesh_refinement import dense_compare
from scripts.coupled_contact_v13.campaign import REPORT,write
maps={'squeeze':8,'slide':8,'curved':8,'moving_sphere':12,'edge_drop':16}
paths=[REPORT/n for n in ['consistent-mass-diagnostic.json','consistent-higher-refinement.json','consistent-other-spatial.json','consistent-development-qualification.json']]
reports=[json.loads(p.read_text()) for p in paths]
lookup={}
for d in reports[:3]:
 for r in d['runs']:lookup[tuple(r['spec'])]=r
predictors={tuple(r['spec']):r for r in reports[3]['runs']}
def sample(row,query):
 with np.load(ROOT/row['path']/'trajectory.npz') as d:
  W=interpolation(d['rest'],d['tets'],query);x=np.einsum('qn,tnc->tqc',W,d['frames']);return x,d['times'].copy(),d['centers'].copy()
rows=[]
for n,refcells in maps.items():
 for E,mu in [(6000.,.35),(3000.,.1)]:
  ref=lookup[(n,E,mu,refcells)];query=build(n,E,mu,(2,2,2)).rest;y,t,cy=sample(ref,query)
  for cells in [4,8]:
   pred=predictors[(n,E,mu,1.25e-5,.0015,1)] if cells==4 else lookup[(n,E,mu,8)]
   x,tx,cx=sample(pred,query);assert np.allclose(t,tx,rtol=0,atol=1e-10)
   rows.append(dict(scenario=n,young=E,friction=mu,predictor_cells=cells,predictor_dt=.0000125 if cells==4 else .000025,reference_cells=refcells,reference_dt=.000025,reference_spatial_qualified=n!='edge_drop',same_grid=cells==refcells,world_rms_m=float(np.sqrt(np.mean((x[1:]-y[1:])**2))),shape_rms_m=float(np.sqrt(np.mean([shape(a,b)**2 for a,b in zip(x[1:],y[1:])]))),body_center_rms_m=float(np.max(np.sqrt(np.mean((cx[1:]-cy[1:])**2,axis=(0,2))))),dense_surface=dense_compare(pred,ref),predictor_path=pred['path'],reference_path=ref['path']))
write(REPORT/'consistent-oracle-bias.json',dict(scope='Development nominal-parameter resolution bias only; primary predictor4 uses12.5us. Predictor8 same-grid cases are labeled and are not discretization-transfer evidence. Edge16 is explicitly an UNQUALIFIED stress reference, not ground truth.',source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},input_report_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},rows=rows))
for r in rows:
 if r['predictor_cells']==4:print(r['scenario'],r['young'],r['world_rms_m']*1000,r['shape_rms_m']*1000,r['body_center_rms_m']*1000)
