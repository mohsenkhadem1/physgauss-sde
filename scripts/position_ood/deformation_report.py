"""Population deformation audit, independent of all model predictions."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,OUT,protocol
from scripts.position_ood.seal import verify
from scripts.position_ood.score import deformation
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.sequential_reference import write

def residual(reference,moved):
 X=reference-reference.mean(0);Y=moved-moved.mean(0);U,_,V=np.linalg.svd(X.T@Y);D=np.eye(3);D[-1,-1]=np.linalg.det(U@V);return Y-X@(U@D@V)

def main():
 verify();rows=[]
 for split in ['test_id','test_material','test_environment']:
  for i in range(len(protocol()[split])):
   with np.load(BASE/f'fields/{split}-{i}.npz') as z:field=z['field']
   absolute=deformation(field);increment=[]
   for u in field:
    nr=residual(points()+field[0],points()+u);motion=u-field[0];inc=float(np.sqrt(np.mean(nr**2)));total=float(np.sqrt(np.mean(motion**2)));increment.append(dict(nonrigid_rms_mm=1000*inc,motion_rms_mm=1000*total,nonrigid_fraction=inc/max(total,1e-15)))
   rows.append(dict(split=split,object=i,absolute=absolute,incremental=increment))
 summary={}
 for key in ['absolute','incremental']:
  values=np.array([[np.mean([r[key][a][m] for a in range(4,9)]) for m in ['nonrigid_rms_mm','motion_rms_mm','nonrigid_fraction']] for r in rows]);summary[key]={metric:dict(median=float(np.median(values[:,j])),minimum=float(values[:,j].min()),maximum=float(values[:,j].max()),p10_p90=np.quantile(values[:,j],[.1,.9]).tolist()) for j,metric in enumerate(['nonrigid_rms_mm','motion_rms_mm','nonrigid_fraction'])}
 result=dict(objects=len(rows),actions=[4,5,6,7,8],summary=summary,rows=rows,scope='Each object contributes its mean across five new positional interaction patterns. Absolute nonrigid RMS removes the best rigid fit to canonical geometry. Incremental nonrigid RMS removes the best rigid fit from the already-deformed nominal state. Coordinate RMS in millimetres, at the same 81 query points as prediction metrics.')
 write(OUT/'population-deformation.json',result);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
