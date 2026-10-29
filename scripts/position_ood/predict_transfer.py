"""Transfer existing prefix fits to additional world-position queries."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,within_deadline
from scripts.position_ood.observations import make
from scripts.position_ood.runtime import dense_model
from scripts.phystwin.sequential_fixture import points
from scripts.phystwin.sequential_reference import write

def run(split,i,method='joint',mesh=2):
    within_deadline()
    if 'test' in split.split('_'):
        from scripts.position_ood.seal import verify
        verify()
    original={'transfer_development':'development','transfer_calibration':'calibration','transfer_test':'test_id'}[split];source=BASE/f'inverse/{original}-{i}/{method}-n{mesh}-p3.json';record=json.loads(source.read_text());obs=np.load(make(split,i));model=dense_model(mesh,field_frame='material' if method=='material_frame' else 'world')
    if method.startswith('extrapolated'):
        from scripts.position_ood.extrapolated import ExtrapolatedPositionFEM
        assert mesh==3
        model=ExtrapolatedPositionFEM(field_frame='material' if method=='extrapolated_material_frame' else 'world')
    best=record['best'];ids=np.argmin(np.linalg.norm(model.geo.rest[:,None]-points()[None],axis=-1),axis=0)
    with torch.no_grad():pred=np.array([model.predict(model.tensor(best['theta']),model.tensor(pose),best['layout'])[ids].cpu().numpy() for pose in obs['poses']])
    dest=BASE/f'inverse/{split}-{i}/{method}-n{mesh}-p3.json';record.update(split=split,source_fit=str(source.relative_to(BASE)),scope='Same acquired prefix and fitted physical parameters; new whole-object spatial queries.');write(dest,record)
    for row in record['rows']:write(dest.with_name(dest.stem+f'-layout{row["layout"]}.json'),row)
    np.savez_compressed(dest.with_suffix('.npz'),predicted=pred,poses=obs['poses'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='transfer_development');p.add_argument('--object',type=int,default=1);p.add_argument('--method',default='joint');p.add_argument('--mesh',type=int,default=2);a=p.parse_args();run(a.split,a.object,a.method,a.mesh)
