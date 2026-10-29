"""New spatial queries for a previously fitted native MPM model; no refitting."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,within_deadline
from scripts.position_ood.seal import verify
from scripts.position_ood.observations import make
from scripts.position_ood.physgaussian import PositionMPM
from scripts.position_ood.fit_mpm import parameters
from scripts.phystwin.sequential_reference import write

def run(i):
 seal=verify();within_deadline();tag=seal['settings']['mpm']['tag'];source=BASE/tag/f'test_id-{i}';fit=json.loads((source/'summary.json').read_text());destination=BASE/tag/f'transfer_test-{i}';destination.mkdir(parents=True,exist_ok=True)
 if (destination/'predictions.npz').exists():return
 obs=np.load(make('transfer_test',i));model=PositionMPM(fit['resolution'],fit['grid'],fit['dt'],rpic=fit['rpic'],attachment_width=fit['attachment_width'],ramp_seconds=fit['ramp_seconds']);model.parameters(**parameters(fit['best']['z'],fit['best']['layout']));pred=[];drift=[]
 for pose in obs['poses']:
  within_deadline();r=model.solve(pose,seconds=fit['seconds']);pred.append(r['field']);drift.append(r['drift_m'])
 np.savez_compressed(destination/'predictions.npz',predicted=pred,drift_m=drift,poses=obs['poses']);write(destination/'summary.json',dict(state='complete',source=str(source.relative_to(BASE)),scope='Same physical parameters fitted from the original three observations; new whole-object positional queries, no acquired target response or refitting.'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--object',type=int,required=True);a=p.parse_args();run(a.object)
