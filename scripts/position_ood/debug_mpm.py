from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.physgaussian import PositionMPM
from scripts.position_ood.fit_mpm import parameters
from scripts.position_ood.study import BASE
r=json.loads((BASE/'mpm/development-1/summary.json').read_text());obs=np.load(BASE/'observations/development-1.npz');m=PositionMPM(3,96,.00005,rpic=-1.,constitutive='neo_hookean');m.parameters(**parameters(r['best']['z'],r['best']['layout']));result=m.solve(obs['poses'][4],seconds=3.,block=10,safety_check=True);u=result.pop('field');ref=np.load(BASE/'fields/development-1.npz')['field'][4];result['rmse_mm']=float(1000*np.sqrt(np.mean((u-ref)**2)));print('DEBUG_MPM',result,flush=True)
