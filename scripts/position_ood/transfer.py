"""Move the whole object through space while preserving material coordinates.

This separate extension leaves the original ten-action cohort untouched. Both
controllers translate together, so environmental frame dependence can be tested
without confusing a coordinate change with a new constitutive material.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood import study
from scripts.position_ood.fixture import controls as local_controls
from scripts.position_ood.pilot import ACTIONS as LOCAL_ACTIONS
from scripts.phystwin.sequential_reference import write

TRANSFER_ACTIONS=[*LOCAL_ACTIONS[:3],('global_x',.04),('global_y',.04),('global_z',.04),('global_x',-.04),('global_y',-.04),('global_z',-.025)]

def controls(name,amplitude):
    if name.startswith('global_'):
        p=np.zeros((2,6));p[:,{'x':0,'y':1,'z':2}[name[-1]]]=amplitude;return p
    return local_controls(name,amplitude)


def configure():
    original=study.protocol();path=study.BASE/'transfer-protocol.json';source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not path.exists():write(path,dict(status='additional development-defined transfer challenge; freeze before held evaluation',actions=TRANSFER_ACTIONS,reference_levels=original['reference_levels'],transfer_development=[original['development'][i] for i in [0,1,2]],transfer_test=[original['test_id'][i] for i in range(9)],source=source))
    s=json.loads(path.read_text());assert s['source']==source,'Transfer code changed after reference generation'
    study.protocol=lambda:s;study.ACTIONS=TRANSFER_ACTIONS;study.controls=controls;return s

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='transfer_development');p.add_argument('--object',type=int,default=1);a=p.parse_args();configure();study.reference(a.split,a.object)
