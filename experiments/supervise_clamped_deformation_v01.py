"""Bounded owned-process supervisor for the frozen clamped diagnostic."""
import argparse,json
from experiments.run_clamped_deformation_v01 import ROOT,PROTOCOL,DESIGN
from experiments.supervise_surrogate_sbc_continuation import supervise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--digest',required=True);a=p.parse_args();out=(ROOT/PROTOCOL).parent
    command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',str(ROOT/'.venv/bin/python'),'-u','-m','experiments.run_clamped_deformation_v01','run','--digest',a.digest]
    receipt=supervise(command,out/'actual.log',out/'supervisor-receipt.json',wall_s=DESIGN['resources']['wall_s'],RSS_bytes=DESIGN['resources']['RSS_bytes'])
    print(json.dumps(receipt));raise SystemExit(0 if receipt.get('completed')else 1)
