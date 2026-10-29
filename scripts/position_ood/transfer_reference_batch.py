"""Resumable independent spatial-query references for calibration and testing."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.transfer import configure
from scripts.position_ood import study
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=3);a=p.parse_args();s=configure()
for split in ['transfer_calibration','transfer_test']:
    for i in range(a.worker,len(s[split]),a.workers):study.reference(split,i)
