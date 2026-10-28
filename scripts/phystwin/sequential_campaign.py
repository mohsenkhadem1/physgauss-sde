"""Durable v06 data worker; checkpoints are local and never dropped on failure."""
from pathlib import Path
import argparse,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);a=p.parse_args()
 for split,mesh in [('bank',2),('bank',4),('development',4),('calibration',4),('test',4)]:
  subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/phystwin/sequential_reference.py'),'--split',split,'--mesh',str(mesh),'--worker',str(a.worker),'--workers','3'],cwd=ROOT,check=True)
 print('DATA_WORKER_COMPLETE',a.worker,flush=True)
if __name__=='__main__':main()
