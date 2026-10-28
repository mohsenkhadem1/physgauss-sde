"""Run the three frozen inference stages with persistent subprocess logs."""
from pathlib import Path
import argparse,subprocess,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser();p.add_argument('--observations',choices=['direct','rgbd'],required=True);p.add_argument('--start',choices=['development','calibration','test'],default='development');a=p.parse_args();stages=['development','calibration','test']
 directory=ROOT/'runs/phystwin-sequential-support-v06'/('analysis' if a.observations=='direct' else 'analysis-rgbd')
 for stage in stages[stages.index(a.start):]:
  sealpath=directory/'seal.json'
  if sealpath.exists():
   from scripts.phystwin.sequential_policy import verify
   seal=json.loads(sealpath.read_text());verify(seal)
   assert json.loads((directory/'design.json').read_text())['fixed_sequence']==seal['fixed_sequence']
   if stage=='development' or (directory/f'{stage}-summary.json').exists():
    print('REUSE_SEALED',a.observations,stage,flush=True);continue
   assert stage=='test','Incomplete calibration after sealing requires an explicit recovery audit'
  subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/phystwin/sequential_policy.py'),'--stage',stage,'--observations',a.observations],check=True,cwd=ROOT)
 print('ANALYSIS_COMPLETE',a.observations,flush=True)
if __name__=='__main__':main()
