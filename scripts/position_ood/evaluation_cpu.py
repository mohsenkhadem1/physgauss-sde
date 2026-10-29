"""Primary CPU work followed by assistance on the same frozen inverse model."""
from pathlib import Path
import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);a=p.parse_args()
subprocess.run([sys.executable,str(Path(__file__).with_name('campaign_jobs.py')),'--kind','cpu','--worker',str(a.worker)],check=True)
subprocess.run([sys.executable,str(Path(__file__).with_name('active_assist.py')),'--worker',str(a.worker)],check=True)
