"""Freeze acquisition calibration as soon as all its independent objects complete."""
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE,within_deadline
from scripts.position_ood.active_calibration import arrays,calibrate,evaluate,POLICIES
while True:
 within_deadline()
 try:
  for i in range(12):
   for policy in [None,*POLICIES]:arrays('transfer_calibration',i,policy)
  calibrate();break
 except FileNotFoundError:time.sleep(30)
while True:
 within_deadline();result=evaluate()
 if result['complete'] and result['matched_frame_ablation']['complete']:break
 time.sleep(60)
