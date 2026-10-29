"""Run acquisition calibration then test work on the same bounded worker."""
from pathlib import Path
import subprocess,sys
for split in ['transfer_calibration','transfer_test']:
 subprocess.run([sys.executable,str(Path(__file__).with_name('active_jobs.py')),'--split',split,*sys.argv[1:]],check=True)
