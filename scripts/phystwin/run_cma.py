"""Run the official first-stage CMA-ES search; suppress optional OpenGL previews only."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('runs/third_party/PhysTwin'))
p.add_argument('--case',default='single_lift_sloth')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--iterations',type=int,default=20)
a = p.parse_args()
root,out = a.root.resolve(),a.output.resolve()
out.mkdir(parents=True,exist_ok=True)
os.environ['PHYSTWIN_HEADLESS']='1'
os.environ['WANDB_MODE']='disabled'
sys.path.insert(0,str(root))
import numpy as np
import torch
from qqtt import OptimizerCMA
from qqtt.utils import cfg,logger
import qqtt.data.real_data as real_data
import qqtt.engine.cma_optimize_warp as cma_module
real_data.visualize_pc = lambda *args,**kwargs: None
cma_module.visualize_pc = lambda *args,**kwargs: None
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
cfg.load_from_yaml(str(root/'configs'/('cloth.yaml' if 'cloth' in a.case or 'package' in a.case else 'real.yaml')))
split=json.loads((root/'data/different_types'/a.case/'split.json').read_text())
logger.set_log_file(path=str(out),name='cma')
t0=time.time()
optimizer=OptimizerCMA(data_path=str(root/'data/different_types'/a.case/'final_data.pkl'),
    base_dir=str(out),train_frame=split['train'][1])
os.chdir(out)  # Keep CMA's default outcmaes logs with the ignored local run.
optimizer.optimize(max_iter=a.iterations)
(out/'summary.json').write_text(json.dumps(dict(case=a.case,iterations=a.iterations,
    elapsed_seconds=time.time()-t0,split=split,upstream_commit=subprocess.check_output(
    ['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
    scope='Official first-stage search from real.yaml defaults, author-processed observations; no author CMA parameters loaded. Optional OpenGL previews disabled.'),indent=2)+'\n')
print('CMA_COMPLETE',flush=True)
