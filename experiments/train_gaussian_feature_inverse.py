"""Compact mask-geometry or explicitly RGBD physical-posterior experiments.

This wrapper selects measured features and a smaller flow than the preserved
raw-CNN pilot. Prior mode trains the same posterior with zero observation context.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse
import json
from train_gaussian_inverse import train

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',nargs='+',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--observation-mode',choices=['rgb','rgbd'],default='rgb')
    parser.add_argument('--texture-calibration',help='New RGB geometry+texture feature version with frozen TRAIN color calibration')
    parser.add_argument('--context-mode',choices=['features','prior'],default='features')
    parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--selection-metric',choices=['nll','trajectory_rmse'],default='nll')
    parser.add_argument('--operator-samples',type=int,default=4);parser.add_argument('--category-weight',type=float,default=0.)
    parser.add_argument('--catalog');parser.add_argument('--resume');parser.add_argument('--operator-checkpoint');parser.add_argument('--operator-weight',type=float,default=0.)
    parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--eval-every',type=int,default=50)
    parser.add_argument('--learning-rate',type=float,default=1e-4);parser.add_argument('--hidden',type=int,default=64)
    parser.add_argument('--flow-layers',type=int,default=4);parser.add_argument('--seed',type=int,default=61003)
    parser.add_argument('--size',type=int,nargs=2,default=[192,160]);args=parser.parse_args()
    print(json.dumps(train(**vars(args),input_mode='features'),indent=2))
