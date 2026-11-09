"""Native GIC train_dynamic on a stem-trial scene: camera 0 held out, all frames of the episode,
plus the labelled force adapter when STEM_FORCE_JSON is set.
STEM_GIC_MODE=appearance stops after the native Gaussian/deformation training stage."""
import os, sys, runpy
from pathlib import Path
R = Path(__file__).resolve().parents[2]
vendor = R/'runs/gap-crossmethod-v02/vendor/gic'
sys.path.insert(0, str(vendor)); sys.path.insert(0, str(Path(__file__).parent))
import gic_force_adapter
import taichi as _ti
_ti_init = _ti.init
def _init_capped(*a, **k):
    # GIC requests device_memory_fraction=0.5; the stem episodes keep 250-370 training images on the
    # same 11 GB GPU, so the Taichi pool is capped (memory only; numerics unchanged).
    if 'device_memory_fraction' in k:
        k['device_memory_fraction'] = min(k['device_memory_fraction'], float(os.environ.get('STEM_TI_MEM', '0.3')))
    return _ti_init(*a, **k)
_ti.init = _init_capped
print('force adapter spec:', gic_force_adapter.install(), flush=True)
from scene import dataset_readers
original = dataset_readers.readPACNeRFInfo
def partition(*args, **kwargs):
    info = original(*args, **kwargs)
    if not kwargs.get('read_cam', True):
        return info
    n = int(os.environ['STEM_FRAMES'])
    train = [c for c in info.train_cameras if c.uid != 0]
    test = [c for c in info.test_cameras if c.uid == 0]
    assert len(train) == 10*n and len(test) == n, (len(train), len(test))
    return info._replace(train_cameras=train, test_cameras=test)
dataset_readers.sceneLoadTypeCallbacks['PacNeRF'] = partition
if os.environ.get('STEM_GIC_MODE') == 'appearance':
    os.chdir(vendor)
    import train_dynamic as td
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
    from utils.general_utils import safe_state
    from utils.system_utils import check_gs_model
    parser = ArgumentParser(); model = ModelParams(parser); pipeline = PipelineParams(parser); op = OptimizationParams(parser)
    parser.add_argument("--config_file", default='config/torus.json', type=str)
    gs_args, phys_args = get_combined_args(parser); safe_state(gs_args.quiet)
    dataset = model.extract(gs_args)
    if not check_gs_model(dataset.model_path, gs_args.save_iterations):
        td.training(dataset, op.extract(gs_args), pipeline.extract(gs_args), gs_args.test_iterations + list(range(10000, 40001, 1000)), gs_args.save_iterations)
    print('appearance stage complete', flush=True)
else:
    runpy.run_path(str(vendor/'train_dynamic.py'), run_name='__main__')
