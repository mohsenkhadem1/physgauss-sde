from pathlib import Path
import hashlib,json,os
import numpy as np
import torch
from experiments.campaign_point_active_views import score_censored_views
from experiments.censored_range_observation import CensoredRanges,CensoredRangeModel
assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
torch.set_num_threads(2)
source=Path('experiments/campaign_point_active_views.py');assert hashlib.sha256(source.read_bytes()).hexdigest()=='af03ee5556dbec64ad7253ffbd688a0e01757a153938395850dd1691d7e32890'
rng=np.random.default_rng(20261125);latent=rng.normal(size=(32,3));target=latent@rng.normal(size=(3,96))*.001
candidates=[CensoredRanges(rng.random((32,3))>.1,latent*(j+1)*.05) for j in range(4)]
model=CensoredRangeModel(np.eye(3)*.01+np.ones((3,3))*.001,.1,.5,.02)
cpu=score_censored_views(target,candidates,model,device='cpu',feature_chunk_size=17)
torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
gpu=score_censored_views(target,candidates,model,device='cuda:0',feature_chunk_size=17);torch.cuda.synchronize()
keys=['prior_target_variance_m2','expected_target_variance_m2','expected_variance_reduction_m2','target_variance_mc_se_m2','selected_advantage_mc_se_m2']
diffs={k:float(np.max(np.abs(np.asarray(cpu[k])-np.asarray(gpu[k])))) for k in keys}
passed=cpu['selected_index']==gpu['selected_index'] and all(v<=1e-18 for v in diffs.values())
r={'completed':passed,'synthetic_only':True,'cohort_or_prior_or_physics_access':False,'physical_GPU':1,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'pid':os.getpid(),'particles':32,'target_features':96,'views':4,'rays':3,'fantasies_each':32,'selected_CPU':cpu['selected_index'],'selected_GPU':gpu['selected_index'],'max_absolute_difference':diffs,'predeclared_abs_tolerance_m2':1e-18,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'torch_version':torch.__version__,'CUDA_version':torch.version.cuda,'device_name':torch.cuda.get_device_name(0),'utility_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path('runs/operator-dev/point-active-views-cuda-parity-v01/result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
assert passed
