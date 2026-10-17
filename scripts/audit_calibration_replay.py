"""Reproduce archived cube validation aggregates without training a network."""
import argparse
import json
from pathlib import Path
import torch
from physgauss.calibrated_diffusion import trajectory_features,CalibratedDiffusion,amplitude_metrics


def audit(calibration,output):
    record=json.loads(Path(calibration).read_text());settings=record['settings']
    features,metadata=trajectory_features(settings['validation_seeds'],settings['substeps'])
    network=CalibratedDiffusion()
    network.load_state_dict({k:torch.tensor(v,dtype=torch.float64) for k,v in record['state_dict'].items()})
    values=amplitude_metrics(network,features,settings['validation_innovation_seed'])
    comparison={k:{'archived':record['validation'][k],'replayed':v,
        'absolute_difference':abs(v-record['validation'][k])} for k,v in values.items() if isinstance(v,(int,float))}
    result={'matching_validation_feature_byte_hashes':sum(a['features_sha256']==b['features_sha256']
        for a,b in zip(metadata,record['validation_trajectories'])),
        'validation_trajectories':len(metadata),'metrics':comparison,
        'qualification':'post-hoc check of archived cube calibration aggregates on current host; not an exact feature replay or PAC calibration'}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',default='examples/calibrated-noise-setup/calibration.json');p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled=True;backend.strategy='greedy'
    audit(a.calibration,a.output)
