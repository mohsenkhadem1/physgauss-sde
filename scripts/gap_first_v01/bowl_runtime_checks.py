"""Reproducible equation/impulse/checkpoint checks, independent of test outcomes."""
import argparse,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
from .bowl_physics import run,build,command
from .bowl_torch import Batch
from .bowl_tests import test_geometry_and_contact,test_risk_and_action,test_rotating_bowl_wall_velocity

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',default='reports/gap-first-v01/bowl/runtime-checks.json');a=p.parse_args()
    test_geometry_and_contact();test_risk_and_action();test_rotating_bowl_wall_velocity()
    cpu=run(young=2500,friction=.35,penalty=38400,duration=.12,dt=5e-5)
    tensor=Batch([[2500,.35]],device=a.device).run(duration=.12,dt=5e-5,interaction='settle')
    result=dict(device=a.device,position_parity_max_m=float(np.max(abs(cpu['frames']-tensor['frames'][0]))),
                boxcar_force_parity_max_N=float(np.max(abs(cpu['reaction_averages']-tensor['reaction_averages'][0]))))
    momentum=(cpu['velocities']*cpu['mass'][None,:,None]).sum(1)
    impulse=np.diff(cpu['times'])[:,None]*(cpu['mass'].sum()*np.array([0,0,-9.81])-cpu['reaction_averages'][1:].sum(1))
    result['linear_momentum_impulse_error_Ns']=float(np.max(abs(np.diff(momentum,axis=0)-impulse)))
    with tempfile.TemporaryDirectory(prefix='bowl-restart-check-') as directory:
        checkpoint=Path(directory)/'state.npz';b=Batch([[2500,.35]],device=a.device);b.x[:,:,2]-=.025
        try:b.run(duration=.04,dt=5e-5,interaction='settle',checkpoint_path=checkpoint,deadline_unix=time.time()+.15)
        except RuntimeError as e:
            if 'deadline' not in str(e).lower():raise
        with np.load(checkpoint) as saved:result['checkpoint_step']=int(saved['next_step'])
        resumed=Batch([[2500,.35]],device=a.device).run(duration=.04,dt=5e-5,interaction='settle',checkpoint_path=checkpoint)
        b=Batch([[2500,.35]],device=a.device);b.x[:,:,2]-=.025
        full=b.run(duration=.04,dt=5e-5,interaction='settle')
        result['restart_position_max_m']=float(np.max(abs(resumed['frames']-full['frames'])))
        result['restart_boxcar_force_max_N']=float(np.max(abs(resumed['reaction_averages']-full['reaction_averages'])))
    assert result['position_parity_max_m']<1e-10
    assert result['boxcar_force_parity_max_N']<1e-9
    assert result['linear_momentum_impulse_error_Ns']<1e-10
    assert result['restart_position_max_m']<1e-10
    assert result['restart_boxcar_force_max_N']<1e-9
    result['status']='pass'
    source=Path(__file__).resolve();repository=source.parents[2]
    result['source_sha256']={str(p.relative_to(repository)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,source.with_name('bowl_physics.py'),source.with_name('bowl_torch.py')]}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
