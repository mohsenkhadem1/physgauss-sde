"""Joint observation of completed development mechanical states, never new solves."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
from scripts.position_ood.smooth_fixture import SmoothMaterialFixture
from scripts.position_ood.fixture import controls
from scripts.physical_refinement_v15.state_scene import scene_from_state
from scripts.physical_refinement_v15.observations import Camera,render_scene
ROOT=Path(__file__).resolve().parents[2]
def main():
    source=ROOT/'runs/physical-refinement-v15/mechanics-pilot/homogeneous/n4-q3';record=json.loads((source/'complete.json').read_text());case=record['spec']['case'];model=SmoothMaterialFixture(4,**case,gravity=(0,0,0),field_amplitude_J=0.,order=3,device='cpu');out=ROOT/'reports/physical-refinement-v15/state-observation-pilot-v2';out.mkdir(parents=True,exist_ok=True);rows=[]
    cam=Camera((.12,-.48,.20),(.12,0,0),320,240,420.)
    for i,(name,amplitude)in enumerate(record['spec']['actions']):
        path=source/f'action{i}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==record['artifact_sha256'][path.name]
        with np.load(path)as f:u=f['nodal_displacement'].copy()
        poses=controls(name,amplitude);image=render_scene(scene_from_state(model,u,case['layout'],poses),cam)
        Image.fromarray(np.round(image['rgb']*255).astype('uint8')).save(out/f'{name}.png');np.savez_compressed(out/f'{name}.npz',rgb=image['rgb'],depth=image['depth'])
        rows.append(dict(action=name,position_command=poses.tolist(),mechanical_state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),image=str((out/f'{name}.png').relative_to(ROOT)),depth_valid_pixels=int(np.isfinite(image['depth']).sum()),mechanical_metrics=record['rows'][i]))
    sources=[Path(__file__),ROOT/'scripts/physical_refinement_v15/state_scene.py',ROOT/'scripts/physical_refinement_v15/observations.py',ROOT/'scripts/physical_refinement_v15/fixture_geometry.py']
    (out/'report.json').write_text(json.dumps(dict(scope='Same saved developmentstate drives mechanicalmetrics and materialboundaryRGBD. No fittedimage-specific deformation or new mechanicalsolve; numericalqualification remains separate.',rows=rows,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources}),indent=2)+'\n');print('Rendered',len(rows),'saved developmentstates')
if __name__=='__main__':main()
