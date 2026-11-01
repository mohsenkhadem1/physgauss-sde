"""Development geometry-only observation pilot, not mechanical qualification."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
from scripts.physical_refinement_v15.observations import Camera,box_mesh,render_scene,compare_rgbd
ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'reports/physical-refinement-v15/observation-pilot';out.mkdir(parents=True,exist_ok=True)
    block=box_mesh([0,0,0],[.08,.06,.05],[.18,.55,.72])
    # Actual rear fixture faces coincide with block y=.03 at the support patch.
    fixtures=[box_mesh([x,.04,0],[.012,.02,.016],[.48,.48,.50])for x in [-.02,.02]]
    cameras={'front':Camera((0.,-.30,0.),(0.,0.,0.)),'rear':Camera((0.,.30,0.),(0.,0.,0.))}
    result={}
    for view,cam in cameras.items():
        images=[render_scene([block,fixture],cam)for fixture in fixtures]
        result[view]=compare_rgbd(*images)
        for i,image in enumerate(images):
            Image.fromarray(np.round(image['rgb']*255).astype('uint8')).save(out/f'{view}-support{i}.png')
            np.savez_compressed(out/f'{view}-support{i}.npz',rgb=image['rgb'],depth=image['depth'])
        # Hidden modulus is deliberately not an appearance input. Equal exterior states
        # are rendered identically regardless of the unseen mechanical hypothesis.
        result[view]['hidden_material_static']=compare_rgbd(images[0],render_scene([block,fixtures[0]],cam))
    record=dict(scope='Geometry-only development sensor pilot. Supports are physical rear fixtures, not painted labels; their mechanical patch binding still must be qualified with solver. No test actions/targets.',metrics=result,mesh=dict(block_size_m=[.08,.06,.05],fixture_size_m=[.012,.02,.016],fixture_centers_m=[[-.02,.04,0],[.02,.04,0]],mechanical_contact_face_y_m=.03),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),ROOT/'scripts/physical_refinement_v15/observations.py']})
    (out/'report.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
