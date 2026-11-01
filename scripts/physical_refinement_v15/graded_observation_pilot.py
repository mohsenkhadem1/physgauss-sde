"""Same-footprint static-view development demonstration; no mechanical outcomes."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.physical_refinement_v15.fixture_geometry import layout_fixtures
from scripts.physical_refinement_v15.observations import Camera,box_mesh,render_scene,compare_rgbd
ROOT=Path(__file__).resolve().parents[2]
def main():
 out=ROOT/'reports/physical-refinement-v15/graded-observation-pilot-v2';out.mkdir(parents=True,exist_ok=True);block=box_mesh([.12,0,0],[.24,.08,.08],[.18,.55,.72]);scenes={k:[block]+layout_fixtures(k)[0]for k in ['parallel_lower','crossed']};metrics={}
 for view,eye in [('left',(-.30,0,0)),('right',(.54,0,0))]:
  cam=Camera(eye,(.12,0,0),256,192,250.);images={name:render_scene(scene,cam)for name,scene in scenes.items()};metrics[view]=compare_rgbd(*images.values())
  for name,image in images.items():Image.fromarray(np.round(image['rgb']*255).astype('uint8')).save(out/f'{view}-{name}.png');np.savez_compressed(out/f'{view}-{name}.npz',rgb=image['rgb'],depth=image['depth'])
 result=dict(metrics=metrics,scope='Static development observation only; initialleftview hides rightfixture, additionalrightview resolves visible support-area distribution. Shared exact continuumfootprint; finite ligament renderer is declared homogenized support visualization, not resolvedfixturevalidation. Hidden material+/-gradient has same exterior/staticRGBD.',sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),ROOT/'scripts/physical_refinement_v15/fixture_geometry.py',ROOT/'scripts/physical_refinement_v15/observations.py',ROOT/'scripts/position_ood/smooth_fixture.py']})
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
