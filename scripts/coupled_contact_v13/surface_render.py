"""Reuse the fair world-coordinate renderer in an isolated surface-study namespace."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
BASE=ROOT/'runs/coupled-surface-v13';REPORT=ROOT/'reports/coupled-surface-v13'
FIXED_CASES=[['surface_test0','curved'],['surface_test3','moving_sphere'],['surface_test7','edge_drop']]


def implementation():
    name='scripts.coupled_contact_v13._surface_isolated_renderer'
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/coupled_contact_v13/render.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    from scripts.coupled_contact_v13.surface_scenarios import build,command
    module.BASE=BASE;module.REPORT=REPORT;module.build=build;module.command=command
    return module


def main():
    protocol=json.loads((REPORT/'protocol.json').read_text())
    if protocol['video_cases']!=FIXED_CASES:raise ValueError('Surface fixed-video declaration differs from renderer')
    if protocol['fps']!=100:raise ValueError('Four-times-slower label requires100Hz source at25fps')
    for obj,scene in FIXED_CASES:
        reference=BASE/'reference'/obj/scene/'trajectory.npz';prediction=BASE/'evaluation'/obj/scene/'predictions.npz'
        if not reference.exists() or not prediction.exists():raise RuntimeError('Fixed video source unavailable: '+obj+'/'+scene)
        with np.load(reference) as ref:times=ref['times'].copy()
        if len(times)!=61 or not np.allclose(times,np.arange(61)/100,atol=1e-10,rtol=0):raise ValueError('Unexpected saved source timeline')
        for family in ('fem','native','rigid'):
            path=BASE/'prediction'/obj/scene/family/'trajectory.npz'
            if path.exists():
                with np.load(path) as arr:
                    if arr['times'].shape!=times.shape or not np.allclose(arr['times'],times,atol=1e-10,rtol=0):raise ValueError('Prediction timeline differs: '+str(path))
    module=implementation();module.main()
    path=REPORT/'video-provenance.json';rows=json.loads(path.read_text())
    expected={f'{obj}-{scene}.mp4' for obj,scene in FIXED_CASES}
    if {Path(row['path']).name for row in rows}!=expected:raise RuntimeError('Fixed surface clips omitted')
    for row in rows:
        row['surface_renderer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        row['shared_renderer_sha256']=hashlib.sha256((ROOT/'scripts/coupled_contact_v13/render.py').read_bytes()).hexdigest()
        row['scene_source_sha256']=hashlib.sha256((ROOT/'scripts/coupled_contact_v13/surface_scenarios.py').read_bytes()).hexdigest()
        row['study']='Separately frozen surface quadrature contact, shared world scale and camera; quarter-speed playback'
    module.write(path,rows)


if __name__=='__main__':main()
