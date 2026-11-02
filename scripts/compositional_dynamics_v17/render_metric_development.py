"""Development-only visual score check on preserved coarse dynamic states.
The n2 comparator is NOT a qualified reference and neither is a learned prediction.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.physical_refinement_v15.observations import Camera,box_mesh
from scripts.compositional_dynamics_v17.render_sequence import render_frame
from scripts.compositional_dynamics_v17.render_metrics import compare_renderings
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-dynamics-v17';RAW=ROOT/'runs/compositional-dynamics-v17/prototype'

def main():
    output=R/'render-metric-development.json'
    if output.exists():raise ValueError('Preserve prior development scores')
    paths=[RAW/f'drop-n{n}-dt0.00005000/trajectory.npz'for n in (1,2)]
    arrays=[]
    for p in paths:
        with np.load(p)as a:arrays.append({k:a[k].copy()for k in ['times_s','surface_frames']})
    if not np.array_equal(arrays[0]['times_s'],arrays[1]['times_s']):raise ValueError('Different saved times')
    selected=[0,65,75,125,250]
    cam=Camera((.23,-.35,.20),(0,0,-.01),256,192,275.)
    rigids=[box_mesh([0,0,-.051],[.32,.25,.002],[.75,.77,.79])]
    rows=[]
    for i in selected:
        a,b=[render_frame(x['surface_frames'][i],cam,rigids,time_s=float(x['times_s'][i]))for x in arrays]
        rows.append(compare_renderings(a,b))
    sources=[Path(__file__),ROOT/'scripts/compositional_dynamics_v17/render_metrics.py',ROOT/'scripts/compositional_dynamics_v17/render_sequence.py']
    result=dict(scope='Development coarse-n1 versus coarse-n2 actual dynamic states. Neither is qualified truth; no learned-model/framework comparison or rendering-parity claim.',indices=selected,rows=rows,input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources},ssim_reference='https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html#skimage.metrics.structural_similarity')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print([(x['time_s'],round(x['psnr_foreground_union_db'],2),round(x['ssim_rgb_foreground_union'],4),x['silhouette_iou'])for x in rows])

if __name__=='__main__':main()
