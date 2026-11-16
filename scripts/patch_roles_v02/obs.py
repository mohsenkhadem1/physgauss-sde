"""RGB observations for step-2 truths with the unchanged Fig-1 pipeline: render the 11-camera scene
(stem_trial_v01/render.py) and track the 20 block markers by two-view LK + triangulation
(stem_trial_v01/rgb_track.py, called through fig1's arguments).  Outputs go to runs/patch-roles-v02/.
Usage: obs.py EPISODE TAG   (truth file runs/patch-roles-v02/truth/EPISODE-TAG.npz)"""
import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'scripts/fig1_v01')); sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import block as B, render, rgb_track

OUT = R/'runs/patch-roles-v02'


def main(episode, tag):
    z = np.load(OUT/f'truth/{episode}-{tag}.npz'); m = B.mesh(tuple(z['cells']))
    sc = OUT/f'scenes/{episode}-{tag}'
    if not (sc/'all_data.json').exists():
        render.write_scene(sc, z['X'][None]+z['U'], z['X'], m.surface_triangles(), B.EPISODE['fps'])
    rgb_track.main(episode, tag, scene_root='runs/patch-roles-v02/scenes', truth_root='runs/patch-roles-v02/truth',
                   out_root='runs/patch-roles-v02/rgb', markers=B.MARKERS, cells=lambda z: B.mesh(tuple(z['cells'])), normals=B.NORMALS)


if __name__ == '__main__':
    main(*sys.argv[1:3])
