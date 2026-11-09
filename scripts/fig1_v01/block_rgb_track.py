"""RGB-only block markers (same rgb_track pipeline as the stem): two fitting cameras per marker, triangulation."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; R = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import block as B, rgb_track
rgb_track.main(sys.argv[1], sys.argv[2], scene_root='runs/fig1-v01/block/scenes', truth_root='runs/fig1-v01/block/truth',
               out_root='runs/fig1-v01/block/rgb', markers=B.MARKERS, cells=lambda z: B.mesh(tuple(z['cells'])), normals=B.NORMALS)
