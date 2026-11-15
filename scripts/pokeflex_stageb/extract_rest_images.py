"""Stage B: extract COLOR camera images of FREE frames only (rest frame = start frame of every poke segment, plus
every frame of the development take) using the protocol's own extractor (scripts/pokeflex_v01/extract.py) with a
member filter. Held-out / excluded frames are never extracted (the filter asserts that no whitelisted frame is in
P.heldout_frames). The robot/forces members are skipped (no rewrite of robot_traj.json / forces.npy); existing
meshes/confidences are skipped by extract.py's size check. extract.py regenerates the take's sealed manifest.
usage: python -I extract_rest_images.py [--obj NAME ...] [--take TAKE ...]"""
import argparse, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'scripts/pokeflex_v01'))
import pokeflex as P
import extract as X

_orig = X.classify
CUR = dict(frames=set())


def classify(rel):
    c = _orig(rel)
    if c == 'img':
        parts = rel.split('/')  # rig/i/channel/NNNNN.png
        return 'img' if parts[2] == 'color' and int(parts[3][:5]) in CUR['frames'] else None
    if c == 'robot':
        return None
    return c


X.classify = classify


def whitelist(obj, take):
    pk = P.segments(obj, take)['pokes']
    if P.is_dev(obj, take):
        return set(range(1, 151))
    fr = {q['start_frame'] for q in pk}
    assert not (fr & P.heldout_frames(obj, take)), (obj, take)
    return fr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--obj', nargs='*'); ap.add_argument('--take', nargs='*')
    a = ap.parse_args()
    objs = a.obj or json_objs()
    for obj in objs:
        for take in P.takes(obj):
            if a.take and take not in a.take:
                continue
            CUR['frames'] = whitelist(obj, take)
            zp = X.SRC/'poking'/obj/f'{take}.zip'
            print(obj, take, sorted(CUR['frames']) if len(CUR['frames']) < 20 else 'all (dev)', X.extract_take(zp, 'poking', obj, images=True), flush=True)


def json_objs():
    import json
    return json.loads((R/'scripts/pokeflex_v01/protocol.json').read_text())['objects_considered']


if __name__ == '__main__':
    main()
