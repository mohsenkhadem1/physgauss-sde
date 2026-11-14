"""Selective, idempotent extraction of PokeFlex takes (licensed data: output goes only to git-ignored runs/).

Layout (OUT = runs/pokeflex-v01):
  OUT/takes/dropping/<Obj>/<take>/meshes/mesh-fNNNNN.obj (+.mtl)       passive footage (free)
  OUT/takes/poking/<Obj>/<take>/robot_traj.json                         robot input, forces REMOVED (free)
  OUT/takes/poking/<Obj>/<take>/cameras/<rig>/<i>/camera_parameters.json (free)
  OUT/sealed/poking/<Obj>/<take>/forces.npy                             (150,6) F/T wrench (gated by pokeflex.py)
  OUT/sealed/poking/<Obj>/<take>/meshes/, mesh_confidence/              (gated per frame by pokeflex.py)
  OUT/sealed/poking/<Obj>/<take>/images/<rig>/<i>/{color,depth}/N.png   only with --images (gated per frame)
  OUT/sealed/poking/<Obj>/<take>/MANIFEST.sha256.json                   sha256 of every sealed file
Atlas textures are skipped unless --atlas.  Archives are untrusted: run with python -I; member names are
whitelisted by regex, nothing is executed.  Incomplete downloads (*.part, unreadable zips) are skipped.

usage: python -I extract.py [--kind dropping|poking|all] [--obj NAME ...] [--images] [--atlas] [--src DIR] [--out DIR]
"""
import argparse, hashlib, json, re, shutil, sys, zipfile
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
SRC = R/'runs/third_party/PokeFlex-data'
OUT = R/'runs/pokeflex-v01'

RX = {  # relative member path (after the archive's top folder) -> class
    'mesh': re.compile(r'^meshes/mesh-f\d{5}\.obj$'),
    'mtl': re.compile(r'^meshes/MatLib-F\d{5}\.mtl$'),
    'atlas': re.compile(r'^meshes/Atlas-F\d{5}\.png$'),
    'conf': re.compile(r'^mesh_confidence/\d{5}\.npy$'),
    'robot': re.compile(r'^robot_data\.json$'),
    'cam': re.compile(r'^(volucam|realsense|kinect)/[01]/camera_parameters\.json$'),
    'img': re.compile(r'^(volucam|realsense|kinect)/[01]/(color|depth)/\d{5}\.png$'),
}


def classify(rel):
    for k, rx in RX.items():
        if rx.match(rel):
            return k
    return None


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_member(z, info, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix+'.tmp')
    with z.open(info) as s, open(tmp, 'wb') as d:
        shutil.copyfileobj(s, d, 1 << 20)
    tmp.replace(dst)


def extract_take(zpath, kind, obj, images=False, atlas=False, out=OUT):
    take = zpath.stem  # <Obj>_T<k> (dropping archives use a shortened top folder, e.g. Dice_T1/)
    free = out/'takes'/kind/obj/take; sealed = out/'sealed'/kind/obj/take
    want = {'mesh', 'mtl', 'conf', 'robot', 'cam'} | ({'img'} if images else set()) | ({'atlas'} if atlas else set())
    stamp = f'{zpath.stat().st_size}:{int(zpath.stat().st_mtime)}:{sorted(want)}'
    done = free/'.extracted'
    if done.exists() and done.read_text() == stamp:
        return 'up-to-date'
    try:
        z = zipfile.ZipFile(zpath)
    except (zipfile.BadZipFile, OSError) as e:
        return f'skip (unreadable: {e.__class__.__name__})'
    n = 0
    with z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split('/')
            if len(parts) < 2 or any(p in ('', '.', '..') for p in parts):
                continue
            rel = '/'.join(parts[1:]); c = classify(rel)
            if c not in want:
                continue
            if c == 'robot':
                rd = json.loads(z.read(info))
                assert isinstance(rd, list) and all(isinstance(r, dict) for r in rd)
                forces = np.array([[float(x) for x in r['forces']] for r in rd], dtype=np.float64)
                frames = [str(r['frame']) for r in rd]
                traj = [{k: v for k, v in r.items() if k != 'forces'} for r in rd]
                free.mkdir(parents=True, exist_ok=True); sealed.mkdir(parents=True, exist_ok=True)
                (free/'robot_traj.json').write_text(json.dumps(traj))
                np.save(sealed/'forces.npy', forces); (sealed/'forces_frames.json').write_text(json.dumps(frames))
                n += 1; continue
            if kind == 'dropping' or c == 'cam':
                dst = free/(('cameras/'+rel) if c == 'cam' else rel)
            elif c == 'img':
                dst = sealed/'images'/rel
            else:
                dst = sealed/rel
            if dst.exists() and dst.stat().st_size == info.file_size:
                continue
            write_member(z, info, dst); n += 1
    if kind == 'poking' and sealed.exists():
        man = {p.relative_to(sealed).as_posix(): sha(p) for p in sorted(sealed.rglob('*'))
               if p.is_file() and p.name != 'MANIFEST.sha256.json'}
        (sealed/'MANIFEST.sha256.json').write_text(json.dumps(man, indent=0))
    free.mkdir(parents=True, exist_ok=True); done.write_text(stamp)
    return f'extracted {n} files'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='all', choices=['dropping', 'poking', 'all'])
    ap.add_argument('--obj', nargs='*'); ap.add_argument('--take', nargs='*')
    ap.add_argument('--images', action='store_true'); ap.add_argument('--atlas', action='store_true')
    ap.add_argument('--src', default=str(SRC)); ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    kinds = ['dropping', 'poking'] if a.kind == 'all' else [a.kind]
    for kind in kinds:
        for zp in sorted(Path(a.src, kind).glob('*/*.zip')):  # *.zip.part never matches
            obj = zp.parent.name
            if a.obj and obj not in a.obj or a.take and zp.stem not in a.take:
                continue
            print(kind, obj, zp.stem, extract_take(zp, kind, obj, a.images, a.atlas, Path(a.out)), flush=True)


if __name__ == '__main__':
    sys.exit(main())
