"""Fetch only official processed inputs and reference results; raw assets stay local."""
import argparse, hashlib, json, zlib
from pathlib import Path
from remotezip import RemoteZip

p = argparse.ArgumentParser()
p.add_argument('--case', default='single_lift_sloth')
p.add_argument('--root', type=Path, default=Path('runs/third_party/PhysTwin'))
a = p.parse_args()
records = []
for archive in ['data', 'experiments_optimization', 'experiments']:
    url = f'https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/{archive}.zip'
    with RemoteZip(url) as z:
        prefix = f'data/different_types/{a.case}/' if archive == 'data' else f'{archive}/{a.case}/'
        names = [n for n in z.namelist() if n.startswith(prefix) and (
            (archive == 'data' and n[len(prefix):] in ['final_data.pkl','gt_track_3d.pkl','split.json','calibrate.pkl','metadata.json']) or
            (archive == 'experiments_optimization' and n.endswith('/optimal_params.pkl')) or
            (archive == 'experiments' and (n.endswith('/inference.pkl') or '/train/best_' in n)))]
        if not names:
            raise RuntimeError(f'No example found in {archive}: {a.case}')
        for name in names:
            dest = a.root / ('reference' if archive == 'experiments' else '') / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                payload = z.read(name)  # Zip reader verifies CRC.
                dest.write_bytes(payload)
            info = z.getinfo(name)
            if dest.stat().st_size != info.file_size or zlib.crc32(dest.read_bytes()) != info.CRC:
                raise RuntimeError(f'Cached input differs from official archive: {dest}')
            records.append(dict(url=url, member=name, path=str(dest), bytes=dest.stat().st_size,
                                sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
            print(name, dest.stat().st_size, flush=True)
(a.root / f'{a.case}-download-manifest.json').write_text(json.dumps(records, indent=2)+'\n')
