"""Fetch declared small SoftVTBench shards with atomic downloads and provenance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.external_benchmark.common import *
import urllib.request, concurrent.futures

def download(item):
    check_deadline()
    p = item['path']; dest = BASE / 'raw/SoftVTBench' / p
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://huggingface.co/datasets/Arthur12137/SoftVTBench/resolve/main/' + p
    if not dest.exists() or dest.stat().st_size != item['size']:
        partial = dest.with_suffix(dest.suffix + '.partial')
        with urllib.request.urlopen(url, timeout=60) as source, partial.open('wb') as out:
            while True:
                check_deadline(); chunk = source.read(1024 * 1024)
                if not chunk: break
                out.write(chunk)
        if partial.stat().st_size != item['size']: raise RuntimeError('Download size mismatch')
        partial.replace(dest)
    result = dict(url=url, path=str(dest.relative_to(ROOT)), size=dest.stat().st_size, sha256=sha256(dest))
    write_json(dest.with_suffix('.provenance.json'), result)
    print('Downloaded', p, flush=True)
    return result

if __name__ == '__main__':
    items = []
    # First three task IDs, declared before fitting; never select by prediction quality.
    for subset in ['object-soft', 'spatial-soft']:
        for task in range(3):
            family = 'object' if subset == 'object-soft' else 'spatial'
            directory = f'{subset}/libero_{family}/libero_{family}_task{task}/replayed_demos'
            url = 'https://huggingface.co/api/datasets/Arthur12137/SoftVTBench/tree/main/' + directory
            with urllib.request.urlopen(url, timeout=30) as r: records = json.load(r)
            items.extend(x for x in records if x['path'].endswith('.hdf5'))
    with concurrent.futures.ThreadPoolExecutor(3) as pool: results = list(pool.map(download, items))
    write_json(REPORT / 'softvtbench-downloads.json', results)
