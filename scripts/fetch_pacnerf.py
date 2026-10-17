"""Select official PAC-NeRF ZIP members using verified HTTP byte ranges.

Avoids downloading the complete 8 GB multi-material release. ZIP CRCs are
verified by zipfile; an input manifest records SHA256 for each extracted file.
Only public UCLA Box links from the authors' repository are used.
"""
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
import urllib.request
import zipfile
from collections import OrderedDict

SHARED = '039jwttn9ibac73h3kit5vsuiepy7j4y'
ARCHIVES = {'data': ('1624530949617', 2945025303),
            'simulation_data': ('1624526625558', 5453120054),
            'GT_table2': ('1624515112576', 4098)}


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.cache = OrderedDict()
        # Follow the public Box redirect once. Never persist its temporary URL.
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-0'}), timeout=120) as r:
            self.url = r.geturl()

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0: raise ValueError('Negative archive offset')
        return self.position

    def read(self, size=-1):
        if size < 0: size = self.size-self.position
        size = min(size, self.size-self.position)
        if size <= 0: return b''
        start, end = self.position, self.position+size-1
        # ZIP reads a header, name, then payload separately. Aligned blocks
        # turn adjacent tiny requests into a single public range transfer.
        block_size = 4*1024*1024
        chunks = []
        for block in range(start//block_size, end//block_size+1):
            lo, hi = block*block_size, min(self.size, (block+1)*block_size)-1
            data = self._range(lo, hi)
            chunks.append(data[max(0, start-lo):min(hi-lo+1, end-lo+1)])
        self.position += size
        return b''.join(chunks)

    def _range(self, start, end):
        size = end-start+1
        key = (start, end)
        if key not in self.cache:
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                    raise ValueError('Server did not honor the exact archive range')
                data = r.read()
            if len(data) != size: raise ValueError('Truncated archive range')
            self.cache[key] = data
            if len(self.cache) > 16: self.cache.popitem(last=False)
        else: data = self.cache[key]
        return data


def fetch(output, archive, scenes, listing=False, pattern=None):
    file_id, size = ARCHIVES[archive]
    url = ('https://ucla.app.box.com/index.php?rm=box_download_shared_file'
           f'&shared_name={SHARED}&file_id=f_{file_id}')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(RangeFile(url, size)) as z:
        for item in z.infolist():
            parts = Path(item.filename).parts
            if not any('/'+scene.strip('/')+'/' in '/'+item.filename for scene in scenes) and archive != 'GT_table2':
                continue
            if item.is_dir(): continue
            if pattern and not re.search(pattern, item.filename): continue
            if listing:
                print(item.filename, item.file_size, item.compress_size, flush=True); continue
            destination = (output/item.filename).resolve()
            if not destination.is_relative_to(output): raise ValueError('Unsafe ZIP member')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.stat().st_size != item.file_size:
                data = z.read(item)  # validates member CRC
                destination.write_bytes(data)
            data = destination.read_bytes()
            if len(data) != item.file_size or zipfile.crc32(data) != item.CRC:
                raise ValueError('Cached member does not match official archive CRC')
            records.append({'path': item.filename, 'bytes': len(data), 'crc32': item.CRC,
                            'sha256': hashlib.sha256(data).hexdigest()})
            print('verified', item.filename, len(data), flush=True)
    if not listing:
        (output/f'{archive}-manifest.json').write_text(json.dumps({
            'source': 'https://github.com/xuan-li/PAC-NeRF',
            'public_shared_folder': f'https://ucla.app.box.com/s/{SHARED}',
            'archive': archive+'.zip', 'box_file_id': file_id, 'archive_size_bytes': size,
            'scenes': scenes, 'members': records}, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('data/pacnerf'))
    p.add_argument('--archive', choices=list(ARCHIVES), required=True)
    p.add_argument('--scenes', nargs='+', default=['torus', 'bird'])
    p.add_argument('--list', action='store_true')
    p.add_argument('--pattern', help='Optional member-name regular expression')
    a = p.parse_args(); fetch(a.output, a.archive, a.scenes, a.list, a.pattern)
