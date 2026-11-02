"""Restore byte-exact large JSON reports stored compressed in Git."""
import gzip,hashlib,json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[2]
    mapping=root/'reports/compositional-neural-mechanics-v16/compressed-evidence.json'
    restored=0
    for row in json.loads(mapping.read_text())['files']:
        target=(root/row['original']).resolve();source=(root/row['compressed']).resolve()
        if not target.is_relative_to(root) or not source.is_relative_to(root):raise ValueError('Evidence path outside repository')
        data=gzip.decompress(source.read_bytes())
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Evidence digest mismatch')
        if target.exists():
            if target.read_bytes()!=data:raise ValueError(f'Preserve changed local evidence: {target}')
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);restored+=1
    print(f'Verified {len(json.loads(mapping.read_text())["files"])} compressed reports; restored {restored}.')

if __name__=='__main__':main()
