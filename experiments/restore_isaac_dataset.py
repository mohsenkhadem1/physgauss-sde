"""Download, verify and safely extract the permanently archived Isaac dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request


def restore(index,output):
    index=json.loads(Path(index).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=True)
    archive=output/index['archive'];digest=hashlib.sha256()
    with archive.open('wb') as target:
        for part in index['parts']:
            url=index['raw_base_url']+'/'+part['name']
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'physgauss-dataset-restore'}),timeout=120) as response:
                content=response.read()
            if len(content)!=part['bytes'] or hashlib.sha256(content).hexdigest()!=part['sha256']:
                raise ValueError('Downloaded part failed integrity: '+part['name'])
            digest.update(content);target.write(content);print(part['name'],flush=True)
    if archive.stat().st_size!=index['bytes'] or digest.hexdigest()!=index['sha256']:
        raise ValueError('Combined archive failed integrity')
    # The data filter rejects absolute paths, escaping links and traversal.
    with tarfile.open(archive,'r:xz') as stream:stream.extractall(output,filter='data')
    print('Verified and restored to',output.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',default='examples/isaac-boundary-orbit/raw-archive.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();restore(args.index,args.output)
