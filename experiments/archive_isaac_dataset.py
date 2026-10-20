"""Archive complete generated evidence into verified, portable Git-sized chunks."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def archive(paths,output):
    paths=list(map(Path,paths));output=Path(output);output.mkdir(parents=True,exist_ok=False)
    files=sorted({p for root in paths for p in ([root] if root.is_file() else root.rglob('*')) if p.is_file()})
    assert files
    metadata=[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]
    tarpath=output/'isaac-boundary-orbit-v01.tar.xz'
    with tarfile.open(tarpath,'w:xz',preset=3) as tar:
        for path in files:tar.add(path,arcname=str(path),recursive=False)
    # Verify every original byte against a streaming archive round trip.
    expected={r['path']:r for r in metadata};seen=set()
    with tarfile.open(tarpath,'r:xz') as tar:
        for member in tar:
            data=tar.extractfile(member).read();record=expected[member.name]
            assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
            seen.add(member.name)
    assert seen==set(expected)
    parts=[]
    with tarpath.open('rb') as stream:
        for i,block in enumerate(iter(lambda:stream.read(2_000_000),b'')):
            part=output/f'{tarpath.name}.part-{i:03d}';part.write_bytes(block)
            parts.append({'name':part.name,'bytes':len(block),'sha256':digest(part),
                          'git_blob_sha1':hashlib.sha1(f'blob {len(block)}\0'.encode()+block).hexdigest()})
    record={'archive':tarpath.name,'archive_bytes':tarpath.stat().st_size,'archive_sha256':digest(tarpath),
            'files':metadata,'files_roundtrip_verified':len(files),'total_source_bytes':sum(r['bytes'] for r in metadata),'parts':parts,
            'restore':'Concatenate part-000, part-001, ... in lexical order, verify SHA256, then tar -xJf the reconstructed archive'}
    (output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['files','parts']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('paths',nargs='+');parser.add_argument('--output',required=True)
    args=parser.parse_args();archive(args.paths,args.output)
