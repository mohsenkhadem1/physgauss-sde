"""Verify multipart research archives and restore regular files without clobbering."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from archive_research_artifacts import digest


def restore(manifest,parts,destination):
    manifest=Path(manifest);parts=Path(parts);destination=Path(destination).resolve()
    record=json.loads(manifest.read_text());archive=parts/record['archive']
    if not archive.exists():
        temporary=archive.with_suffix(archive.suffix+'.assembling')
        with temporary.open('xb') as output:
            for part in record['parts']:
                path=parts/part['name']
                if path.stat().st_size!=part['bytes'] or digest(path)!=part['sha256']:raise ValueError('Part checksum mismatch')
                with path.open('rb') as source:shutil.copyfileobj(source,output)
        temporary.replace(archive)
    if archive.stat().st_size!=record['archive_bytes'] or digest(archive)!=record['archive_sha256']:
        raise ValueError('Archive checksum mismatch')
    expected={row['path']:row for row in record['files']};seen=set()
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected:raise ValueError('Unlisted/nonregular archive member')
            relative=Path(member.name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe member path')
            target=destination/relative
            if not target.resolve().is_relative_to(destination):raise ValueError('Destination symlink escape')
            row=expected[member.name]
            if target.exists():
                if target.stat().st_size!=row['bytes'] or digest(target)!=row['sha256']:raise FileExistsError(f'Refusing to replace different file:{target}')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,target.open('xb') as output:shutil.copyfileobj(source,output)
                if target.stat().st_size!=row['bytes'] or digest(target)!=row['sha256']:raise ValueError('Restored file mismatch')
            seen.add(member.name)
    if seen!=set(expected):raise ValueError('Missing restored files')
    return {'files_verified':len(seen),'archive_sha256':record['archive_sha256'],'destination':str(destination)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True)
    p.add_argument('--parts',required=True);p.add_argument('--destination',required=True);a=p.parse_args()
    print(json.dumps(restore(a.manifest,a.parts,a.destination),indent=2))
