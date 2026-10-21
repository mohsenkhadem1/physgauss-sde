"""Freeze explicitly selected research evidence into audited portable chunks."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def archive(paths,output,name,part_bytes=40_000_000):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    files=sorted({p for root in map(Path,paths) for p in ([root] if root.is_file() else root.rglob('*')) if p.is_file()})
    if not files:raise ValueError('No explicitly selected files')
    for path in files:
        if path.is_absolute() or '..' in path.parts or path.is_symlink():raise ValueError('Only relative regular evidence files supported')
    if part_bytes<1_000_000 or part_bytes>49_000_000:raise ValueError('Use1–49MBGit-sized chunks')
    records=[{'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)} for path in files]
    archive_path=output/(name+'.tar.gz')
    # Provenance trees can contain hardlinked copies. Store each named file's
    # bytes so the portable archive retains the regular-file-only contract.
    # Symbolic links are still rejected above before any content is added.
    with tarfile.open(archive_path,'w:gz',compresslevel=1,dereference=True) as tar:
        for path in files:tar.add(path,arcname=str(path),recursive=False)
    expected={row['path']:row for row in records};seen=set()
    with tarfile.open(archive_path,'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected:raise ValueError('Unexpected archive member')
            h=hashlib.sha256();count=0
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block);count+=len(block)
            record=expected[member.name]
            if h.hexdigest()!=record['sha256'] or count!=record['bytes']:raise ValueError('Round-trip corruption or source changed during archiving')
            seen.add(member.name)
    if seen!=set(expected):raise ValueError('Missing archive members')
    # Recheck originals as well; a growing training directory is not frozen evidence.
    if any(digest(row['path'])!=row['sha256'] for row in records):raise ValueError('Source changed during snapshot')
    parts=[]
    with archive_path.open('rb') as stream:
        for index,block in enumerate(iter(lambda:stream.read(part_bytes),b'')):
            path=output/(archive_path.name+f'.part-{index:04d}');path.write_bytes(block)
            parts.append({'name':path.name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest(),
                          'git_blob_sha1':hashlib.sha1(f'blob {len(block)}\0'.encode()+block).hexdigest()})
    result={'archive':archive_path.name,'archive_bytes':archive_path.stat().st_size,
            'archive_sha256':digest(archive_path),'files':records,'parts':parts,
            'files_roundtrip_verified':len(records),'total_source_bytes':sum(row['bytes'] for row in records),
            'qualification':'Explicitly selected frozen evidence; other live experiments may continue',
            'restore':'Use restore_research_artifacts.py with this manifest and a directory containing all part files'}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ['files','parts']}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paths-json',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--part-bytes',type=int,default=40_000_000);a=p.parse_args()
    archive(json.loads(a.paths_json.read_text()),a.output,a.name,a.part_bytes)
