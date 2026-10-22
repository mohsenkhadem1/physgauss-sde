"""Completed family negatives, read-only integrity validation, and source controls."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sha=lambda b:hashlib.sha256(b).hexdigest()
components=[('c38-static-prequalification-v01','completed-publication-inventory.json'),('c38-supported-grading-v01','publication-inventory.json'),('c38-static-graded-pilot-v01','preparation-inventory.json'),('c38-static-graded-pilot-v01','completed-publication-inventory.json'),('c38-static-motion-diagnostic-v02','publication-inventory.json'),('p2-supported-gravity-dop-source-v01','publication-inventory.json'),('p2-supported-gravity-parity-v01','publication-inventory.json'),('p2-supported-gravity-parity-v01','completed-publication-inventory.json'),('p2-warp-96-bridge-v03','preparation-publication-inventory.json'),('c38-static-graded-refinement-v01','incident-publication-inventory.json'),('c38-graded-capture-revalidation-v01','publication-inventory.json')]
paths=set();inventories=[]
for name,filename in components:
 p=ROOT/'runs/operator-dev'/name/filename;inventories.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes())))
 for row in json.loads(p.read_text())['files']:
  if sha((ROOT/row['path']).read_bytes())!=row['sha256']:raise ValueError('Changed frozen artifact '+row['path'])
  paths.add(row['path'])
 paths.add(str(p.relative_to(ROOT)))
paths.update(['docs/GAUSSIAN_C38_SUPPORTED_FAMILY_PROGRESS.md','docs/GAUSSIAN_C38_FORWARD_PROGRESS.md','runs/operator-dev/c38g-supported-family-v01/build.py'])
files=[]
for name in sorted(paths):
 p=ROOT/name;b=p.read_bytes();b.decode('utf-8')
 if p.is_symlink() or p.suffix not in ['.py','.md','.json','.txt','.log']:raise ValueError('Source/compact UTF8 text only')
 files.append(dict(path=name,bytes=len(b),sha256=sha(b),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
m=dict(schema='C38G-supported-family-v01',base_commit='91c5a795386db737fae8f0845ace59548c476be2',base_tree='075fa795d70019514e8f28b10c129c686117302a',component_inventories=inventories,files=files,file_count=len(files),total_bytes=sum(r['bytes'] for r in files),original_graded_case_JSON_recovered=False,all6_original_graded_captures_preserved=True,postincident_revalidation_new_provenance=True,raw_datasets_or_media_or_model_checkpoints=False,varied_or_dynamic_cohort_admitted=False,campaign_complete=False)
p=ROOT/'runs/campaign-c38g-paths.json';p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(dict(files=len(files),bytes=m['total_bytes'],manifest_sha256=sha(p.read_bytes()))))
