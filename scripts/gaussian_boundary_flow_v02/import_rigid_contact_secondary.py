"""Byte-preserving import after all secondary candidates terminate; no scoring.

Never moves source files, edits seals/plans, or changes primary stage records.
"""
import argparse,json,shutil
from pathlib import Path
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed,sha


def file_manifest(directory):
    result={}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink():raise ValueError('Symlink in immutable secondary source')
        if path.is_file():result[str(path.relative_to(directory))]=sha(path)
    return result


def copy_verified(source,destination):
    source,destination=Path(source),Path(destination)
    load_sealed(source);manifest=file_manifest(source)
    if destination.exists():
        if file_manifest(destination)!=manifest:raise ValueError('Existing import differs; do not overwrite')
        load_sealed(destination);return manifest
    temporary=destination.with_name(destination.name+'.importing')
    if temporary.exists():raise FileExistsError('Prior incomplete import preserved; no automatic overwrite')
    shutil.copytree(source,temporary)
    if file_manifest(source)!=manifest or file_manifest(temporary)!=manifest:
        raise ValueError('Source changed or copied bytes differ')
    load_sealed(temporary);temporary.rename(destination)
    return manifest


def import_all(source_root,cohort_root,addendum_path):
    source_root,cohort_root=Path(source_root),Path(cohort_root);addendum_path=Path(addendum_path)
    addendum=json.loads(addendum_path.read_text());summary=json.loads((source_root/'summary.json').read_text())
    if summary.get('all12_terminal')is not True or summary.get('secondary_addendum_sha256')!=sha(addendum_path):
        raise PermissionError('All twelve secondary cases must be terminal under this addendum')
    cases=addendum['cases'];rows=summary['cases']
    if len(cases)!=12 or len(rows)!=12 or {r['case'] for r in rows}!=set(cases):raise ValueError('Changed secondary cohort')
    results=[]
    for row in rows:
        case=row['case'];status=json.loads((source_root/(case+'-status.json')).read_text())
        if status!=row or row['state'] not in ('complete','failed','blocked','interrupted'):
            raise PermissionError('Secondary status changed or still live')
        if row['state']!='complete':results.append(dict(case=case,state=row['state'],imported=False));continue
        source=source_root/case;plan,_,_,_=load_sealed(source)
        if plan.get('secondary_addendum_sha256')!=sha(addendum_path) or plan.get('validation_protocol_sha256')!=addendum['primary_protocol_sha256']:
            raise ValueError('Secondary source provenance mismatch')
        if sha(source/'prediction.npz')!=row['prediction_sha256']:raise ValueError('Secondary status digest mismatch')
        parent=cohort_root/case
        if not parent.is_dir():raise ValueError('Primary cohort case directory missing')
        destination=parent/'rigid_contact';manifest=copy_verified(source,destination)
        record=dict(case=case,state='complete',imported=True,source=str(source.resolve()),destination=str(destination.resolve()),
                    secondary_addendum_sha256=sha(addendum_path),files_sha256=manifest,
                    policy='Immutable byte copy; original source retained, primary status/gates unchanged, no evaluator called')
        sidecar=parent/'rigid_contact-import.json'
        if sidecar.exists() and json.loads(sidecar.read_text())!=record:raise ValueError('Changed import provenance')
        if not sidecar.exists():sidecar.write_text(json.dumps(record,indent=2)+'\n')
        results.append(record)
    return dict(cases=results,secondary=True,target_access=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--cohort-root',required=True);p.add_argument('--addendum',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError('Import report already exists')
    report=import_all(a.source_root,a.cohort_root,a.addendum);out.write_text(json.dumps(report,indent=2)+'\n')
