"""Compact, complete per-fit evidence for publication without raw checkpoints."""
from pathlib import Path
import csv,json,hashlib,gzip
ROOT=Path(__file__).resolve().parents[2]
def csvwrite(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def main():
    for version in ['joint-field-identification-v01','joint-field-surface-v04']:
        report=ROOT/'reports'/version;manifest=json.loads((report/'continuous/manifest.json').read_text());docs=[];hashes={}
        # Export the frozen cohort only; preflight development smoke records also
        # live in this directory and are not members of the declared experiment.
        for j in manifest['allowed_jobs']:
            p=report/'continuous'/f"{j['shape']}-{j['population']}-case{j['case']:03d}-{j['stage']}-n{j['particles']}-s{j['seed']}.json"
            d=json.loads(p.read_text())
            assert d.get('complete'),p
            hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();docs.append(d)
        assert len(docs)==len(manifest['allowed_jobs'])==288
        rows=[];stages=[]
        for d in docs:
            j=d['job'];key=f"{j['shape']}-{j['population']}-case{j['case']:03d}-{j['stage']}-n{j['particles']}-s{j['seed']}"
            rows.append(dict(key=key,**{k:v for k,v in j.items() if k!='cells'},**d['metrics'],log_evidence=d['log_evidence'],evaluation_count=d['evaluation_count'],invalid_evaluation_count=d['invalid_evaluation_count'],unique_particles=d['unique_particles'],stage_count=len(d['stages']),invocation_seconds=d['invocation_seconds'],device=d['device'],actions=json.dumps(d['actions']),parameter_mean=json.dumps(d['parameter_mean']),parameter_quantiles05_50_95=json.dumps(d['parameter_quantiles05_50_95']),protocol_hash=d['protocol']['manifest_sha256']))
            for stage in d['stages']:stages.append(dict(key=key,**stage))
        csvwrite(report/'continuous-all-fit-metrics.csv',rows);csvwrite(report/'continuous-stage-diagnostics.csv',stages)
        (report/'continuous-report-hashes.json').write_text(json.dumps(hashes,indent=2))
    for version,count in [('joint-field-hierarchy-v02',64),('joint-field-hierarchy-surface-v05',64),('joint-field-temporal-v06',32)]:
        report=ROOT/'reports'/version;rows=[];hashes={}
        for shape in ['block','step']:
          for case in range(count):
            p=report/shape/f'case-{case:03d}.json';d=json.loads(p.read_text());hashes[str(p.relative_to(report))]=hashlib.sha256(p.read_bytes()).hexdigest()
            for r in d['rows']:rows.append(dict(shape=shape,case=case,truth_tau=d['truth_tau'],**r))
        assert len(rows)==2*count*12
        csvwrite(report/'all-case-metrics.csv',rows);(report/'case-report-hashes.json').write_text(json.dumps(hashes,indent=2))
    for version in ['joint-field-identification-v01','joint-field-hierarchy-v02','joint-field-fine-v03','joint-field-surface-v04','joint-field-hierarchy-surface-v05','joint-field-temporal-v06']:
        for p in (ROOT/'reports'/version).rglob('*.csv'):
            p.with_suffix(p.suffix+'.gz').write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    print('Complete compact evidence exported (CSV and reproducible gzip copies)')
if __name__=='__main__':main()
