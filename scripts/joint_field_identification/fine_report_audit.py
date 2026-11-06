"""Independent paired-record and aggregation check for completed v03 reference."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/joint-field-fine-v03'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    summary=json.loads((REPORT/'summary.json').read_text()); records={};sources={};maxidentity=0.
    for shape in ('block','step'):
      truth=ROOT/f'runs/joint-field-fine-v03/{shape}/fine-truth.npz';th=sha(truth)
      for model in ('coarse8','fine24'):
        bank=ROOT/(f'runs/joint-field-identification-v01/{shape}.npz' if model=='coarse8' else f'runs/joint-field-fine-v03/{shape}.npz');bh=sha(bank)
        for case in range(16):
          path=REPORT/f'evaluation/{shape}/{model}/case-{case:03d}.json';d=json.loads(path.read_text());sources[str(path.relative_to(ROOT))]=sha(path)
          assert (d['shape'],d['model'],d['case'])==(shape,model,case)
          assert d['truth_sha256']==th and d['bank_sha256']==bh
          assert d['evaluation_plan_sha256']==sha(REPORT/'evaluation-plan.json')
          seen=set()
          for r in d['rows']:
            k=(shape,case,model,r['policy'],r['step']);assert k not in seen;seen.add(k);records[k]=r
            err=abs(r['rmse_mm']**2-.5*(r['hidden_rmse_mm']**2+r['changed_load_rmse_mm']**2));maxidentity=max(maxidentity,err);assert err<1e-10
          assert len(seen)==30
    for shape in ('block','step'):
      for case in range(16):
        for policy in ('fixed','fixed-view','fixed-probe','random'):
          for step in range(1,5):
            assert records[(shape,case,'coarse8',policy,step)]['action']==records[(shape,case,'fine24',policy,step)]['action']
    errors=[]
    for item in summary['summary']:
      rows=[records[(s,c,item['model'],item['policy'],item['step'])] for s in ('block','step') for c in range(16)]
      assert item['n']==len(rows)==32
      for key,value in item.items():
        if key not in ('model','policy','step','n'):errors.append(abs(value-sum(r[key] for r in rows)/32))
    rng=np.random.default_rng(190001);ib=rng.integers(0,16,(4000,16));is_=rng.integers(0,16,(4000,16));cierrors=[]
    for pair in summary['paired']:
      delta={}
      for s in ('block','step'):
        delta[s]=np.array([records[(s,c,'fine24',pair['policy'],pair['step'])][pair['metric']]-records[(s,c,'coarse8',pair['policy'],pair['step'])][pair['metric']] for c in range(16)])
      estimate=.5*(delta['block'].mean()+delta['step'].mean());assert abs(estimate-pair['fine_minus_coarse'])<1e-12
      boots=.5*(delta['block'][ib].mean(axis=1)+delta['step'][is_].mean(axis=1));ci=np.quantile(boots,[.025,.975]);cierrors.append(float(np.max(abs(ci-pair['ci95']))))
    assert max(errors)<1e-12 and max(cierrors)<1e-12
    out=dict(passed=True,case_documents=64,independent_truth_cases=32,model_comparisons=2,raw_rows=len(records),max_aggregate_difference=max(errors),max_paired_bootstrap_CI_difference=max(cierrors),max_target_equal_group_MSE_identity_mm2=maxidentity,summary_sha256=sha(REPORT/'summary.json'),sources=sources,scope='Same truth and potential noise are paired by shape/case. Fixed-policy evidence is shared; active policies can differ. Bootstrap preserves16cases in each fixed geometry. This is not population-of-shapes inference.')
    (REPORT/'aggregation-audit.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k!='sources'})
if __name__=='__main__':main()
