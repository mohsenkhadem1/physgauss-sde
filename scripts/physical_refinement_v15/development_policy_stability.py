"""Limited fantasy stability audit at four worlds/two tracks/repeat zero.
No policies retuned: replay the original two decision histories, score256
fantasies under original and independent seed, compare choices and MCSE.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.physical_refinement_v15 import development_policy as d


def main():
    path=d.ROOT/'reports/physical-refinement-v15/development-policy-g17-s97.json'
    output=d.ROOT/'reports/physical-refinement-v15/development-policy-fantasy-stability.json'
    if output.exists():raise ValueError('Preserve previous audit')
    report=json.loads(path.read_text());source=dict(report['source_sha256']);source[str(Path(__file__).relative_to(d.ROOT))]=d.sha(__file__)
    if any(d.sha(d.ROOT/k)!=v for k,v in source.items()):raise ValueError('Source changed')
    expanded=d.expand(d.assemble(17),17,97);design=d.mechanical_design(expanded);rows=[];evidence={}
    for world in d.development_worlds():
        for track in ('V','VF'):
            saved=next(r for r in report['rows'] if r['world']==world['name'] and r['track']==track and r['repeat']==0 and r['policy']=='physical_risk')
            initial,queries=d.prediction_queries(expanded,world['initial_camera'],track)
            y,h=d.observation(world['name'],0,'initial',world['initial_camera'],initial);evidence.update(h)
            prefix=d.Likelihood(initial).log_density(y);terms={}
            for q,(name,action,camera) in zip(queries,d.menu(world['initial_camera'])):
                if name in saved['sequence']:
                    y,h=d.observation(world['name'],0,action,camera,q);evidence.update(h);terms[name]=d.Likelihood(q).log_density(y)
            for step in range(2):
                history=saved['sequence'][:step];weights=d.update(expanded['weights'],prefix,terms,history)
                for offset in (0,1000000):
                    selected,info=d.choose(weights,queries,design,'physical_risk',used=history,seed=saved['seed']+step+offset,fantasies=256)
                    rows.append(dict(world=world['name'],track=track,repeat=0,step=step+1,history=history,
                                     original_selected=saved['sequence'][step],selected=selected,agrees=selected==saved['sequence'][step],
                                     seed=saved['seed']+step+offset,fantasies=256,acquisition=info))
            print(world['name'],track,'stability complete',flush=True)
    if any(d.sha(d.ROOT/k)!=v for k,v in source.items()):raise ValueError('Sources changed')
    output.write_text(json.dumps(dict(scope=__doc__,input_report_sha256=d.sha(path),source_sha256=source,evidence_sha256=evidence,
                                     comparisons=len(rows),agreements=sum(r['agrees'] for r in rows),rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
