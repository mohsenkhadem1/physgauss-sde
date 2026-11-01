"""POST-HOC SECONDARY comparator: one development-trained probe pair per track.

Ignores camera strata; fits only the original four development worlds/repeats.
Then scores previously committed reserved pair outcomes. No forecasts, primary
protocols, acquisition policies or primary scores are changed.
"""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/physical-refinement-v15'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute(development,reserved):
    choices=[];comparisons=[];summaries=[]
    for track in ('V','VF'):
        candidates={}
        for row in development['pair_rows']:
            if row['track']==track and all(name.startswith('probe_') for name in row['sequence']):
                candidates.setdefault(tuple(row['sequence']),[]).append(row)
        if len(candidates)!=6:raise ValueError('All six development probe pairs required')
        losses={}
        for pair,rows in candidates.items():
            if len(rows)!=16 or {(r['world'],r['repeat']) for r in rows}!={(f'development-{i}',j) for i in range(4) for j in range(4)}:raise ValueError('Complete four-world development cohort required')
            losses[pair]=float(np.mean([r['metrics']['normalized_design_mse'] for r in rows]))
        selected=min(losses,key=losses.get) # declaration-order tie break
        choices.append(dict(track=track,sequence=list(selected),development_mean_normalized_loss=losses[selected],
                            candidates=[dict(sequence=list(k),development_mean_normalized_loss=v) for k,v in losses.items()]))
        final=[r for r in reserved['rows'] if r['track']==track and r['is_final']]
        active=[r for r in final if r['policy']=='physical_risk']
        baseline=[r for r in final if r['policy'].startswith('pair-') and tuple(r['sequence'])==selected]
        identities={(r['world'],r['repeat']) for r in active}
        if len(active)!=48 or len(baseline)!=48 or identities!={(r['world'],r['repeat']) for r in baseline} or len({w for w,_ in identities})!=12:raise ValueError('All matched reserved cases required')
        bycase={(r['world'],r['repeat']):r for r in baseline}
        for world in sorted({w for w,_ in identities}):
            a=[r for r in active if r['world']==world];b=[bycase[(r['world'],r['repeat'])] for r in a]
            metrics={}
            for key in a[0]['metrics']:
                am=float(np.mean([r['metrics'][key] for r in a]));bm=float(np.mean([r['metrics'][key] for r in b]));delta=am-bm
                metrics[key]=dict(active=am,global_fixed_probe=bm,active_minus_fixed=delta,
                    outcome='near_tie' if abs(delta)<=1e-12 else ('win' if (delta>0 if key=='normalized_design_utility' else delta<0) else 'loss'))
            comparisons.append(dict(track=track,world=world,initial_camera=a[0]['initial_camera'],sequence=list(selected),metrics=metrics))
        rows=[r for r in comparisons if r['track']==track];summary={}
        for key in rows[0]['metrics']:
            summary[key]=dict(active=float(np.mean([r['metrics'][key]['active'] for r in rows])),
                global_fixed_probe=float(np.mean([r['metrics'][key]['global_fixed_probe'] for r in rows])),
                active_minus_fixed=float(np.mean([r['metrics'][key]['active_minus_fixed'] for r in rows])),
                wins=sum(r['metrics'][key]['outcome']=='win' for r in rows),losses=sum(r['metrics'][key]['outcome']=='loss' for r in rows),near_ties=sum(r['metrics'][key]['outcome']=='near_tie' for r in rows))
        summaries.append(dict(track=track,worlds=12,repeats_per_world=4,metrics=summary))
    return dict(status='POST-HOC SECONDARY comparator; primary results unchanged',scope=__doc__,
                selection='One camera-independent probe pair per V/VF track minimizes mean normalized development design loss across4worlds×4repeats; all6pairs considered',
                absolute_near_tie_tolerance=1e-12,selected_pairs=choices,world_comparisons=comparisons,summary=summaries)


def main():
    dev=REPORT/'development-policy-g17-s97.json';reserved=REPORT/'reserved-scores.json';out=REPORT/'posthoc-global-fixed-probe.json'
    if out.exists():raise ValueError('Preserve previous diagnostic')
    if sha(reserved)!='aaeca76f8883517ea0f3990bd8132c7245b9bb13f9623a12a2a6cb65e4f16949':raise ValueError('Released score hash mismatch')
    d=json.loads(dev.read_text())
    # Pair training uses development data only; compute() accesses reserved scores only after selecting its pair.
    r=json.loads(reserved.read_text());result=compute(d,r)
    result.update(input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (dev,reserved)},source_sha256=sha(__file__),
                  protocol_sha256=r['protocol_sha256'],method_freeze_sha256=r['method_freeze_sha256'])
    out.write_text(json.dumps(result,indent=2)+'\n')
    lines=[result['status'],'']
    for choice,summary in zip(result['selected_pairs'],result['summary']):
        lines.append(choice['track']+' selected '+ ' then '.join(choice['sequence']))
        for key,m in summary['metrics'].items():lines.append(f"  {key}: active {m['active']:.9g}, fixed {m['global_fixed_probe']:.9g}; {m['wins']} wins/{m['losses']} losses/{m['near_ties']} near-ties")
    out.with_suffix('.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
