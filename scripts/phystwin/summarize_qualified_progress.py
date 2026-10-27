"""Compact development-only numerical evidence while v05 is running."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.phystwin.qualified_reference import BASE,ROOT,protocol,checked_compare,write_json


def main():
    spec=protocol();out=ROOT/'reports/phystwin-qualified-support-v05';out.mkdir(parents=True,exist_ok=True)
    audits=[]
    for i in range(len(spec['development'])):
        p=BASE/'qualification'/f'development-{i}.json'
        if p.exists():audits.append(dict(object=i,**json.loads(p.read_text())))
    cross=[]
    for i in [1,11]:
        a=BASE/'tensor-diagnostic'/f'development-{i}'/'n20-initial.npz';b=BASE/'refinements'/f'development-{i}'/'n20-q3-initial.npz'
        if a.exists() and b.exists():
            with np.load(a) as az,np.load(b) as bz:cross.append(dict(object=i,**checked_compare(az,bz,spec,spec['development'][i],'initial')))
    status=dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),development_qualified=sum(r['qualified'] for r in audits),development_total=12,
        bank8_count=len(list((BASE/'banks/n8').glob('bank9-*.npz'))),bank8_total=324,
        independent_space_comparison=cross,development_audits=audits)
    write_json(out/'numerical-progress.json',status)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for i,color in [(1,'#aa4b38'),(7,'#218675'),(9,'#614f98'),(11,'#397bb3')]:
        rows=next((a['spatial'] for a in audits if a['object']==i),[])
        for ax,key,scale in zip(axes,['rms_mm','traction_relative','elastic_J_relative'],[1,100,100]):
            ax.plot([r['fine'] for r in rows],[scale*max(a[key] for a in r['actions']) for r in rows],'-o',label=f'Object {i}',color=color,ms=3)
    for ax,title,gate in zip(axes,['Displacement RMS change (mm)','Support traction change (%)','Elastic energy change (%)'],[.02,2,1]):
        ax.axhline(gate,color='black',ls='--',lw=1,label='Gate');ax.set_yscale('log');ax.set_xlabel('Fine mesh n');ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Worst of four loads, successive graded P2 meshes; development only')
    fig.savefig(out/'development-convergence.png',dpi=150);plt.close(fig)
    lines=['# Development numerical diagnosis','',f"Snapshot: {status['updated_utc']}. {status['development_qualified']}/12 development references qualified; the comparison is not yet complete.",'',
        'The displacement changes become small sooner than the support-force changes. A small position error therefore does not, by itself, validate the inferred support response. The original energy, strain, traction and equilibrium gates remain in force. Failed objects are retained.','',
        '![Development refinement](development-convergence.png)','',
        'An independent Q2 hexahedral space uses the same log-J material and consistent surface Robin law. Basis reproduction, analytic affine energy, energy/force/tangent differentiation, surface integration and an exact full-support translation are tested. At n=20, the initial-load Q2/P2 comparisons are:','',
        '| Development object | Displacement RMS difference (mm) | Traction relative difference | All spatial cross-checks pass |','|---|---:|---:|---|']
    for r in cross:lines.append(f"| {r['object']} | {r['rms_mm']:.6f} | {100*r['traction_relative']:.3f}% | {r['pass']} |")
    lines+=['','This cross-space agreement is useful but does not replace successive-refinement qualification. The stiff object still failed the Q2 n=16-to-20 traction gate; increasing normal-direction grading did not fix it. Q2 remains a diagnostic, not campaign truth. The P2 campaign continues to n=32/40 where needed, serializing these largest meshes to bound memory.','',
        'Reproduce the diagnostic with `.venv/bin/python scripts/phystwin/diagnose_tensor_fixture.py --case 1` (or `--case 11`; optionally `--x-power 2`). Raw fields remain under `runs/phystwin-qualified-support-v05/`. The campaign supervisor records its current stage and logs under `status/`, and stops on failed references or inference checks. No calibration/test result is inferred from this development snapshot.','']
    (out/'NUMERICS.md').write_text('\n'.join(lines));print(json.dumps({k:v for k,v in status.items() if k not in ['development_audits','independent_space_comparison']}))

if __name__=='__main__':main()
