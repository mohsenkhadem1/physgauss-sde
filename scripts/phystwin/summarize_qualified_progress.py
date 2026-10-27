"""Development-only numerical evidence while the v05 inference campaign runs."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.qualified_reference import BASE,ROOT,protocol,write_json


def main():
    spec=protocol();out=ROOT/'reports/phystwin-qualified-support-v05';out.mkdir(parents=True,exist_ok=True)
    audits=[];prior=[];plot_audits=[]
    for i in range(len(spec['development'])):
        p=BASE/'qualification'/f'development-{i}.json'
        if not p.exists():continue
        a=dict(object=i,**json.loads(p.read_text()));audits.append(a)
        old=[dict(object=i,**json.loads((BASE/q).read_text())) for q in a.get('prior_p2_attempts',[])]
        prior.extend(old)
        if a.get('space','P2')=='P2':plot_audits.append(a)
        elif old:plot_audits.append(max(old,key=lambda r:r['latest_mesh']))
    validation=json.loads((BASE/'tensor-validation.json').read_text()) if (BASE/'tensor-validation.json').exists() else None
    status=dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),development_qualified=sum(r['qualified'] for r in audits),development_total=12,
        bank8_count=len(list((BASE/'banks/n8').glob('bank9-*.npz'))),bank12_count=len(list((BASE/'banks/n12').glob('bank9-*.npz'))),bank_total=324,
        tensor_method_enabled=(BASE/'tensor-reference-adoption.json').exists(),development_audits=audits,retained_p2_attempts=prior)
    write_json(out/'numerical-progress.json',status)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for i,color in [(1,'#aa4b38'),(7,'#218675'),(9,'#614f98'),(11,'#397bb3')]:
        rows=next((a['spatial'] for a in plot_audits if a['object']==i),[])
        for ax,key,scale in zip(axes,['rms_mm','traction_relative','elastic_J_relative'],[1,100,100]):
            ax.plot([r['fine'] for r in rows],[scale*max(a[key] for a in r['actions']) for r in rows],'-o',label=f'Object {i}',color=color,ms=3)
    for ax,title,gate in zip(axes,['Displacement RMS change (mm)','Support traction change (%)','Elastic energy change (%)'],[.02,2,1]):
        ax.axhline(gate,color='black',ls='--',lw=1,label='Gate');ax.set_yscale('log');ax.set_xlabel('Fine mesh n');ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Retained P2 attempts: worst of four loads, development only')
    fig.savefig(out/'development-convergence.png',dpi=150);plt.close(fig)
    lines=['# Development numerical diagnosis','',f"Snapshot: {status['updated_utc']}. {status['development_qualified']}/12 development references qualified. This snapshot does not claim a completed held-out comparison.",'',
        'The P2 displacement changes became small sooner than support-force changes. Small position error alone did not validate the support response. The figure retains the original P2 attempts, including failures. Accuracy gates were not loosened.','',
        '![Retained P2 development refinements](development-convergence.png)','',
        'The validated replacement uses Q2 hexahedra concentrated near support edges: x-power-2 grading, cosine grading toward y edges and the z support cut/outer edges, with four times as many z cells. It uses the same log-J material and consistent Robin surface law. The candidate sequence n=6,10,14,18,22 retains two consecutive spatial passes and q3/q4 checks under all four loads. Basis reproduction, affine energy, energy/force/tangent differentiation, surface integration, and exact full-support translation are tested.','']
    if validation and validation['passed']:
        lines+=['Both required development fixtures qualified. The independent comparison against the previously qualified P2 reference for development object 11 is:','',
            '| Load | Displacement RMS difference (mm) | Traction relative difference | All cross-checks pass |','|---|---:|---:|---|']
        for r in validation['qualified_p2_cross_comparison']:lines.append(f"| {r['action']} | {r['rms_mm']:.6f} | {100*r['traction_relative']:.3f}% | {r['pass']} |")
        r=validation['additional_finer_initial_check']
        lines+=['',f"An additional Q2 n=18-to-20 check on the stiff object's initial load also passes: RMS difference {r['rms_mm']:.8f} mm, traction difference {100*r['traction_relative']:.3f}%. This additional check covers one load; the accepted reference separately passed all four loads and quadrature checks.",'',
            'The numerical method is enabled only through the hashed [validation](tensor-validation.json) and [adoption rule](tensor-reference-adoption.json). Already-qualified P2 fields are preserved. New half-face references use edge-refined Q2; full-face references retain P2 with qualified Q2 fallback. This is simulator meshing, not disclosure of the true fixture to inference or probing policies.','']
    else:lines+=['Candidate validation is incomplete. Q2 is not yet enabled as campaign truth.','']
    lines+=['The P2 n=40 continuation was stopped after n=32 failed: a single final pass could not satisfy the two-pass rule. The separate finer Q2 run was retained through its completed initial-load check, then stopped before an unnecessary second load. Completed fields and failure audits remain local.','',
        'Reproduce candidate qualification with `.venv/bin/python scripts/phystwin/qualified_tensor_reference.py --index 1` and `--index 11`; enable it with `validate_tensor_reference.py` only after its checks pass. The additional finer diagnostic is available through `diagnose_tensor_fixture.py --case 1 --edge-grading --levels 12 16 20 --actions initial`. Raw fields remain under `runs/phystwin-qualified-support-v05/`. Supervisor status and logs are under `status/`.','',
        'These are finite numerical checks, not absolute continuum-error bounds or validation against real measurements. Calibration/test results do not select this method.','']
    (out/'NUMERICS.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in status.items() if k not in ['development_audits','retained_p2_attempts']}))

if __name__=='__main__':main()
