"""Check physical diagnostics on actual exported geometry, not just hidden solves."""
import json,runpy
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import load_phystwin_predictor
from scripts.gaussian_boundary_flow_v02.object_backend import ObjectBoundaryBackend
NoFloorDiagnostic=runpy.run_path('reports/gaussian-boundary-flow-v02/object-backend-audit-reproduce.py')['NoFloorDiagnostic']

def main():
    torch.set_num_threads(2);data=load_phystwin_predictor('single_lift_sloth',64);frames=np.arange(data['train_end'])
    rng=np.random.default_rng(913);zs=[np.zeros(4),*rng.normal(size=(3,4))];rows=[]
    with torch.no_grad():
        for mode,cls in [('on',ObjectBoundaryBackend),('off',NoFloorDiagnostic)]:
            b=cls(data,'cuda:2')
            for i,z in enumerate(zs):
                x,d=b.rollout(z,frames);exported=data['nodes'][None]+x-x[:1];theta=b.bind(z);j=[];r=[];invalid=[]
                for f,p in enumerate(exported):
                    q=b.model.t((p-b.model.origin)/b.model.scale);j.append(b.model.min_jacobian(q))
                    try:
                        _,res,_=b.model.evaluate(q,theta,f);r.append(float(res.norm()))
                    except ValueError as e:invalid.append({'frame':f,'error':str(e)})
                row={'floor':mode,'world':i,'internal_min_J':min(a['minJ']for a in d),'exported_min_J':min(j),'internal_max_residual':max(a['residual']for a in d),'exported_max_residual':max(r)if r else None,'exported_invalid_frames':invalid,'first_equilibrium_offset_RMS_mm':float(np.sqrt(np.mean((x[0]-data['nodes'])**2))*1000)}
                rows.append(row);print(row,flush=True)
    Path('reports/gaussian-boundary-flow-v02/object-rebase-audit-evidence.json').write_text(json.dumps({'scope':'Same four exposed prior worlds; prefix only; exported geometry is nodes+position-position_at_frame0','results':rows},indent=2)+'\n')
if __name__=='__main__':main()
