"""Evaluate genuinely withheld intermediate RGB frames, separately from forecasts."""
from pathlib import Path
import json
import torch
from scripts.gaussian_boundary_flow.model import Block
from scripts.gaussian_boundary_flow.visual import render


def main():
    torch.set_num_threads(2);b=Block();data=torch.load('runs/gaussian-boundary-flow-rgb-v01/evaluation.pt',weights_only=True)
    with torch.no_grad():
        truth=b.rollout(data['truth']);pred=data['prediction']['sine'];rows=[]
        for i in range(0,len(truth),8):
            gt=render(b,truth[i:i+8,:17],size=24)
            ours=render(b,pred[i:i+8,:17],size=24)[:,1::2]
            # Exact known endpoint images; no extra observation noise for either.
            blend=(gt[:,0:-1:2]+gt[:,2::2])/2
            target=gt[:,1::2]
            rows.append(torch.stack([((ours-target)**2).mean((1,2,3,4)),((blend-target)**2).mean((1,2,3,4))],-1))
        mse=torch.cat(rows)
        report={'scope':'Eight withheld mid-interval frames in observed prefix; distinct from forecasting','cases':len(truth),'methods':['physical_flow_mean','linear_RGB_blending'],'PSNR_dB_mean':(-10*mse.clamp_min(1e-12).log10()).mean(0).tolist(),'MSE_per_case':mse.tolist(),'limitation':'Fixed known appearance and camera; no claim of superiority over specialized video interpolation models'}
    Path('reports/gaussian-boundary-flow-v01/interpolation.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items()if k!='MSE_per_case'})

if __name__=='__main__':main()
