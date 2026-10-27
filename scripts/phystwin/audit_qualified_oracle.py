"""Post-test privileged action audit; never changes a saved policy or its seal."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from scripts.phystwin.qualified_reference import BASE,protocol,write_json
from scripts.phystwin.qualified_policy import OUT,interpolators,fit,observations,verify_seal
from scripts.phystwin.active_support_policy import Experiment


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args()
    spec=protocol();seal=json.loads((OUT/'seal.json').read_text());verify_seal(seal)
    summary=json.loads((OUT/'test-summary.json').read_text());model=json.loads((OUT/'model-selection.json').read_text())['model']
    interp=interpolators(model);mu,C,_=fit(spec,interp,model);ex=Experiment(interp,summary['grid_size'],mu,C,spec['noise_sigma_m'],a.device)
    ys,truth=observations(spec,'test');saved=np.load(OUT/'test-predictions.npz');z=ndtri(qmc.Sobol(3,scramble=True,seed=8832).random_base2(9));risks=[];low=[]
    for i,y0 in enumerate(ys[0]):
        f=np.load(BASE/'fields'/f'test-{i//spec["noise_repeats"]}-qualified.npz')['fields'].ravel();r=[];c=[]
        for j,probe in enumerate(ex.probes):
            y=np.c_[np.broadcast_to(y0,(len(z),len(y0))),f[probe]+spec['noise_sigma_m']*z]
            errors=np.mean((ex.models[j+1].infer(y)['mean']-truth[i])**2,axis=1);r.append(float(errors.mean()));c.append(float(errors[:128].mean()))
        risks.append(r);low.append(c)
        if i%8==0:print('ORACLE',i,len(ys[0]),flush=True)
    risks=np.array(risks);low=np.array(low);oracle=risks.argmin(1);rows=np.arange(len(risks))
    choices=dict(fixed=np.full(len(rows),seal['fixed_action']-1),adaptive=saved['adaptive']-1,fixture_side_heuristic=saved['heuristic']-1,oracle=oracle)
    scores={k:dict(expected_rmse_m=float(np.sqrt(risks[rows,c].mean())),mean_excess_mse_over_oracle=float(np.mean(risks[rows,c]-risks.min(1))),oracle_choice_agreement=float(np.mean(c==oracle))) for k,c in choices.items()}
    scores['random']=dict(expected_rmse_m=float(np.sqrt(risks.mean())),mean_excess_mse_over_oracle=float(np.mean(risks.mean(1)-risks.min(1))))
    write_json(OUT/'oracle-audit.json',dict(scope='Privileged diagnostic using the true held object response to integrate future measurement noise. Not an eligible policy, not used for tuning, and not directly comparable to realized-outcome RMSE. Oracle gaps include hidden-parameter uncertainty, not just algorithm error.',samples=512,choice_refinement_agreement=int(np.sum(low.argmin(1)==oracle)),trials=len(rows),maximum_risk_refinement_difference_m2=float(abs(low-risks).max()),scores=scores,risks=risks.tolist()))
    verify_seal(seal);print('COMPLETE',json.dumps(scores),flush=True)

if __name__=='__main__':main()
