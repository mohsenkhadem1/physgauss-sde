"""Post-hoc nominal contact-integration diagnostic for the reduced transfer model.

After the first poor external-data result, compare its zero-noise/zero-load
frame-held contact drift with a converged RK4 solve of the SAME reduced ODE.
This is not the original MPM truth and does not alter the frozen experiment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pacnerf_comparison import make_model
from physgauss.cli import source_fingerprint


def audit(inputs,output):
    rows=[]
    for index in [0,8]:
        model,_=make_model(inputs,index);m=model.modes
        basis=model.basis.numpy();rest=model.data['rest'].numpy();eigen=model.eigen.numpy();dt=1/24
        def rhs(state):
            q,v=state[:m],state[m:]
            height=rest[:,1]+basis[:,1]@q
            nodal=-9.8+6*np.logaddexp(0,-height/.01)
            return np.concatenate((v,-eigen*q-.8*v+basis[:,1].T@nodal))
        def reference(substeps):
            state=np.zeros(2*m);values=[state.copy()];h=dt/substeps
            for _ in range(13):
                for _ in range(substeps):
                    a=rhs(state);b=rhs(state+h*a/2);c=rhs(state+h*b/2);d=rhs(state+h*c)
                    state=state+h*(a+2*b+2*c+d)/6
                values.append(state.copy())
            return np.array(values)
        fine,finer=reference(192),reference(384)
        z=torch.zeros(1,model.dimension,dtype=torch.float64)
        f=model.transition(z,dt)[0].numpy();drive=np.zeros((2*m,m))
        for point,weight in zip([-.8611363115940526,-.3399810435848563,.3399810435848563,.8611363115940526],
            [.3478548451374538,.6521451548625461,.6521451548625461,.3478548451374538]):
            drive+=weight*dt/2*model.transition(z,dt*(point+1)/2)[0,:,m:].numpy()
        state=np.zeros(2*m);coarse=[state.copy()]
        for _ in range(13):
            height=rest[:,1]+basis[:,1]@state[:m]
            external=basis[:,1].T@(-9.8+6*np.logaddexp(0,-height/.01))
            state=f@state+drive@external;coarse.append(state.copy())
        phi=model.renderer.gaussian_basis.numpy();coarse=np.array(coarse)
        error=np.einsum('tm,gcm->tgc',coarse[:,:m]-finer[:,:m],phi)
        ref_error=np.einsum('tm,gcm->tgc',fine[:,:m]-finer[:,:m],phi)
        rows.append({'scene':model.data['scene'],'nominal_standardized_parameters':[0.,0.],
            'zero_noise_zero_applied_load':True,
            'coarse_vs_rk4_384_all_coordinate_rmse_m':float(np.sqrt((error**2).mean())),
            'coarse_vs_rk4_384_gap4_coordinate_rmse_m':float(np.sqrt((error[4:10]**2).mean())),
            'coarse_vs_rk4_384_gap6_coordinate_rmse_m':float(np.sqrt((error[5:9]**2).mean())),
            'reference192_vs384_coordinate_rmse_m':float(np.sqrt((ref_error**2).mean())),
            'maximum_coarse_coordinate_error_m':float(np.abs(error).max())})
    result={'source_fingerprint':source_fingerprint(),'checks':rows,
        'qualification':'post-hoc zero-noise nominal reduced mean ODE diagnostic, not MPM validation or an error decomposition of the posterior; experiment unchanged',
        'finding':'frame-held contact is a material numerical approximation; the accurate linear propagator does not guarantee accurate contact motion'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2);audit(a.inputs,a.output)
