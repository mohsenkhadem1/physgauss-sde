"""Independent Van Loan audit of PAC-NeRF linear propagation/noise quadrature.

Uses only supplied initial geometry. This does NOT validate the reduced material
law or held-contact approximation against the original MPM simulation.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from pacnerf_comparison import make_model
from physgauss.cli import source_fingerprint


@torch.no_grad()
def audit(inputs, output):
    rows=[]
    for case in [0,8]:
        model,_=make_model(inputs,case);m=model.modes;d=2*m
        rest=model.data['rest']
        direction=(rest[model.edges[:,1]]-rest[model.edges[:,0]])/model.length[:,None]
        # Independently assemble equal/opposite nodal factors, then project.
        nodal=torch.einsum('ne,ec->nce',model.incidence,direction)
        factor=torch.einsum('ncm,nce->me',model.basis,nodal)*model.internal_scale/math.sqrt(len(model.edges)/len(rest))
        spectral=factor@factor.T+model.acc_noise_floor**2*torch.eye(m,dtype=rest.dtype)
        for p0,p1 in [(-3.,-3.),(0.,0.),(3.,3.),(3.,-3.),(-3.,3.)]:
            z=rest.new_zeros(1,model.dimension);z[0,:2]=z.new_tensor([p0,p1])
            a=rest.new_zeros(d,d);a[:m,m:]=torch.eye(m,dtype=rest.dtype)
            a[m:,:m]=-torch.diag(model.eigen*math.exp(.7*p0))
            a[m:,m:]=-torch.eye(m,dtype=rest.dtype)*.8*math.exp(.7*p1)
            f=torch.matrix_exp(a*model.dt)
            van_loan=rest.new_zeros(2*d,2*d)
            van_loan[:d,:d]=a;van_loan[d:,d:]=-a.T
            van_loan[m:d,d+m:]=spectral
            block=torch.matrix_exp(van_loan*model.dt)
            q=block[:d,d:]@f.T
            affine=rest.new_zeros(d+m,d+m);affine[:d,:d]=a
            affine[m:d,d:]=torch.eye(m,dtype=rest.dtype)
            drive=torch.matrix_exp(affine*model.dt)[:d,d:]
            _,q4,g4,_=model.terms(z)
            g=drive[None]@model.acceleration_mapping
            rows.append({'scene':model.data['scene'],'standardized_log_stiffness':p0,'standardized_log_damping':p1,
                'transition_max_abs_error':float((model.transition(z,model.dt)[0]-f).abs().max()),
                'noise_covariance_relative_frobenius_error':float((q4[0,0]-q).norm()/q.norm()),
                'load_mapping_relative_frobenius_error':float((g4[0]-g).norm()/g.norm()),
                'minimum_noise_eigenvalue':float(torch.linalg.eigvalsh(q4[0,0]).min())})
    passed=all(r['transition_max_abs_error']<1e-10 and r['noise_covariance_relative_frobenius_error']<1e-6
        and r['load_mapping_relative_frobenius_error']<1e-7 and r['minimum_noise_eigenvalue']>0 for r in rows)
    result={'source_fingerprint':source_fingerprint(),'all_pass':passed,'checks':rows,
        'qualification':'independent full-matrix exponential and Van Loan covariance; parameter range +/-3 prior standardized units; initial geometry only; not material/contact-model validation'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_pass':passed,'max_covariance_relative_error':max(r['noise_covariance_relative_frobenius_error'] for r in rows),
        'max_load_relative_error':max(r['load_mapping_relative_frobenius_error'] for r in rows)},indent=2))
    if not passed:raise ValueError('Linear propagation audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2);audit(a.inputs,a.output)
