"""Independent Van Loan audit of corrected PAC linear-system integrals.

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
    for case in [0,2,8,10]:
        model,_=make_model(inputs,case);m=model.modes;d=2*m
        rest=model.data['rest']
        delta=rest[model.edges[:,1]]-rest[model.edges[:,0]]
        direction=delta/(delta.square().sum(-1,keepdim=True)+1e-12).sqrt()
        _,_,_,features=model.terms(rest.new_zeros(1,model.dimension))
        amplitude=model.network(features[0,0])/model.constant if model.noise=='neural' else torch.ones(len(model.edges),dtype=rest.dtype)
        # Independently assemble equal/opposite nodal factors, then project.
        nodal=torch.einsum('ne,ec,e->nce',model.incidence,direction,amplitude)
        factor=torch.einsum('ncm,nce->me',model.basis,nodal)*model.internal_scale/math.sqrt(len(model.edges)/len(rest))
        spectral=factor@factor.T+model.acc_noise_floor**2*torch.eye(m,dtype=rest.dtype)
        for p0,p1 in [(-3.,-3.),(0.,0.),(3.,3.),(3.,-3.),(-3.,3.),(9.,-3.),(9.,0.),(9.,3.)]:
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
            q=(q+q.T)/2;root=torch.linalg.cholesky(q)
            difference=torch.linalg.solve_triangular(root,q4[0,0]-q,upper=False)
            difference=torch.linalg.solve_triangular(root,difference.T,upper=False).T
            rows.append({'scene':model.data['scene'],'noise':model.noise,'standardized_log_stiffness':p0,'standardized_log_damping':p1,
                'transition_max_abs_error':float((model.transition(z,model.dt)[0]-f).abs().max()),
                'transition_relative_frobenius_error':float((model.transition(z,model.dt)[0]-f).norm()/f.norm()),
                'noise_covariance_relative_frobenius_error':float((q4[0,0]-q).norm()/q.norm()),
                'noise_covariance_relative_eigen_error':float(torch.linalg.eigvalsh((difference+difference.T)/2).abs().max()),
                'load_mapping_relative_frobenius_error':float((g4[0]-g).norm()/g.norm()),
                'minimum_noise_eigenvalue':float(torch.linalg.eigvalsh(q4[0,0]).min())})
    passed=all(r['transition_relative_frobenius_error']<1e-8 and r['noise_covariance_relative_eigen_error']<1e-7
        and r['load_mapping_relative_frobenius_error']<1e-8 and r['minimum_noise_eigenvalue']>0 for r in rows)
    result={'source_fingerprint':source_fingerprint(),'all_pass':passed,'checks':rows,
        'integrator':'balanced exact modal-pair/affine exponentials',
        'qualification':'independent full-matrix/Van Loan covariance, both scenes/noise types, stiffness -3/0/3/9 and damping -3/0/3; supplied initial geometry only; not posterior exploration or material/contact validation'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'all_pass':passed,'max_covariance_relative_error':max(r['noise_covariance_relative_frobenius_error'] for r in rows),
        'max_load_relative_error':max(r['load_mapping_relative_frobenius_error'] for r in rows)},indent=2))
    if not passed:raise ValueError('Linear propagation audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2);audit(a.inputs,a.output)
