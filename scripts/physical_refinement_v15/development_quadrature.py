"""Development-only posterior quadrature comparison on all fixed query histories."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,itertools,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.physical_refinement_v15.banks import BASE,ROOT,assemble,expand,mechanical_design
from scripts.physical_refinement_v15.development_data import development_worlds,menu,prediction_queries,observation
from scripts.physical_refinement_v15.posterior import Likelihood,posterior,scale_summary


def compute(bank,gradient_count,scale_count):
    expanded=expand(bank,gradient_count,scale_count)
    design=mechanical_design(expanded)
    rows=[];weights=[];evidence={}
    for world in development_worlds():
        for track in ('V','VF'):
            initial,queries=prediction_queries(expanded,world['initial_camera'],track)
            likelihoods={q.name:Likelihood(q) for q in [initial]+queries}
            for repeat in range(4):
                y,h=observation(world['name'],repeat,'initial',world['initial_camera'],initial);evidence.update(h)
                prefix=likelihoods[initial.name].log_density(y)
                terms={}
                for q,(name,action,camera) in zip(queries,menu(world['initial_camera'])):
                    y,h=observation(world['name'],repeat,action,camera,q);evidence.update(h)
                    terms[name]=likelihoods[name].log_density(y)
                histories=[()]+[(q.name,) for q in queries]+list(itertools.combinations([q.name for q in queries],2))
                for history in histories:
                    w,z=posterior(expanded['weights'],[prefix]+[terms[name] for name in history])
                    rows.append(dict(world=world['name'],track=track,repeat=repeat,history=list(history),
                                     log_evidence=z,scale=scale_summary(expanded['rows'],w)))
                    weights.append(w)
    weights=np.asarray(weights)
    mean=weights@design.values
    variance=np.maximum(weights@(design.values**2)-mean**2,0.)
    return rows,mean,np.sqrt(variance),evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gradient-count',type=int,default=5)
    parser.add_argument('--coarse',type=int,default=25);parser.add_argument('--fine',type=int,default=49)
    args=parser.parse_args();bank=assemble(args.gradient_count)
    out=BASE/'posterior-quadrature'/f'g{args.gradient_count}-s{args.coarse}-{args.fine}'
    if out.exists():raise ValueError('Preserve earlier quadrature run; do not overwrite')
    out.mkdir(parents=True)
    results=[]
    for count in (args.coarse,args.fine):
        rows,mean,sd,evidence=compute(bank,args.gradient_count,count)
        np.savez_compressed(out/f's{count}.npz',mean=mean,sd=sd)
        (out/f's{count}.json').write_text(json.dumps(dict(rows=rows,evidence_sha256=evidence),indent=2)+'\n')
        results.append((rows,mean,sd))
        print('completed scale quadrature',count,'histories',len(rows),flush=True)
    (a,am,asd),(b,bm,bsd)=results
    if [{k:v for k,v in r.items() if k not in ('log_evidence','scale')} for r in a] != [{k:v for k,v in r.items() if k not in ('log_evidence','scale')} for r in b]:
        raise ValueError('Quadrature comparisons must use identical outcomes/histories')
    # Features per action:81x3aligned displacement,3force,3torque.
    delta=np.abs(am-bm).reshape(len(a),3,249)
    std_delta=np.abs(asd-bsd).reshape(len(a),3,249)
    reference_sd=bsd.reshape(len(a),3,249)
    checks=dict(shape_mean=bool((delta[:,:,:243]<=.00005).all()),
                shape_sd=bool((std_delta[:,:,:243]<=.00001+.05*reference_sd[:,:,:243]).all()),
                force_mean=bool((delta[:,:,243:246]<=.001).all()),
                torque_mean=bool((delta[:,:,246:]<=.0001).all()),
                force_sd=bool((std_delta[:,:,243:246]<=.0001+.05*reference_sd[:,:,243:246]).all()),
                torque_sd=bool((std_delta[:,:,246:]<=.00001+.05*reference_sd[:,:,246:]).all()),
                log_evidence=bool(max(abs(x['log_evidence']-y['log_evidence']) for x,y in zip(a,b))<=.01),
                scale_mean=bool(max(abs(x['scale']['mean']-y['scale']['mean']) for x,y in zip(a,b))<=.005),
                scale_sd=bool(max(abs(x['scale']['sd']-y['scale']['sd']) for x,y in zip(a,b))<=.005),
                scale_quantiles=bool(max(max(abs(np.array(x['scale']['quantiles_05_50_95'])-np.array(y['scale']['quantiles_05_50_95']))) for x,y in zip(a,b))<=.01))
    report=dict(scope='Development fixed-observation posterior summaries only; policy-score and gradient refinement remain separate',gradient_count=args.gradient_count,
                scale_counts=[args.coarse,args.fine],histories=len(a),checks=checks,passed=all(checks.values()),
                maximum_shape_mean_difference_mm=float(1000*delta[:,:,:243].max()),
                maximum_force_mean_difference_N=float(delta[:,:,243:246].max()),maximum_torque_mean_difference_Nm=float(delta[:,:,246:].max()),
                maximum_log_evidence_difference=max(abs(x['log_evidence']-y['log_evidence']) for x,y in zip(a,b)))
    path=ROOT/'reports/physical-refinement-v15'/f'posterior-scale-g{args.gradient_count}-{args.coarse}-{args.fine}.json'
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
