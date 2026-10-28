"""Development-only 64-versus-256 outcome-sampling audit; no policy retuning."""
from pathlib import Path
import json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.phystwin.v08_study import BASE,OUT,Experiment,interpolators,fit,protocol,select,write
while len(list((BASE/'fields').glob('bank*-n4-q3.npz')))<192 or not (BASE/'development-observations.npz').exists():time.sleep(10)
s=protocol();interps=interpolators();bias,C=fit(interps);ex=Experiment(interps,65,bias,C);z=np.load(BASE/'development-observations.npz');rows=[]
for obj in range(6):
 y=z['measurements'][obj*3];low,current=ex.choose([],y,64);high,_=ex.choose([],y,256);a=select(low,current,s['costs'],s['cost_penalty_m2']);b=select(high,current,s['costs'],s['cost_penalty_m2']);rows.append(dict(object=obj,choice64=a,choice256=b,agreement=a==b,risk64_m2=low.tolist(),risk256_m2=high.tolist(),current_variance_m2=current));print('UTILITY_AUDIT',obj,a,b,flush=True)
write(OUT/'utility-sampling.json',dict(scope='First six development objects, first noise repeat, initial decision only; 65 cubed quadrature fixed. 64 versus 256 Sobol outcomes. Diagnostics only; primary policy remains frozen at 64 outcomes. Not a full numerical convergence proof.',agreements=sum(r['agreement'] for r in rows),decisions=len(rows),rows=rows))
